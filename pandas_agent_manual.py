"""
PANDAS AGENT MANUAL - Core RAG Engine

Manual implementation of a Pandas Agent using native OpenAI library.
This agent:
- Interprets questions in natural language
- Generates Python code to query dataframes
- Executes code on real NIST data
- Follows the ReAct format (Reasoning + Acting)
- Exposes reasoning tokens for transparency

This is the CORE of the dynamic RAG system for compliance dialogue.
"""
import os
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
from openai import OpenAI


class ManualPandasAgent:
    """
    Manual Pandas Agent with full control and transparency.
    
    Uses the native openai library (not LangChain) to:
    - Have full access to reasoning_content
    - Have complete control over the ReAct flow
    - Provide detailed debugging for each step
    """
    
    def __init__(
        self,
        base_url: str = "https://gpustack.ing.unibs.it/v1",
        model_name: str = "gpt-oss",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        verbose: bool = True
    ):
        """
        Initializes the manual Pandas Agent.
        
        Args:
            base_url: URL for the UniBS LLM cluster
            model_name: Model to use
            api_key: API key (default from env GPUSTACK_API_KEY)
            temperature: LLM Temperature (0.0 = deterministic)
            verbose: If True, prints all details
        """
        if api_key is None:
            api_key = os.environ.get(
                "GPUSTACK_API_KEY",
                "gpustack_8a00037ab4220858_6479d07686028e2f970357b1a81200e4"
            )
        
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        
        if verbose:
            print(f"✅ ManualPandasAgent initialized")
            print(f"   Model: {model_name}")
            print(f"   Temperature: {temperature}")
            print(f"   Base URL: {base_url}")
    
    def _create_prompt(self, df: pd.DataFrame, question: str) -> str:
        """
        Creates the ReAct-style prompt to query a dataframe.
        
        Args:
            df: DataFrame to query
            question: Natural language question
            
        Returns:
            Formatted prompt for the model
        """
        # Dataframe info, TO DO: depending on the profiling step or question, tell it which database to query
        df_info = f"""You have access to a pandas dataframe called `df` with the following structure:

Shape: {df.shape[0]} rows × {df.shape[1]} columns
Columns: {list(df.columns)}

First 3 rows as reference:
{df.head(3).to_string()}

"""
        
        # Add category information if 'Category' column is present, this is only valid for the first two csv files
        if 'Category' in df.columns:
            unique_categories = df['Category'].unique()
            categories_sample = list(unique_categories[:10])  # First 10 categories
            df_info += f"""
To search by topic (e.g., "data protection", "access control"), use the Category column.
Example: df[df['Category'].str.contains('Data Security', case=False)]
"""
        
        # ReAct Instructions
        instructions = """To answer questions about this dataframe, follow this format STRICTLY:

Thought: [Describe what you need to do to answer the question]
Action: python
Action Input: [Write ONE line of Python code using the variable `df`]

After you provide the Action Input, I will execute it and return:
Observation: [execution result]

You can then either:
1. Use another Thought/Action/Action Input cycle if needed
2. Or provide the final answer with: Final Answer: [your answer]

IMPORTANT RULES:
- Use ONLY the variable name `df` for the dataframe
- Write simple, single-line Python expressions
- Valid examples: len(df), df['Column'].unique(), df[df['X'] > 5]
- Do NOT use complex multi-line code or loops
- Be precise and concise

"""
        
        prompt = df_info + instructions + f"\nQuestion: {question}\n\nBegin:\n"
        
        return prompt
    
    def _parse_response(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parses the model's response to extract Action Input or Final Answer.
        
        Args:
            content: Content of the response (can be None)
            
        Returns:
            Tuple (code_to_execute, final_answer)
            - code_to_execute: Python code to execute (if found)
            - final_answer: Final answer (if found)
        """
        # Handle None or empty case
        if content is None or content == "":
            return None, None
        
        # Check for Final Answer (explicit)
        if "Final Answer:" in content:
            final_answer = content.split("Final Answer:")[-1].strip()
            return None, final_answer
        
        # Check for implicit final answer (table, formatted list, narrative answer)
        # If there's no "Action:" or "Thought:" and the content looks like an answer
        content_lower = content.lower()
        has_react_keywords = any(kw in content for kw in ["Action:", "Action Input:", "Thought:", "Observation:"])
        
        # If it doesn't have ReAct keywords and looks formatted (markdown table, list, paragraph)
        if not has_react_keywords:
            # Short answer without ReAct keyword → almost certainly it's the direct answer
            # Ex: the model answers only "CM-8" after having already executed the code
            if len(content.strip()) < 200:
                return None, content.strip()

            # Look for final answer indicators for longer answers
            final_indicators = [
                "subcategories related to",
                "| subcategory_id |",  # Markdown table header
                "these are all",
                "the following",
                "here are the",
                "subcategory_id"
            ]
            
            if any(indicator in content_lower for indicator in final_indicators):
                # Probably it's a final answer
                return None, content.strip()
        
        # Look for Action Input
        if "Action Input:" in content:
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if "Action Input:" in line:
                    # Code can be on the same line or the line below
                    code = line.split("Action Input:")[-1].strip()
                    if not code and i + 1 < len(lines):
                        code = lines[i + 1].strip()
                    
                    # Remove any backticks
                    code = code.strip('`').strip()
                    
                    if code:
                        return code, None
        
        return None, None
    
    def _execute_code(self, code: str, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Executes Python code safely (with eval).
        
        Args:
            code: Python code to execute
            df: DataFrame to execute on
            
        Returns:
            Tuple (success, result_or_error)
        """
        try:
            # Execute in controlled environment
            result = eval(code, {"df": df, "pd": pd, "len": len, "str": str})

            # Avoid the "..." truncation typical of pandas representation.
            # This way the JSON log contains the complete execution content.
            if isinstance(result, pd.DataFrame):
                return True, result.to_string(index=True, max_colwidth=None)

            if isinstance(result, pd.Series):
                return True, result.to_string(max_rows=None)

            return True, str(result)
        except Exception as e:
            return False, f"Error: {type(e).__name__}: {e}"
    
    def _save_log_json(
        self,
        filepath: str,
        question: str,
        result: Dict,
        df: pd.DataFrame,
        initial_prompt: Optional[str] = None
    ):
        """
        Saves the complete query log to a JSON file.
        
        Args:
            filepath: Path of the JSON file where to save
            question: Original question
            result: Query result (with history)
            df: Used DataFrame
        """
        import json
        from datetime import datetime
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'initial_prompt': initial_prompt,
            'dataframe_info': {
                'shape': list(df.shape),
                'columns': list(df.columns)
            },
            'model': self.model_name,
            'temperature': self.temperature,
            'result': {
                'success': result['success'],
                'answer': result['answer'],
                'iterations': result['iterations']
            },
            'history': result['history']
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"\n💾 Log saved to: {filepath}")
    
    def query(
        self,
        df: pd.DataFrame,
        question: str,
        max_iterations: int = 5,
        log_to_json: Optional[str] = None
    ) -> Dict:
        """
        Executes a query on a dataframe in natural language.
        
        This is the main method that implements the ReAct cycle:
        1. Sends prompt to the model
        2. Model responds with Thought/Action/Action Input
        3. Executes the Python code
        4. Returns Observation to the model
        5. Repeats until Final Answer or max iterations
        
        Args:
            df: DataFrame to query
            question: Natural language question
            max_iterations: Max number of ReAct iterations
            log_to_json: If specified, saves complete log in this JSON file
            
        Returns:
            Dict with:
                - answer: Final answer
                - success: True if completed successfully
                - iterations: Number of iterations used
                - history: Complete conversation history for debugging
        """
        if self.verbose:
            print("\n" + "="*80)
            print("🔍 MANUAL PANDAS AGENT - QUERY START")
            print("="*80)
            print(f"Question: {question}")
            print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Create initial prompt
        initial_prompt = self._create_prompt(df, question)
        
        if self.verbose:
            print("\n📤 INITIAL PROMPT:")
            print("-"*80)
            print(initial_prompt[:500] + "..." if len(initial_prompt) > 500 else initial_prompt)
            print("-"*80)
        
        # Initialize conversation
        messages = [
            {
                "role": "system",
                "content": "You are a Python data analysis assistant. Follow the ReAct format strictly. Be concise."
            },
            {
                "role": "user",
                "content": initial_prompt
            }
        ]
        
        history = []  # For complete tracking
        
        # ReAct Cycle
        for iteration in range(max_iterations):
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"🔄 ITERATION {iteration + 1}/{max_iterations}")
                print("="*80)
            
            # Call LLM
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=2048
                )
                
                # Defensively catch providers returning `choices: null` (e.g. OpenRouter free endpoints)
                if not getattr(response, "choices", None):
                    raise ValueError(f"Malformed API response (missing choices). Raw: {response}")
                    
                choice = response.choices[0]
                message = choice.message
                reasoning = getattr(message, 'reasoning_content', None)
                
                # Log LLM response
                if self.verbose:
                    print(f"\n📥 LLM RESPONSE:")
                    content_len = len(message.content) if message.content else 0
                    print(f"   Content ({content_len} chars):")
                    print("-"*80)
                    print(message.content if message.content else "[None - Empty response]")
                    print("-"*80)
                    
                    if reasoning:
                        print(f"\n🧠 REASONING ({len(reasoning)} chars):")
                        print("-"*80)
                        print(reasoning)
                        print("-"*80)
                    
                    print(f"\n📊 TOKENS:")
                    input_t = response.usage.prompt_tokens if getattr(response, 'usage', None) else 0
                    output_t = response.usage.completion_tokens if getattr(response, 'usage', None) else 0
                    print(f"   Input: {input_t}")
                    print(f"   Output: {output_t}")
                    if getattr(response, 'usage', None) and getattr(response.usage, 'completion_tokens_details', None):
                        details = response.usage.completion_tokens_details
                        if getattr(details, 'reasoning_tokens', None) is not None:
                            print(f"   Reasoning: {details.reasoning_tokens}")
                
                # Save in history
                history.append({
                    'iteration': iteration + 1,
                    'llm_response': message.content,
                    'reasoning': reasoning,
                    'tokens': {
                        'input': response.usage.prompt_tokens,
                        'output': response.usage.completion_tokens
                    }
                })
                
                # Add LLM response to conversation (handle None)
                response_content = message.content if message.content else ""
                messages.append({
                    "role": "assistant",
                    "content": response_content
                })
                
                # Parse response
                code_to_execute, final_answer = self._parse_response(message.content)
                
                # Check for Final Answer
                if final_answer:
                    if self.verbose:
                        print(f"\n✅ FINAL ANSWER FOUND:")
                        print("-"*80)
                        print(final_answer)
                        print("-"*80)
                    
                    history[-1]['final_answer'] = final_answer
                    
                    result = {
                        'answer': final_answer,
                        'success': True,
                        'iterations': iteration + 1,
                        'history': history
                    }
                    
                    # Save JSON log if requested
                    if log_to_json:
                        self._save_log_json(log_to_json, question, result, df, initial_prompt)
                    
                    return result
                
                # Execute code if present
                if code_to_execute:
                    if self.verbose:
                        print(f"\n🔧 EXECUTING CODE:")
                        print(f"   {code_to_execute}")
                    
                    success, result = self._execute_code(code_to_execute, df)
                    
                    if self.verbose:
                        if success:
                            print(f"✅ RESULT: {result}")
                        else:
                            print(f"❌ ERROR: {result}")
                    
                    history[-1]['code_executed'] = code_to_execute
                    history[-1]['execution_result'] = result
                    history[-1]['execution_success'] = success
                    
                    # Add observation to conversation
                    observation = f"Observation: {result}"
                    messages.append({
                        "role": "user",
                        "content": observation
                    })
                    
                    if self.verbose:
                        print(f"\n📥 OBSERVATION SENT BACK TO LLM:")
                        # Truncate for visualization but show complete length
                        if len(observation) > 500:
                            print(f"   {observation[:500]}...")
                            print(f"   [Truncated for visualization - Complete length: {len(observation)} chars]")
                            print(f"   [The LLM receives the COMPLETE content, not truncated]")
                        else:
                            print(f"   {observation}")
                else:
                    # No Action Input or Final Answer found
                    if self.verbose:
                        if message.content is None:
                            print("\n⚠️  LLM returned None/empty content")
                        else:
                            print("\n⚠️  No Action Input or Final Answer found in response")
                    
                    # If content is None/empty, consider it a failure
                    if message.content is None or message.content.strip() == "":
                        if self.verbose:
                            print("   Cannot continue without valid LLM response")
                        break
                    
                    # Otherwise it might be a non-standard format, try again
                    break
                
            except Exception as e:
                if self.verbose:
                    print(f"\n❌ ERROR during LLM call or execution:")
                    print(f"   {type(e).__name__}: {e}")
                
                return {
                    'answer': f"Error: {e}",
                    'success': False,
                    'iterations': iteration + 1,
                    'history': history,
                    'error': str(e)
                }
        
        # Max iterations reached
        if self.verbose:
            print(f"\n⚠️  MAX ITERATIONS ({max_iterations}) reached without Final Answer")
        
        result = {
            'answer': "Unable to complete - max iterations reached",
            'success': False,
            'iterations': max_iterations,
            'history': history
        }
        
        # Save JSON log if requested
        if log_to_json:
            self._save_log_json(log_to_json, question, result, df, initial_prompt)
        
        return result
    
    def query_simple(self, df: pd.DataFrame, question: str) -> str:
        """
        Simplified version of query() that returns only the answer.
        
        Args:
            df: DataFrame to query
            question: Question
            
        Returns:
            Answer as a string
        """
        result = self.query(df, question)
        return result.get('answer', 'Error: No answer')


# Quick test
if __name__ == "__main__":
    print("\n" + "="*80)
    print("TEST: ManualPandasAgent")
    print("="*80)
    
    # Create test dataframe
    test_df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'City': ['Rome', 'Milan', 'Naples', 'Turin']
    })
    
    print(f"\nTest DataFrame:\n{test_df}")
    
    # Create agent
    agent = ManualPandasAgent(verbose=True)
    
    # Test query
    question = "How many people are over 30 years old?"
    print(f"\n\nQUESTION: {question}\n")
    
    result = agent.query(test_df, question)
    
    print("\n" + "="*80)
    print("📊 FINAL RESULT")
    print("="*80)
    print(f"Answer: {result['answer']}")
    print(f"Success: {result['success']}")
    print(f"Iterations: {result['iterations']}")
    print("="*80)
