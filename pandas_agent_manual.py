"""
PANDAS AGENT MANUAL - Core RAG Engine

Implementazione manuale di un Pandas Agent usando libreria OpenAI nativa.
Questo agent:
- Interpreta domande in linguaggio naturale
- Genera codice Python per interrogare dataframe
- Esegue il codice su dati reali NIST
- Segue il formato ReAct (Reasoning + Acting)
- Espone reasoning tokens per trasparenza

Questo è il CUORE del sistema RAG dinamico per il dialogo compliance.
"""
import os
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
from openai import OpenAI


class ManualPandasAgent:
    """
    Pandas Agent manuale con pieno controllo e trasparenza.
    
    Usa la libreria openai nativa (non LangChain) per:
    - Accesso completo a reasoning_content
    - Controllo totale sul flusso ReAct
    - Debugging dettagliato di ogni step
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
        Inizializza il Pandas Agent manuale.
        
        Args:
            base_url: URL del cluster LLM UniBS
            model_name: Modello da usare
            api_key: API key (default da env GPUSTACK_API_KEY)
            temperature: Temperatura LLM (0.0 = deterministico)
            verbose: Se True, stampa tutti i dettagli
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
            print(f"✅ ManualPandasAgent inizializzato")
            print(f"   Model: {model_name}")
            print(f"   Temperature: {temperature}")
            print(f"   Base URL: {base_url}")
    
    def _create_prompt(self, df: pd.DataFrame, question: str) -> str:
        """
        Crea il prompt ReAct-style per interrogare un dataframe.
        
        Args:
            df: DataFrame da interrogare
            question: Domanda in linguaggio naturale
            
        Returns:
            Prompt formattato per il modello
        """
        # Informazioni sul dataframe
        df_info = f"""You have access to a pandas dataframe called `df` with the following structure:

Shape: {df.shape[0]} rows × {df.shape[1]} columns
Columns: {list(df.columns)}

First 3 rows as reference:
{df.head(3).to_string()}

"""
        
        # Aggiungi informazioni sulle categorie se presente la colonna Category
        if 'Category' in df.columns:
            unique_categories = df['Category'].unique()
            categories_sample = list(unique_categories[:10])  # Prime 10 categorie
            df_info += f"""
To search by topic (e.g., "data protection", "access control"), use the Category column.
Example: df[df['Category'].str.contains('Data Security', case=False)]
"""
        
        # Istruzioni ReAct
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
        Parsa la risposta del modello per estrarre Action Input o Final Answer.
        
        Args:
            content: Contenuto della risposta (può essere None)
            
        Returns:
            Tuple (code_to_execute, final_answer)
            - code_to_execute: Codice Python da eseguire (se trovato)
            - final_answer: Risposta finale (se trovata)
        """
        # Gestisci caso None o empty
        if content is None or content == "":
            return None, None
        
        # Check per Final Answer (esplicito)
        if "Final Answer:" in content:
            final_answer = content.split("Final Answer:")[-1].strip()
            return None, final_answer
        
        # Check per risposta finale implicita (tabella, lista formattata, risposta narrativa)
        # Se non c'è "Action:" o "Thought:" e il contenuto sembra una risposta
        content_lower = content.lower()
        has_react_keywords = any(kw in content for kw in ["Action:", "Action Input:", "Thought:", "Observation:"])
        
        # Se non ha keyword ReAct e sembra una risposta formattata (markdown table, lista, paragrafo)
        if not has_react_keywords:
            # Cerca indicatori di risposta finale
            final_indicators = [
                "subcategories related to",
                "| subcategory_id |",  # Markdown table header
                "these are all",
                "the following",
                "here are the",
                "subcategory_id"
            ]
            
            if any(indicator in content_lower for indicator in final_indicators):
                # Probabilmente è una risposta finale
                return None, content.strip()
        
        # Cerca Action Input
        if "Action Input:" in content:
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if "Action Input:" in line:
                    # Codice può essere sulla stessa riga o riga dopo
                    code = line.split("Action Input:")[-1].strip()
                    if not code and i + 1 < len(lines):
                        code = lines[i + 1].strip()
                    
                    # Rimuovi eventuali backticks
                    code = code.strip('`').strip()
                    
                    if code:
                        return code, None
        
        return None, None
    
    def _execute_code(self, code: str, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Esegue codice Python in modo sicuro (con eval).
        
        Args:
            code: Codice Python da eseguire
            df: DataFrame su cui eseguire
            
        Returns:
            Tuple (success, result_or_error)
        """
        try:
            # Esegui in ambiente controllato
            result = eval(code, {"df": df, "pd": pd, "len": len, "str": str})

            # Evita il troncamento con "..." tipico della rappresentazione pandas.
            # In questo modo il log JSON contiene il contenuto completo dell'esecuzione.
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
        Salva il log completo della query in un file JSON.
        
        Args:
            filepath: Path del file JSON dove salvare
            question: Domanda originale
            result: Risultato della query (con history)
            df: DataFrame usato
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
        Esegue una query su un dataframe in linguaggio naturale.
        
        Questo è il metodo principale che implementa il ciclo ReAct:
        1. Invia prompt al modello
        2. Modello risponde con Thought/Action/Action Input
        3. Esegue il codice Python
        4. Ritorna Observation al modello
        5. Ripete fino a Final Answer o max iterations
        
        Args:
            df: DataFrame da interrogare
            question: Domanda in linguaggio naturale
            max_iterations: Max numero di iterazioni ReAct
            log_to_json: Se specificato, salva log completo in questo file JSON
            
        Returns:
            Dict con:
                - answer: Risposta finale
                - success: True se completato con successo
                - iterations: Numero di iterazioni usate
                - history: Storia completa della conversazione per debugging
        """
        if self.verbose:
            print("\n" + "="*80)
            print("🔍 MANUAL PANDAS AGENT - QUERY START")
            print("="*80)
            print(f"Question: {question}")
            print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Crea prompt iniziale
        initial_prompt = self._create_prompt(df, question)
        
        if self.verbose:
            print("\n📤 INITIAL PROMPT:")
            print("-"*80)
            print(initial_prompt[:500] + "..." if len(initial_prompt) > 500 else initial_prompt)
            print("-"*80)
        
        # Inizializza conversazione
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
        
        history = []  # Per tracking completo
        
        # Ciclo ReAct
        for iteration in range(max_iterations):
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"🔄 ITERATION {iteration + 1}/{max_iterations}")
                print("="*80)
            
            # Chiama LLM
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=500
                )
                
                choice = response.choices[0]
                message = choice.message
                
                # Log risposta LLM
                if self.verbose:
                    print(f"\n📥 LLM RESPONSE:")
                    content_len = len(message.content) if message.content else 0
                    print(f"   Content ({content_len} chars):")
                    print("-"*80)
                    print(message.content if message.content else "[None - Empty response]")
                    print("-"*80)
                    
                    if message.reasoning_content:
                        print(f"\n🧠 REASONING ({len(message.reasoning_content)} chars):")
                        print("-"*80)
                        print(message.reasoning_content)
                        print("-"*80)
                    
                    print(f"\n📊 TOKENS:")
                    print(f"   Input: {response.usage.prompt_tokens}")
                    print(f"   Output: {response.usage.completion_tokens}")
                    if hasattr(response.usage, 'completion_tokens_details'):
                        details = response.usage.completion_tokens_details
                        print(f"   Reasoning: {details.reasoning_tokens}")
                
                # Salva in history
                history.append({
                    'iteration': iteration + 1,
                    'llm_response': message.content,
                    'reasoning': message.reasoning_content,
                    'tokens': {
                        'input': response.usage.prompt_tokens,
                        'output': response.usage.completion_tokens
                    }
                })
                
                # Aggiungi risposta LLM alla conversazione (gestisci None)
                response_content = message.content if message.content else ""
                messages.append({
                    "role": "assistant",
                    "content": response_content
                })
                
                # Parsa risposta
                code_to_execute, final_answer = self._parse_response(message.content)
                
                # Check per Final Answer
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
                    
                    # Salva log JSON se richiesto
                    if log_to_json:
                        self._save_log_json(log_to_json, question, result, df, initial_prompt)
                    
                    return result
                
                # Esegui codice se presente
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
                    
                    # Aggiungi observation alla conversazione
                    observation = f"Observation: {result}"
                    messages.append({
                        "role": "user",
                        "content": observation
                    })
                    
                    if self.verbose:
                        print(f"\n📥 OBSERVATION SENT BACK TO LLM:")
                        # Tronca per visualizzazione ma mostra lunghezza completa
                        if len(observation) > 500:
                            print(f"   {observation[:500]}...")
                            print(f"   [Troncato per visualizzazione - Lunghezza completa: {len(observation)} chars]")
                            print(f"   [L'LLM riceve il contenuto COMPLETO, non troncato]")
                        else:
                            print(f"   {observation}")
                else:
                    # Nessun Action Input o Final Answer trovato
                    if self.verbose:
                        if message.content is None:
                            print("\n⚠️  LLM returned None/empty content")
                        else:
                            print("\n⚠️  No Action Input or Final Answer found in response")
                    
                    # Se content è None/empty, consideriamolo un fallimento
                    if message.content is None or message.content.strip() == "":
                        if self.verbose:
                            print("   Cannot continue without valid LLM response")
                        break
                    
                    # Altrimenti potrebbe essere un formato non standard, proviamo ancora
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
        
        # Max iterations raggiunto
        if self.verbose:
            print(f"\n⚠️  MAX ITERATIONS ({max_iterations}) reached without Final Answer")
        
        result = {
            'answer': "Unable to complete - max iterations reached",
            'success': False,
            'iterations': max_iterations,
            'history': history
        }
        
        # Salva log JSON se richiesto
        if log_to_json:
            self._save_log_json(log_to_json, question, result, df, initial_prompt)
        
        return result
    
    def query_simple(self, df: pd.DataFrame, question: str) -> str:
        """
        Versione semplificata di query() che ritorna solo la risposta.
        
        Args:
            df: DataFrame da interrogare
            question: Domanda
            
        Returns:
            Risposta come stringa
        """
        result = self.query(df, question)
        return result.get('answer', 'Error: No answer')


# Test rapido
if __name__ == "__main__":
    print("\n" + "="*80)
    print("TEST: ManualPandasAgent")
    print("="*80)
    
    # Crea dataframe di test
    test_df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'City': ['Rome', 'Milan', 'Naples', 'Turin']
    })
    
    print(f"\nTest DataFrame:\n{test_df}")
    
    # Crea agent
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
