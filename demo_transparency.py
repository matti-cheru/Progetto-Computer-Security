"""
DEMO VISIVO: Mostra tutti i dati scambiati in tempo reale

Questo script esegue una query semplice mostrando ESATTAMENTE:
- Il prompt inviato all'LLM
- La risposta dell'LLM (content)
- Il reasoning interno (reasoning_content)
- Il codice Python generato
- L'esecuzione e il risultato
- I token usati

Tutto in formato visivo e colorato (quando possibile).
"""
from pandas_agent_manual import ManualPandasAgent
from nist_data_loader import NISTDataLoader
import json


def demo_simple_query():
    """Demo di una query semplice con logging visivo completo"""
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║        🎬 DEMO: TRANSPARENCY & TRACEABILITY OF ALL DATA FLOWS 🎬          ║
║                                                                            ║
║  Questo demo mostra ESATTAMENTE cosa succede internamente:                ║
║    • Prompt → LLM                                                          ║
║    • LLM Response (content + reasoning)                                    ║
║    • Python code generato                                                  ║
║    • Execution e risultati                                                 ║
║    • Token usage                                                           ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Carica dati
    print("\n📊 STEP 1: Loading NIST CSF Dataset...")
    loader = NISTDataLoader()
    csf_df = loader.load_csf_mapping()
    print(f"   ✅ Loaded: {len(csf_df)} rows × {len(csf_df.columns)} columns")
    print(f"   Columns: {list(csf_df.columns)}")
    print(f"\n   Preview (first 2 rows):")
    print(csf_df[['Function', 'Subcategory_ID', 'Subcategory_Description']].head(2).to_string(index=False))
    
    # Crea agent
    print("\n\n🤖 STEP 2: Initializing ManualPandasAgent...")
    print("   Using: gpt-oss model via UniBS cluster")
    print("   Temperature: 0.0 (deterministic)")
    
    agent = ManualPandasAgent(verbose=False)  # Gestiremo noi il logging
    print("   ✅ Agent ready")
    
    # Query
    print("\n\n❓ STEP 3: User Question")
    question = "How many subcategories are related to 'Asset Management'?"
    print(f"   Question: \"{question}\"")
    
    print("\n\n🔄 STEP 4: Processing with ManualPandasAgent...")
    print("   This will:")
    print("   1. Send prompt to LLM with dataframe info")
    print("   2. LLM generates Python code (ReAct format)")
    print("   3. Execute code on real NIST data")
    print("   4. LLM formulates final answer")
    
    # Questa volta intercetteremo manualmente per mostrare step-by-step
    from openai import OpenAI
    import os
    
    client = OpenAI(
        base_url="https://gpustack.ing.unibs.it/v1",
        api_key=os.environ.get("GPUSTACK_API_KEY", "gpustack_8a00037ab4220858_6479d07686028e2f970357b1a81200e4")
    )
    
    # Crea prompt
    prompt = agent._create_prompt(csf_df, question)
    
    print("\n\n" + "="*80)
    print("📤 PROMPT SENT TO LLM")
    print("="*80)
    print(prompt + "\n... [truncated for display] ...")
    print("="*80)
    
    print("\n⏳ Waiting for LLM response... (this may take 5-10 seconds)")
    
    # Iteration 1
    messages = [
        {"role": "system", "content": "You are a Python data analysis assistant. Follow the ReAct format strictly."},
        {"role": "user", "content": prompt}
    ]
    
    response1 = client.chat.completions.create(
        model="gpt-oss",
        messages=messages,
        temperature=0.0,
        max_tokens=500
    )
    
    msg1 = response1.choices[0].message
    
    print("\n\n" + "="*80)
    print("📥 LLM RESPONSE - ITERATION 1")
    print("="*80)
    
    print("\n🎯 CONTENT (What the LLM outputs):")
    print("-"*80)
    print(msg1.content)
    print("-"*80)
    
    print("\n🧠 REASONING (Internal thought process):")
    print("-"*80)
    if msg1.reasoning_content:
        print(msg1.reasoning_content)
    else:
        print("[No reasoning content available]")
    print("-"*80)
    
    print("\n📊 TOKEN USAGE:")
    print(f"   Input tokens:  {response1.usage.prompt_tokens}")
    print(f"   Output tokens: {response1.usage.completion_tokens}")
    if hasattr(response1.usage, 'completion_tokens_details'):
        details = response1.usage.completion_tokens_details
        print(f"   Reasoning tokens: {details.reasoning_tokens}")
        print(f"   Regular tokens:   {response1.usage.completion_tokens - details.reasoning_tokens}")
    
    # Parse code
    code, final = agent._parse_response(msg1.content)
    
    if code:
        print("\n\n" + "="*80)
        print("🔧 PYTHON CODE EXTRACTED")
        print("="*80)
        print(f"\nCode to execute: {code}")
        
        print("\n⚙️  Executing code on NIST dataframe...")
        success, result = agent._execute_code(code, csf_df)
        
        if success:
            print(f"✅ Execution successful!")
            print(f"   Result: {result}")
        else:
            print(f"❌ Execution failed!")
            print(f"   Error: {result}")
        
        # Iteration 2 con observation
        print("\n\n" + "="*80)
        print("🔄 SENDING OBSERVATION BACK TO LLM")
        print("="*80)
        
        observation = f"Observation: {result}"
        print(f"   {observation}")
        
        messages.append({"role": "assistant", "content": msg1.content})
        messages.append({"role": "user", "content": observation})
        
        print("\n⏳ Waiting for final answer from LLM...")
        
        response2 = client.chat.completions.create(
            model="gpt-oss",
            messages=messages,
            temperature=0.0,
            max_tokens=500
        )
        
        msg2 = response2.choices[0].message
        
        print("\n\n" + "="*80)
        print("📥 LLM RESPONSE - ITERATION 2 (FINAL)")
        print("="*80)
        
        print("\n🎯 CONTENT:")
        print("-"*80)
        print(msg2.content)
        print("-"*80)
        
        print("\n🧠 REASONING:")
        print("-"*80)
        if msg2.reasoning_content:
            print(msg2.reasoning_content)
        else:
            print("[No reasoning content]")
        print("-"*80)
        
        print("\n📊 TOKEN USAGE (Iteration 2):")
        print(f"   Input tokens:  {response2.usage.prompt_tokens}")
        print(f"   Output tokens: {response2.usage.completion_tokens}")
        if hasattr(response2.usage, 'completion_tokens_details'):
            details = response2.usage.completion_tokens_details
            print(f"   Reasoning tokens: {details.reasoning_tokens}")
        
        # Extract final answer
        _, final_answer = agent._parse_response(msg2.content)
        
        print("\n\n" + "="*80)
        print("✅ FINAL ANSWER")
        print("="*80)
        print(f"\n   {final_answer if final_answer else msg2.content}")
        print("\n" + "="*80)
    
    # Summary
    print("\n\n" + "╔" + "="*78 + "╗")
    print("║" + " "*26 + "SUMMARY OF DATA FLOWS" + " "*31 + "║")
    print("╚" + "="*78 + "╝")
    

if __name__ == "__main__":
    try:
        demo_simple_query()
        print("\n✅ Demo completed successfully!\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
