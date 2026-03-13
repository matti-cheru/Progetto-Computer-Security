"""
TEST: Riprodurre e fixare l'errore "NoneType is not iterable"

Problema: Quando l'LLM restituisce content=None, il metodo _parse_response fallisce.
"""
from pandas_agent_manual import ManualPandasAgent
from nist_data_loader import NISTDataLoader


def test_problematic_query():
    """
    Testa la query che causa l'errore
    """
    print("="*80)
    print("TEST: Riproduzione errore 'NoneType is not iterable'")
    print("="*80)
    
    # Carica dati
    print("\n📊 Loading data...")
    loader = NISTDataLoader()
    csf_df = loader.load_csf_mapping()
    print(f"   ✅ Loaded: {len(csf_df)} rows")
    
    # Crea agent
    print("\n🤖 Creating ManualPandasAgent...")
    agent = ManualPandasAgent(verbose=True)
    
    # Query problematica
    print("\n❓ Executing problematic query...")
    query = "Show me all subcategories related to data protection and access control"
    
    print(f"\n   Query: {query}")
    print("\n⏳ This query often causes the LLM to return None content...")
    print("   We'll catch and display the error.\n")
    
    try:
        result = agent.query(csf_df, query, max_iterations=3)
        
        print("\n" + "="*80)
        print("📊 RESULT:")
        print("="*80)
        print(f"Success: {result['success']}")
        print(f"Answer: {result.get('answer', 'N/A')}")
        print(f"Iterations: {result['iterations']}")
        
        if 'error' in result:
            print(f"\n❌ Error caught: {result['error']}")
        
        # Mostra history per debug
        print("\n📜 ITERATION HISTORY:")
        for i, turn in enumerate(result['history'], 1):
            print(f"\n  Iteration {i}:")
            llm_resp = turn.get('llm_response', 'None')
            llm_resp_display = (llm_resp[:100] if llm_resp else '[None]') + "..."
            print(f"    LLM Response: {llm_resp_display}")
            print(f"    Code executed: {turn.get('code_executed', 'N/A')}")
            exec_result = turn.get('execution_result', 'N/A')
            exec_result_display = (exec_result[:100] if exec_result else '[None]')
            print(f"    Execution result: {exec_result_display}")
            
    except Exception as e:
        print(f"\n❌ EXCEPTION CAUGHT: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)


def test_parse_response_with_none():
    """
    Test diretto del metodo _parse_response con None
    """
    print("\n\n" + "="*80)
    print("TEST: _parse_response con content=None")
    print("="*80)
    
    agent = ManualPandasAgent(verbose=False)
    
    # Test 1: content=None
    print("\n1️⃣  Testing with content=None:")
    try:
        code, answer = agent._parse_response(None)
        print(f"   ✅ Result: code={code}, answer={answer}")
    except Exception as e:
        print(f"   ❌ ERROR: {type(e).__name__}: {e}")
    
    # Test 2: content="" (empty string)
    print("\n2️⃣  Testing with content='' (empty):")
    try:
        code, answer = agent._parse_response("")
        print(f"   ✅ Result: code={code}, answer={answer}")
    except Exception as e:
        print(f"   ❌ ERROR: {type(e).__name__}: {e}")
    
    # Test 3: content con testo ma senza Action Input o Final Answer
    print("\n3️⃣  Testing with text but no Action Input or Final Answer:")
    content = "This is some random text without the expected format."
    try:
        code, answer = agent._parse_response(content)
        print(f"   ✅ Result: code={code}, answer={answer}")
    except Exception as e:
        print(f"   ❌ ERROR: {type(e).__name__}: {e}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              🐛 DEBUG: NoneType Error in Pandas Agent 🐛                  ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

Questo test riproduce l'errore "argument of type 'NoneType' is not iterable"
che si verifica quando l'LLM restituisce content=None.
""")
    
    # Test parsing diretto
    test_parse_response_with_none()
    
    # Test query problematica (potrebbe richiedere tempo)
    print("\n\n" + "🔍 Vuoi testare la query problematica completa?")
    print("   (Richiede chiamata LLM, può richiedere 10-20 secondi)")
    response = input("   Eseguire? (s/n): ").strip().lower()
    
    if response == 's':
        test_problematic_query()
    else:
        print("\n   ⏭️  Skipped")
    
    print("\n✅ Test completato\n")
