"""
TEST COMPLETO: NIST Compliance Dialogue System

Questo test dimostra l'intero flusso del sistema:
1. Dialogue Manager inizia il dialogo
2. Utente risponde (simulato)
3. Sistema estrae informazioni
4. Sistema interroga database NIST (ManualPandasAgent + Direct Methods)
5. Sistema genera prossima domanda
6. Ripeti fino a completamento

TUTTO è loggato: prompts, responses, reasoning, codice eseguito, risultati.
"""
import sys
from dialogue_manager import NISTComplianceDialogueManager


def print_separator(title="", char="="):
    """Helper per stampare separatori"""
    width = 80
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(f"\n{char * width}")


def simulate_dialogue_scenario():
    """
    Simula uno scenario realistico di dialogo compliance.
    
    Scenario: Small Healthcare Clinic
    - 30 employees
    - Stores patient health records (PHI)
    - Cloud-based EHR system
    - Basic security (passwords only)
    - Needs HIPAA compliance
    """
    
    print_separator("NIST CSF 2.0 COMPLIANCE DIALOGUE SYSTEM - FULL TEST", "=")
    print("""
📋 TEST SCENARIO:
   Organization: Small Healthcare Clinic
   Employees: ~30
   Data: Patient Health Information (PHI)
   Infrastructure: Cloud-based EHR
   Current Security: Basic (passwords only)
   Requirements: HIPAA compliance
   
🎯 OBJECTIVE:
   Demonstrate complete dialogue flow with full logging of all data:
   - LLM prompts and responses
   - Reasoning tokens
   - Code generation and execution
   - State management
   - Control identification
""")
    
    input("\n▶️  Press ENTER to start the dialogue simulation...")
    
    # Inizializza Dialogue Manager
    print_separator("INITIALIZATION")
    dialogue = NISTComplianceDialogueManager(verbose=True)
    
    input("\n▶️  Press ENTER to start dialogue...")
    
    # Start dialogue
    print_separator("DIALOGUE START")
    first_question = dialogue.start_dialogue()
    
    # Simulated user responses
    user_responses = [
        # Turn 1: Industry
        "We are a small healthcare clinic providing primary care services to patients in the local community.",
        
        # Turn 2: Size
        "We're a small practice with approximately 30 employees, including doctors, nurses, and administrative staff.",
        
        # Turn 3: Data types
        "We handle sensitive patient data including electronic health records (EHR), personal health information (PHI), medical histories, test results, and insurance information.",
        
        # Turn 4: Infrastructure
        "We use a cloud-based EHR system hosted on AWS. We also have some on-premise workstations and a local file server for administrative documents. Staff access the system remotely sometimes.",
        
        # Turn 5: Maturity
        "Honestly, we have minimal security. Just basic password protection and antivirus software. We know we need to improve for HIPAA compliance.",
        
        # Turn 6: Compliance requirements
        "We need to comply with HIPAA regulations since we handle PHI. We've had some concerns about data breaches in healthcare and want to ensure we're protected."
    ]
    
    # Execute dialogue turns
    for i, user_response in enumerate(user_responses, 1):
        print_separator(f"TURN {i}")
        
        print(f"\n👤 USER RESPONSE:")
        print(f"   {user_response}")
        
        input(f"\n▶️  Press ENTER to process Turn {i}...")
        
        # Process response
        next_question = dialogue.process_user_response(user_response)
        
        print(f"\n{'='*80}")
        print(f"💬 AI NEXT QUESTION:")
        print(next_question)
        print("="*80)
        
        # Check se completato
        if "COMPLETE" in next_question:
            print_separator("DIALOGUE COMPLETED")
            break
        
        input(f"\n▶️  Press ENTER to continue to Turn {i+1}...")
    
    # Export log
    print_separator("EXPORTING CONVERSATION LOG")
    dialogue.export_conversation_log()
    dialogue.export_detailed_log()
    
    # Summary
    print_separator("TEST SUMMARY")
    print(f"""
✅ DIALOGUE COMPLETED SUCCESSFULLY

📊 STATISTICS:
   - Total turns: {dialogue.turn_counter}
   - Phases completed: {len([t for t in dialogue.conversation_history])}
   - Company profile fields populated: {len([v for v in dialogue.company_profile.to_dict().values() if v])}
   - Conversation log saved in: {dialogue.run_dir}

🔍 WHAT WAS DEMONSTRATED:
   almost nothing

📄 Check 'test_dialogue_log.json' for complete conversation record.
📄 Check 'test_dialogue_detailed_log.json' for full prompt/reasoning/agent trace.
""")


def test_manual_pandas_agent_only():
    """
    Test standalone del ManualPandasAgent per verificare funzionamento.
    """
    from pandas_agent_manual import ManualPandasAgent
    from nist_data_loader import NISTDataLoader
    
    print_separator("MANUAL PANDAS AGENT - STANDALONE TEST", "=")
    
    # Carica dati
    loader = NISTDataLoader()
    csf_df = loader.load_csf_mapping()
    
    print(f"\n📊 Dataset loaded: {len(csf_df)} rows × {len(csf_df.columns)} columns")
    print(f"Columns: {list(csf_df.columns)}")
    
    # Crea agent
    agent = ManualPandasAgent(verbose=True)
    
    # Test queries
    test_queries = [
        "How many rows are in the dataset?",
        "How many subcategories have 'Protect' in the Function column?",
        "Show me the unique values in the Function column"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print_separator(f"QUERY {i}: {query}")
        input("\n▶️  Press ENTER to execute query...")
        
        result = agent.query(csf_df, query, max_iterations=3)
        
        print(f"\n{'='*80}")
        print(f"📊 QUERY RESULT:")
        print(f"   Answer: {result['answer']}")
        print(f"   Success: {result['success']}")
        print(f"   Iterations: {result['iterations']}")
        print("="*80)
        
        input("\n▶️  Press ENTER to continue...")


def main():
    """Main test function"""
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║     🧪 NIST CSF 2.0 COMPLIANCE SYSTEM - COMPREHENSIVE TEST SUITE 🧪       ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

This test suite demonstrates:

1️⃣  MANUAL PANDAS AGENT TEST
   - Standalone test of the dynamic query engine
   - Shows ReAct format in action
   - Code generation and execution
   - Full transparency of LLM operations

2️⃣  COMPLETE DIALOGUE SIMULATION
   - Full end-to-end dialogue flow
   - Healthcare clinic scenario
   - Information extraction from natural language
   - State management
   - NIST database queries
   - Report generation

Choose test to run:
""")
    
    print("1. Manual Pandas Agent only (Quick test)")
    print("2. Complete Dialogue Simulation (Full system test)")
    print("3. Both tests")
    
    choice = input("\nYour choice (1/2/3): ").strip()
    
    try:
        if choice == "1":
            test_manual_pandas_agent_only()
        elif choice == "2":
            simulate_dialogue_scenario()
        elif choice == "3":
            test_manual_pandas_agent_only()
            print("\n\n")
            input("▶️  Press ENTER to proceed to full dialogue test...")
            simulate_dialogue_scenario()
        else:
            print("Invalid choice. Running full dialogue test...")
            simulate_dialogue_scenario()
        
        print_separator("ALL TESTS COMPLETED", "=")
        print("""
✅ Test suite execution completed successfully!

📝 NEXT STEPS:
   1. Review test_dialogue_log.json for full conversation trace
   2. Analyze LLM responses and code generation
   3. Evaluate control identification accuracy
   4. Consider implementing Phase 4: Evaluation & Metrics

Thank you for testing! 🚀
""")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
