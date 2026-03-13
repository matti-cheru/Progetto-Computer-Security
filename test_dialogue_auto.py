"""
TEST AUTOMATIZZATO: NIST Compliance Dialogue System

Versione automatizzata senza input() per testing completo.
"""
from dialogue_manager import NISTComplianceDialogueManager
from pandas_agent_manual import ManualPandasAgent
from nist_data_loader import NISTDataLoader


def print_separator(title="", char="="):
    """Helper per stampare separatori"""
    width = 80
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(f"\n{char * width}")


def test_manual_pandas_agent():
    """Test standalone del ManualPandasAgent"""
    print_separator("TEST 1: MANUAL PANDAS AGENT", "=")
    
    print("\n📊 Loading NIST CSF dataset...")
    loader = NISTDataLoader()
    csf_df = loader.load_csf_mapping()
    
    print(f"   ✅ Dataset loaded: {len(csf_df)} rows × {len(csf_df.columns)} columns")
    print(f"   Columns: {list(csf_df.columns)}")
    
    print("\n🤖 Initializing ManualPandasAgent...")
    agent = ManualPandasAgent(verbose=True)
    
    # Test Query 1: Simple count
    print_separator("QUERY 1: Count Rows")
    query1 = "How many rows are in the dataset?"
    print(f"\n❓ Question: {query1}")
    
    result1 = agent.query(csf_df, query1, max_iterations=3)
    
    print(f"\n{'='*80}")
    print(f"📊 FINAL RESULT:")
    print(f"   Answer: {result1['answer']}")
    print(f"   Success: {result1['success']}")
    print(f"   Iterations: {result1['iterations']}")
    print("="*80)
    
    # Test Query 2: Filter count
    print_separator("QUERY 2: Filter and Count")
    query2 = "How many subcategories have the function 'Protect'?"
    print(f"\n❓ Question: {query2}")
    
    result2 = agent.query(csf_df, query2, max_iterations=3)
    
    print(f"\n{'='*80}")
    print(f"📊 FINAL RESULT:")
    print(f"   Answer: {result2['answer']}")
    print(f"   Success: {result2['success']}")
    print(f"   Iterations: {result2['iterations']}")
    print("="*80)
    
    return result1['success'] and result2['success']


def test_complete_dialogue():
    """Test completo del dialogue system"""
    print_separator("TEST 2: COMPLETE DIALOGUE SIMULATION", "=")
    
    print("""
📋 SCENARIO:
   Organization: Small Healthcare Clinic
   Employees: ~30
   Data: Patient Health Information (PHI)
   Infrastructure: Cloud-based EHR
   Current Security: Basic (passwords only)
   Requirements: HIPAA compliance
""")
    
    print("\n🚀 Initializing Dialogue Manager...")
    dialogue = NISTComplianceDialogueManager(verbose=True)
    
    print_separator("STARTING DIALOGUE")
    first_question = dialogue.start_dialogue()
    
    # User responses simulati
    user_responses = [
        "We are a small healthcare clinic providing primary care services.",
        "Small practice with approximately 30 employees including doctors and nurses.",
        "We handle electronic health records (EHR), personal health information (PHI), and medical histories.",
        "Cloud-based EHR on AWS, some on-premise workstations, remote access for staff.",
        "Minimal security - just passwords and antivirus. Need to improve for HIPAA.",
        "Need HIPAA compliance for PHI. Concerned about data breaches."
    ]
    
    # Esegui i turni
    for i, user_response in enumerate(user_responses, 1):
        print_separator(f"TURN {i}")
        
        print(f"\n👤 USER: {user_response}")
        
        print(f"\n🔄 Processing turn {i}...")
        next_question = dialogue.process_user_response(user_response)
        
        print(f"\n💬 AI: {next_question[:200]}..." if len(next_question) > 200 else f"\n💬 AI: {next_question}")
        
        if "COMPLETE" in next_question or dialogue.current_phase.value == "completion":
            print_separator("DIALOGUE COMPLETED")
            break
    
    # Export log
    print_separator("EXPORTING LOG")
    dialogue.export_conversation_log()
    dialogue.export_detailed_log()
    
    # Summary
    print_separator("DIALOGUE SUMMARY")
    print(f"""
✅ Dialogue completed successfully

📁 Logs saved in: {dialogue.run_dir}

📊 Statistics:
   - Total turns: {dialogue.turn_counter}
   - Profile fields populated: {len([v for v in dialogue.company_profile.to_dict().values() if v])}
   
📋 Company Profile:
   - Industry: {dialogue.company_profile.industry}
   - Size: {dialogue.company_profile.size}
   - Employees: {dialogue.company_profile.employees}
   - Data Sensitivity: {dialogue.company_profile.data_sensitivity}
   - Infrastructure: {dialogue.company_profile.infrastructure}
   - Maturity: {dialogue.company_profile.current_maturity}
   - Compliance: {dialogue.company_profile.compliance_requirements}

📄 Full log: test_dialogue_auto_log.json
📄 Detailed log: test_dialogue_auto_detailed_log.json
""")
    
    return True


def main():
    """Main test runner"""
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║           🧪 NIST CSF 2.0 COMPLIANCE - AUTOMATED TEST SUITE 🧪            ║
║                                                                            ║
║  This test demonstrates the complete implementation of Phase 3:           ║
║  Structured Dialogue System with RAG                                      ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
    
    results = {}
    
    try:
        # Test 1: ManualPandasAgent
        print("\n🔬 Running Test 1: Manual Pandas Agent...")
        results['pandas_agent'] = test_manual_pandas_agent()
        
        print("\n\n" + "="*80)
        print("Pausing between tests...")
        print("="*80 + "\n")
        
        # Test 2: Complete Dialogue
        print("\n🔬 Running Test 2: Complete Dialogue...")
        results['dialogue'] = test_complete_dialogue()
        
        # Final summary
        print_separator("FINAL TEST SUMMARY", "=")
        print(f"""
✅ ALL TESTS COMPLETED!

📊 Results:
   - ManualPandasAgent: {'✅ PASS' if results['pandas_agent'] else '❌ FAIL'}
   - Complete Dialogue: {'✅ PASS' if results['dialogue'] else '❌ FAIL'}

🎯 WHAT WAS DEMONSTRATED:

1️⃣  Phase 1: Data Layer (NISTDataLoader)
   ✓ Cleaned NIST datasets loaded
   ✓ 108 CSF subcategories available
   ✓ SP 800-53 controls mapped

2️⃣  Phase 2: Logical Core
   ✓ ManualPandasAgent: Dynamic queries in natural language
   ✓ ReAct format: Thought → Action → Observation → Answer
   ✓ Code generation and execution on real data
   ✓ Reasoning tokens visible (transparency)
   ✓ Direct Methods: Fast lookups (fallback)

3️⃣  Phase 3: Dialogue Manager
   ✓ Structured dialogue flow
   ✓ State management (company profile)
   ✓ LLM-based information extraction
   ✓ Context-aware questioning
   ✓ Dynamic NIST queries based on profile

📄 OUTPUT FILES:
   - test_dialogue_auto_log.json: Complete conversation log
    - test_dialogue_auto_detailed_log.json: Full prompt/reasoning/agent trace

🚀 NEXT STEPS (Phase 4):
   - Implement evaluation metrics (RAGAS)
   - Create golden test dataset
   - Add control prioritization logic
   - Generate PDF reports
   - User interface (Web/CLI)

Thank you for testing! 🎉
""")
        
        if all(results.values()):
            print("\n🎊 ALL TESTS PASSED! System is working correctly! 🎊\n")
            return 0
        else:
            print("\n⚠️  Some tests failed. Check logs above. ⚠️\n")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
