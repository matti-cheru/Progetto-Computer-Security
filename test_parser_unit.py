"""
Test unitario del parser migliorato.
Verifica che riconosca risposte finali anche senza "Final Answer:" esplicito.
"""
from pandas_agent_manual import ManualPandasAgent
import pandas as pd

print("="*80)
print("🧪 TEST UNITARIO: Parser Migliorato")
print("="*80)

# Crea agent
agent = ManualPandasAgent(verbose=False)

# Test Case 1: Risposta con "Final Answer:" esplicito (già funzionava)
print("\n📝 Test 1: Risposta con 'Final Answer:' esplicito")
content1 = """Thought: I have the answer.
Final Answer: The total is 42.
"""
code1, answer1 = agent._parse_response(content1)
print(f"   Code: {code1}")
print(f"   Answer: {answer1}")
print(f"   ✅ PASS" if answer1 == "The total is 42." else "   ❌ FAIL")

# Test Case 2: Risposta con Action Input (normale iterazione ReAct)
print("\n📝 Test 2: Risposta con Action Input")
content2 = """Thought: I need to filter the data.
Action: python
Action Input: df[df['Category'].str.contains('Data Security', case=False)]
"""
code2, answer2 = agent._parse_response(content2)
print(f"   Code: {code2}")
print(f"   Answer: {answer2}")
expected_code = "df[df['Category'].str.contains('Data Security', case=False)]"
print(f"   ✅ PASS" if code2 == expected_code else "   ❌ FAIL")

# Test Case 3: Risposta finale implicita con tabella markdown (il caso problematico!)
print("\n📝 Test 3: Risposta finale implicita con tabella markdown")
content3 = """**Subcategories related to data protection and access control**

| Subcategory_ID | Subcategory_Description |
|----------------|-------------------------|
| PR.AC-1 | Identities and credentials are issued, managed… |
| PR.AC-2 | Physical access to assets is managed and protected… |
| PR.DS-1 | Data‑at‑rest is protected… |

These are all the subcategories that fall under the "Data Security" (data protection) and "Identity Management, Authentication and Access Control" (access control) categories.
"""
code3, answer3 = agent._parse_response(content3)
print(f"   Code: {code3}")
print(f"   Answer (first 100 chars): {answer3[:100] if answer3 else None}...")
print(f"   ✅ PASS - Riconosciuta come risposta finale!" if answer3 is not None else "   ❌ FAIL - Non riconosciuta!")

# Test Case 4: Risposta finale narrativa senza tabella
print("\n📝 Test 4: Risposta finale narrativa")
content4 = """The following subcategories are related to access control:
- PR.AC-1: Identities and credentials are issued
- PR.AC-2: Physical access to assets is managed
- PR.AC-3: Remote access is managed

These subcategories_id provide comprehensive coverage of access control requirements.
"""
code4, answer4 = agent._parse_response(content4)
print(f"   Code: {code4}")
print(f"   Answer (first 100 chars): {answer4[:100] if answer4 else None}...")
print(f"   ✅ PASS - Riconosciuta come risposta finale!" if answer4 is not None else "   ❌ FAIL - Non riconosciuta!")

# Test Case 5: None content (bug fix precedente)
print("\n📝 Test 5: None content")
content5 = None
code5, answer5 = agent._parse_response(content5)
print(f"   Code: {code5}")
print(f"   Answer: {answer5}")
print(f"   ✅ PASS - Gestito correttamente!" if code5 is None and answer5 is None else "   ❌ FAIL")

# Test Case 6: Risposta con Thought ma senza Action (dovrebbe essere riconosciuta come finale?)
print("\n📝 Test 6: Risposta con solo Thought")
content6 = """Thought: The query returned the complete list of subcategories.

Here are all subcategories related to data protection and access control:
PR.AC-1, PR.AC-2, PR.DS-1, PR.DS-2
"""
code6, answer6 = agent._parse_response(content6)
print(f"   Code: {code6}")
print(f"   Answer: {answer6}")
print(f"   ⚠️  Ha keyword ReAct (Thought) quindi non riconosciuta come finale" if answer6 is None else f"   ℹ️  Riconosciuta come finale")

print("\n" + "="*80)
print("✅ Test completato")
print("="*80)
print("\n💡 RIEPILOGO:")
print("   - Il parser ora riconosce risposte finali con tabelle markdown")
print("   - Il parser riconosce risposte finali narrative con indicatori chiave")
print("   - Il parser continua a supportare il formato 'Final Answer:' esplicito")
print("   - Il parser gestisce correttamente None content")
print("   - Risposte con keyword ReAct (Thought/Action) non sono considerate finali")
print("     a meno che non contengano indicatori di risposta finale")
