"""
Test dei fix applicati:
1. Parser riconosce risposte finali senza "Final Answer:" esplicito
2. Logging JSON automatico
3. Visualizzazione corretta di observation troncate
"""
from nist_data_loader import NISTDataLoader
from pandas_agent_manual import ManualPandasAgent
import os
from datetime import datetime

print("="*80)
print("🧪 TEST: Fix Parser + Logging JSON")
print("="*80)

# Carica dati
loader = NISTDataLoader()
df = loader.load_csf_mapping()

print(f"\n📊 Dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# Crea agent
agent = ManualPandasAgent(verbose=True)

# Test la query problematica con logging JSON
question = "Show me all subcategories related to data protection and access control"

# Genera nome file JSON con timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"query_log_{timestamp}.json"

print(f"\n❓ QUESTION: {question}")
print(f"💾 Log file: {log_file}")
print("\n⏳ Executing query...\n")

result = agent.query(df, question, max_iterations=5, log_to_json=log_file)

print("\n" + "="*80)
print("📊 FINAL RESULT")
print("="*80)
print(f"Success: {result['success']}")
print(f"Iterations: {result['iterations']}")
print(f"\nAnswer (first 500 chars):")
print("-"*80)
answer = result.get('answer', 'N/A')
if len(answer) > 500:
    print(answer[:500] + "...")
    print(f"[Risposta completa: {len(answer)} caratteri]")
else:
    print(answer)
print("="*80)

# Verifica che il log JSON esista
if os.path.exists(log_file):
    print(f"\n✅ Log JSON salvato correttamente: {log_file}")
    import json
    with open(log_file, 'r', encoding='utf-8') as f:
        log_data = json.load(f)
    print(f"   - Timestamp: {log_data['timestamp']}")
    print(f"   - Question: {log_data['question']}")
    print(f"   - Success: {log_data['result']['success']}")
    print(f"   - Iterations: {log_data['result']['iterations']}")
    print(f"   - History entries: {len(log_data['history'])}")
else:
    print(f"\n❌ Log JSON non trovato: {log_file}")

print("\n" + "="*80)
print("✅ Test completato")
print("="*80)
