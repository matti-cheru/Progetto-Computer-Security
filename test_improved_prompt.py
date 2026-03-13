"""
Test del fix: verifica che il ManualPandasAgent ora generi codice migliore
usando la colonna Category invece di cercare nelle descrizioni.
"""
from nist_data_loader import NISTDataLoader
from pandas_agent_manual import ManualPandasAgent

print("="*80)
print("🧪 TEST: Query con nuovo prompt migliorato")
print("="*80)

# Carica dati
loader = NISTDataLoader()
df = loader.load_csf_mapping()

print(f"\n📊 Dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# Crea agent
agent = ManualPandasAgent(verbose=True)

# Test la stessa query problematica
question = "Show me all subcategories related to data protection and access control"

print(f"\n❓ QUESTION: {question}")
print("\n⏳ Executing query...\n")

result = agent.query(df, question, max_iterations=5)

print("\n" + "="*80)
print("📊 FINAL RESULT")
print("="*80)
print(f"Success: {result['success']}")
print(f"Answer: {result.get('answer', 'N/A')}")
print(f"Iterations: {result['iterations']}")
print("="*80)
