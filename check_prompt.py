"""
Stampa il prompt generato per verificare che includa le informazioni sulle categorie.
"""
from nist_data_loader import NISTDataLoader
from pandas_agent_manual import ManualPandasAgent

# Carica dati
loader = NISTDataLoader()
df = loader.load_csf_mapping()

# Crea agent
agent = ManualPandasAgent(verbose=False)

# Genera prompt
question = "Show me all subcategories related to data protection and access control"
prompt = agent._create_prompt(df, question)

print("="*80)
print("📤 PROMPT COMPLETO GENERATO")
print("="*80)
print(prompt)
print("="*80)
print(f"\n📏 Lunghezza prompt: {len(prompt)} caratteri")
print("\n🔍 Controllo presenza informazioni categorie:")
if "Available Categories" in prompt:
    print("   ✅ Informazioni sulle categorie PRESENTI")
else:
    print("   ❌ Informazioni sulle categorie ASSENTI")

if "Data Security" in prompt:
    print("   ✅ 'Data Security' presente nel prompt")
else:
    print("   ❌ 'Data Security' NON presente nel prompt")

if "Access Control" in prompt:
    print("   ✅ 'Access Control' presente nel prompt")
else:
    print("   ❌ 'Access Control' NON presente nel prompt")
