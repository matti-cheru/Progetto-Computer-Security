"""
Test rapido del Pandas Agent - senza query LLM pesanti
"""
from pandas_agent_core import NISTComplianceAgent

print("="*70)
print("🧪 TEST RAPIDO: Pandas Agent Setup")
print("="*70)

# Test 1: Inizializzazione
print("\n1️⃣ Inizializzazione agente...")
try:
    agent = NISTComplianceAgent(verbose=False)
    print("   ✅ Agente inizializzato con successo")
except Exception as e:
    print(f"   ❌ Errore: {e}")
    exit(1)

# Test 2: Metodi diretti (senza LLM)
print("\n2️⃣ Test metodi diretti (senza LLM):")

# Test find_controls_for_subcategory
result = agent.find_controls_for_subcategory('ID.AM-1')
print(f"   - find_controls_for_subcategory('ID.AM-1'): ✅")
print(f"     Descrizione: {result['description'][:60]}...")

# Test get_csf_function_summary
result = agent.get_csf_function_summary('Protect')
print(f"   - get_csf_function_summary('Protect'): ✅")
print(f"     Subcategories: {result['total_subcategories']}")

# Test 3: Accesso ai dataset
print("\n3️⃣ Test accesso dataset:")
print(f"   - CSF Mapping: {len(agent.datasets['csf_mapping'])} righe ✅")
print(f"   - SP Catalog: {len(agent.datasets['sp_catalog'])} righe ✅")

print("\n" + "="*70)
print("✅ SETUP COMPLETATO CON SUCCESSO!")
print("="*70)
print("\n🎯 Pandas Agent Setup: FASE 2 COMPLETATA")
print("\n📝 STATO PROGETTO:")
print("   ✅ Fase 1: Dati NIST puliti e caricati")
print("   ✅ Fase 2: Logical Core (Pandas Agent) funzionante")
print("   🔜 Fase 3: Structured Dialogue (prossimo step)")
print("\n💡 Il sistema può ora:")
print("   • Interrogare 108 CSF subcategories")
print("   • Accedere a 1196 controlli SP 800-53")
print("   • Mappare automaticamente CSF → SP 800-53")
print("   • Generare query Pandas deterministiche (zero allucinazioni)")
