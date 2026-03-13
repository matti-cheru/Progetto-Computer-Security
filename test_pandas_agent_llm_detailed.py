"""
TEST: Pandas Agent con LLM - Generazione Codice

Questo script mostra ESATTAMENTE come funziona il Pandas Agent quando usa l'LLM:
1. Riceve una domanda in linguaggio naturale
2. L'LLM genera codice Python/Pandas
3. Il codice viene eseguito sul DataFrame
4. Restituisce la risposta

NOTA: verbose=True mostrerà tutti i passi intermedi!
"""
import os
import sys
import pandas as pd
from pandas_agent_core import NISTComplianceAgent


def print_header(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def test_simple_query():
    """Test con query semplice per vedere il processo"""
    print_header("TEST: Query Semplice con LLM Code Generation")
    
    print("🔧 Inizializzazione agente con verbose=True...")
    print("   (verbose=True mostrerà il codice generato dall'LLM)\n")
    
    # Crea agente con verbose=True
    agent = NISTComplianceAgent(verbose=True, temperature=0.0)
    
    print("✅ Agente pronto!\n")
    
    # Query 1: Contare righe (semplice)
    print_header("Query 1: 'Quante righe ci sono nel dataset CSF?'")
    
    print("💡 Cosa dovrebbe fare l'LLM:")
    print("   1. Capire che vogliamo il numero di righe")
    print("   2. Generare: df.shape[0] oppure len(df)")
    print("   3. Eseguire il codice")
    print("   4. Restituire il risultato\n")
    
    print("🤖 ESECUZIONE AGENT (con verbose=True)...")
    print("-"*80)
    
    try:
        # Accesso diretto al dataframe per riferimento
        actual_count = len(agent.datasets['csf_mapping'])
        print(f"\n📊 [Per riferimento] Numero reale di righe: {actual_count}\n")
        
        # Query con LLM
        answer = agent.query_csf("Quante righe ci sono nel dataset?")
        
        print("-"*80)
        print(f"\n✅ RISPOSTA FINALE: {answer}\n")
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        print("\nMotivi possibili:")
        print("   - Modello non supporta function calling")
        print("   - Output parsing error")
        print("   - Problema di connessione al cluster\n")
    
    # Query 2: Filtrare dati
    print_header("Query 2: 'Mostrami le subcategories della funzione Identify'")
    
    print("💡 Cosa dovrebbe fare l'LLM:")
    print("   1. Capire che vogliamo filtrare per Function == 'Identify'")
    print("   2. Generare: df[df['Function'].str.contains('Identify')]")
    print("   3. Eseguire e mostrare risultati\n")
    
    print("🤖 ESECUZIONE AGENT...")
    print("-"*80)
    
    try:
        # Confronto con metodo diretto
        identify_df = agent.datasets['csf_mapping'][
            agent.datasets['csf_mapping']['Function'].str.contains('Identify', case=False, na=False)
        ]
        print(f"\n📊 [Per riferimento] Subcategories Identify: {len(identify_df)}\n")
        print(f"   Esempi: {identify_df['Subcategory_ID'].head(3).tolist()}\n")
        
        # Query con LLM
        answer = agent.query_csf("Mostrami le prime 3 subcategories della funzione Identify")
        
        print("-"*80)
        print(f"\n✅ RISPOSTA FINALE:\n{answer}\n")
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}\n")


def test_with_manual_pandas():
    """Mostra come faremmo la stessa cosa senza LLM"""
    print_header("CONFRONTO: Stesso Risultato SENZA LLM (Pandas Puro)")
    
    agent = NISTComplianceAgent(verbose=False)
    df = agent.datasets['csf_mapping']
    
    print("📊 Query: 'Quante subcategories ha la funzione Protect?'\n")
    
    print("🐍 Soluzione 1: Pandas puro (senza AI)")
    print("   Codice:")
    print("   df = agent.datasets['csf_mapping']")
    print("   result = df[df['Function'].str.contains('Protect', case=False)]")
    print("   count = len(result)")
    
    result = df[df['Function'].str.contains('Protect', case=False, na=False)]
    count = len(result)
    
    print(f"\n   Risultato: {count} subcategories\n")
    
    print("-"*80)
    
    print("\n🤖 Soluzione 2: Con Pandas Agent (con AI)")
    print("   Codice:")
    print("   answer = agent.query_csf('Quante subcategories ha Protect?')")
    print("\n   Il modello LLM genera internamente codice simile a quello sopra")
    print("   e lo esegue automaticamente.\n")
    
    print("💡 VANTAGGIO del Pandas Agent:")
    print("   - Query in linguaggio naturale (non serve sapere Pandas)")
    print("   - Il codice viene generato automaticamente")
    print("   - Utile per domande complesse dove non sai quale codice scrivere")
    
    print("\n💡 SVANTAGGIO del Pandas Agent:")
    print("   - Più lento (richiede chiamata LLM)")
    print("   - Può fallire se il modello non capisce la domanda")
    print("   - Richiede connessione internet/cluster")
    
    print("\n🎯 STRATEGIA OTTIMALE:")
    print("   - Metodi diretti per query ripetitive e veloci")
    print("   - Pandas Agent per query esplorative e ad-hoc")
    print("   - Hybrid approach: Agent suggerisce il codice, poi lo riusiamo")


def show_dataframe_structure():
    """Mostra la struttura del DataFrame che l'LLM vede"""
    print_header("STRUTTURA DATI: Cosa 'Vede' il Pandas Agent")
    
    agent = NISTComplianceAgent(verbose=False)
    
    print("📊 Dataset CSF Mapping:")
    df = agent.datasets['csf_mapping']
    
    print(f"\n🔤 Colonne disponibili:")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        unique = df[col].nunique()
        print(f"   {i}. {col}")
        print(f"      - Tipo: {dtype}")
        print(f"      - Valori unici: {unique}")
        print(f"      - Valori nulli: {null_count}")
    
    print(f"\n📋 Prime 3 righe (esempio):")
    print(df.head(3).to_string())
    
    print("\n💡 L'LLM riceve:")
    print("   1. La struttura del DataFrame (colonne e tipi)")
    print("   2. Alcuni esempi di dati")
    print("   3. La domanda dell'utente")
    print("   4. Istruzioni su come generare codice Pandas")
    
    print("\n🎯 Output dell'LLM:")
    print("   Codice Python eseguibile che risponde alla domanda")
    print("   Es: df[df['Function'].str.contains('Identify')].shape[0]")


def main():
    """Main function"""
    print("\n" + "="*80)
    print(" "*15 + "PANDAS AGENT - DEEP DIVE: Come Funziona Internamente")
    print("="*80)
    
    print("\n⚠️  NOTA IMPORTANTE:")
    print("   Questo test richiede connessione al cluster LLM UniBS.")
    print("   Se il modello gpt-oss non supporta function calling,")
    print("   vedrai errori ma il concetto sarà comunque dimostrato.\n")
    
    try:
        # 1. Mostra struttura dati
        show_dataframe_structure()
        
        # 2. Test query semplici
        test_simple_query()
        
        # 3. Confronto con/senza LLM
        test_with_manual_pandas()
        
        print_header("TEST COMPLETATO")
        print("✅ Hai visto come funziona il Pandas Agent:")
        print("   1. Struttura del DataFrame")
        print("   2. Generazione codice dall'LLM")
        print("   3. Esecuzione e risultati")
        print("   4. Confronto con Pandas puro")
        
        print("\n🎓 CONCETTI CHIAVE:")
        print("   • Il Pandas Agent è un 'traduttore' da linguaggio naturale a codice")
        print("   • Non 'sa' le risposte - le calcola eseguendo codice sui dati")
        print("   • Zero allucinazioni perché usa solo i dati reali")
        print("   • Deterministico: stessa domanda = stesso codice = stessa risposta")
        
        print("\n" + "="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrotto dall'utente\n")
        return 1
    except Exception as e:
        print(f"\n❌ ERRORE FATALE: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
