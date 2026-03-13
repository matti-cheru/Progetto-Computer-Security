"""
TEST: Pandas Agent con libreria OpenAI nativa invece di LangChain

Obiettivo: Testare il Pandas Agent "manuale" basato sulla libreria openai nativa,
salvando correttamente i log divisi per run e includendo correzioni ai parsing (es. NoneType handling).
"""
import os
from datetime import datetime
import pandas as pd
from nist_data_loader import NISTDataLoader
from pandas_agent_manual import ManualPandasAgent

def test_simple_query(agent: ManualPandasAgent, df: pd.DataFrame, logs_dir: str):
    """Test con query semplice"""
    print("\n" + "#"*80)
    print("#" + " "*25 + "TEST 1: Query Semplice - Conta Righe" + " "*22 + "#")
    print("#"*80 + "\n")
    
    print(f"📊 Dataset: {len(df)} righe × {len(df.columns)} colonne\n")
    
    # Query
    question = "How many rows are in this dataframe?"
    print(f"❓ DOMANDA: {question}\n")
    
    # Log configuration
    log_file = os.path.join(logs_dir, "test1_simple_query.json")
    
    # Esegue query con log esportato
    result = agent.query(df=df, question=question, max_iterations=3, log_to_json=log_file)
    
    print("\n" + "="*80)
    print("RISPOSTA FINALE")
    print("="*80)
    print(f"   {result.get('answer')}")
    
    # Verifica
    actual = len(df)
    print(f"\n✅ VERIFICA: Righe effettive = {actual}")
    print(f"📁 Log dettagliato salvato in: {log_file}")


def test_filter_query(agent: ManualPandasAgent, df: pd.DataFrame, logs_dir: str):
    """Test con query con filtro"""
    print("\n\n" + "#"*80)
    print("#" + " "*20 + "TEST 2: Query con Filtro - Conta Identify" + " "*21 + "#")
    print("#"*80 + "\n")
    
    print(f"📊 Dataset: {len(df)} righe × {len(df.columns)} colonne\n")
    
    # Query
    question = "how many subcategories are part of the function Protect?"
    print(f"❓ DOMANDA: {question}\n")
    
    # Log configuration
    log_file = os.path.join(logs_dir, "test2_filter_query.json")
    
    # Esegue query con log esportato
    result = agent.query(df=df, question=question, max_iterations=3, log_to_json=log_file)
    
    print("\n" + "="*80)
    print("RISPOSTA FINALE")
    print("="*80)
    print(f"   {result.get('answer')}")
    
    # Verifica
    actual = df['Function'].str.contains('Protect', case=False).sum()
    print(f"\n✅ VERIFICA: Subcategories Protect effettive = {actual}")
    print(f"📁 Log dettagliato salvato in: {log_file}")


def main():
    """Main function"""
    
    print("\n" + "#"*80)
    print("#" + " "*15 + "PANDAS AGENT MANUALE con OpenAI Nativa" + " "*24 + "#")  
    print("#"*80)
    
    print("\n📝 Questo script testa il ManualPandasAgent ufficiale:")
    print("   - Utilizza la classe centrale già collaudata in pandas_agent_manual.py")
    print("   - Organizza i log nelle directory isolate run_YYYYMMDD_HHMMSS")
    print("   - Risolve i crash su contenuti vuoti durante la generazione")
    
    # 1. Setup delle directory di log uniche
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.getcwd(), f"run_{timestamp_str}")
    os.makedirs(run_dir, exist_ok=True)
    pandas_logs_dir = os.path.join(run_dir, "pandas_agent_logs")
    os.makedirs(pandas_logs_dir, exist_ok=True)
    print(f"\n📁 Creata directory per resoconto run: {run_dir}")
    
    # 2. Inizializza Agente e Carica i Dati
    try:
        agent = ManualPandasAgent(verbose=True)
        loader = NISTDataLoader()
        csf_df = loader.load_csf_mapping()
        
        # 3. Esegui i test incapsulati che persistono i json nella directory
        test_simple_query(agent, csf_df, pandas_logs_dir)
        test_filter_query(agent, csf_df, pandas_logs_dir)
        
        print("\n\n" + "#"*80)
        print("#" + " "*30 + "TEST COMPLETATI" + " "*33 + "#")
        print("#"*80)
        print(f"\nTutti i log della run sono disponibili in: {run_dir}\n")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrotto dall'utente")
    except Exception as e:
        print(f"\n\n❌ ERRORE IMPREVISTO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
