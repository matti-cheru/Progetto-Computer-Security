"""
TEST RAW: Mostra esattamente cosa succede internamente nel Pandas Agent

Questo script intercetta e stampa:
1. Il prompt ESATTO inviato all'LLM
2. La risposta RAW dell'LLM
3. Il codice Python generato
4. L'output dell'esecuzione del codice
5. Errori e warning reali

NO print() fasulli che dicono "cosa dovrebbe succedere"
SOLO dati reali del sistema!
"""
import os
import sys
import pandas as pd
from typing import Any, Dict, List

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.callbacks import BaseCallbackHandler
from nist_data_loader import NISTDataLoader


class DetailedCallbackHandler(BaseCallbackHandler):
    """
    Callback personalizzato per intercettare TUTTO quello che succede
    """
    
    def __init__(self):
        self.step_counter = 0
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Chiamato quando l'LLM inizia a processare"""
        print("\n" + "="*80)
        print("🤖 LLM START - Invio prompt all'LLM")
        print("="*80)
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n📤 PROMPT #{i} (ESATTO come inviato al modello):")
            print("-"*80)
            print(prompt)
            print("-"*80)
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Chiamato quando l'LLM finisce di generare"""
        print("\n" + "="*80)
        print("✅ LLM END - Risposta ricevuta dal modello")
        print("="*80)
        
        # Estrai il testo della risposta
        if hasattr(response, 'generations'):
            for i, generation_list in enumerate(response.generations, 1):
                for j, generation in enumerate(generation_list, 1):
                    print(f"\n📥 RISPOSTA #{i}.{j} (RAW dall'LLM):")
                    print("-"*80)
                    if hasattr(generation, 'text'):
                        print(generation.text)
                    elif hasattr(generation, 'message'):
                        print(generation.message.content if hasattr(generation.message, 'content') else generation.message)
                    else:
                        print(generation)
                    print("-"*80)
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Chiamato quando l'LLM ha un errore"""
        print("\n" + "="*80)
        print("❌ LLM ERROR")
        print("="*80)
        print(f"Errore: {error}")
        print("-"*80)
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Chiamato quando viene eseguito un tool (es. python_repl_ast)"""
        self.step_counter += 1
        print("\n" + "="*80)
        print(f"🔧 TOOL START - Step {self.step_counter}")
        print("="*80)
        
        tool_name = serialized.get('name', 'Unknown')
        print(f"Tool: {tool_name}")
        print(f"\n📥 INPUT AL TOOL:")
        print("-"*80)
        print(input_str)
        print("-"*80)
    
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Chiamato quando un tool finisce l'esecuzione"""
        print("\n📤 OUTPUT DEL TOOL:")
        print("-"*80)
        print(output)
        print("-"*80)
    
    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        """Chiamato quando un tool ha un errore"""
        print("\n❌ TOOL ERROR:")
        print("-"*80)
        print(f"Errore: {error}")
        print("-"*80)
    
    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        """Chiamato quando l'agent decide un'azione"""
        print("\n" + "="*80)
        print("🎯 AGENT ACTION - Decisione dell'agente")
        print("="*80)
        print(f"Tool scelto: {action.tool}")
        print(f"Input al tool: {action.tool_input}")
        print(f"Log: {action.log}")
        print("-"*80)
    
    def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        """Chiamato quando l'agent finisce"""
        print("\n" + "="*80)
        print("🏁 AGENT FINISH - Completamento")
        print("="*80)
        print(f"Output finale: {finish.return_values}")
        print("-"*80)
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Chiamato quando inizia una chain"""
        print("\n" + "="*80)
        print("⛓️  CHAIN START")
        print("="*80)
        print(f"Input alla chain: {inputs}")
        print("-"*80)
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Chiamato quando finisce una chain"""
        print("\n" + "="*80)
        print("⛓️  CHAIN END")
        print("="*80)
        print(f"Output dalla chain: {outputs}")
        print("-"*80)


def test_simple_query_with_raw_output():
    """Test con una query semplice mostrando TUTTI i dettagli reali"""
    
    print("\n" + "="*100)
    print(" "*30 + "TEST: PANDAS AGENT - RAW OUTPUT")
    print(" "*20 + "Intercettazione completa di TUTTO quello che succede")
    print("="*100)
    
    # Carica i dati
    print("\n📊 STEP 1: Caricamento dati...")
    loader = NISTDataLoader()
    csf_df = loader.load_csf_mapping()
    print(f"   Dataset caricato: {len(csf_df)} righe × {len(csf_df.columns)} colonne")
    print(f"   Colonne: {list(csf_df.columns)}")
    
    # Mostra un sample dei dati
    print("\n📋 Sample dati (prime 2 righe):")
    print(csf_df.head(2).to_string())
    
    # Configura LLM
    print("\n\n🔧 STEP 2: Configurazione LLM...")
    api_key = os.environ.get("GPUSTACK_API_KEY", "gpustack_8a00037ab4220858_6479d07686028e2f970357b1a81200e4")
    
    llm = ChatOpenAI(
        base_url="https://gpustack.ing.unibs.it/v1",
        api_key=api_key,
        model="gpt-oss",
        temperature=0.0,
        verbose=True  # Verbose anche sul modello
    )
    
    print("   LLM configurato:")
    print(f"   - Model: gpt-oss")
    print(f"   - Temperature: 0.0")
    print(f"   - Base URL: https://gpustack.ing.unibs.it/v1")
    
    # Crea callback handler
    callback = DetailedCallbackHandler()
    
    # Crea agent con callback
    print("\n\n🤖 STEP 3: Creazione Pandas Agent con callback personalizzato...")
    
    agent = create_pandas_dataframe_agent(
        llm,
        csf_df,
        verbose=True,  # Verbose dell'agent
        allow_dangerous_code=True,
        return_intermediate_steps=True,  # Restituisce gli step intermedi
        max_iterations=5,
        early_stopping_method="generate"
    )
    
    print("   Agent creato con successo!")
    
    # Test Query 1: Semplicissima
    print("\n\n" + "="*100)
    print("🔍 QUERY 1: Conta righe (semplicissima)")
    print("="*100)
    
    query = "Quante righe ci sono in questo dataset?"
    print(f"\nDomanda: '{query}'")
    print("\n⏳ Invio query all'agent... (potrebbe richiedere 10-20 secondi)\n")
    
    try:
        # Esegui con callback
        result = agent.invoke(
            {"input": query},
            config={"callbacks": [callback]}
        )
        
        print("\n\n" + "="*100)
        print("📊 RISULTATO FINALE")
        print("="*100)
        print(f"Output: {result.get('output', 'N/A')}")
        
        if 'intermediate_steps' in result:
            print(f"\nStep intermedi: {len(result['intermediate_steps'])}")
            for i, (action, observation) in enumerate(result['intermediate_steps'], 1):
                print(f"\n  Step {i}:")
                print(f"    Action: {action.tool}")
                print(f"    Input: {action.tool_input}")
                print(f"    Output: {observation}")
        
    except Exception as e:
        print("\n\n❌❌❌ ERRORE DURANTE ESECUZIONE ❌❌❌")
        print(f"Tipo errore: {type(e).__name__}")
        print(f"Messaggio: {e}")
        
        import traceback
        print("\nStack trace completo:")
        print("-"*80)
        traceback.print_exc()
        print("-"*80)
    
    # Test Query 2: Con filtro
    print("\n\n" + "="*100)
    print("🔍 QUERY 2: Filtraggio (più complessa)")
    print("="*100)
    
    query2 = "Quante subcategories hanno 'Identify' nella colonna Function?"
    print(f"\nDomanda: '{query2}'")
    print("\n⏳ Invio query all'agent...\n")
    
    callback.step_counter = 0  # Reset counter
    
    try:
        result2 = agent.invoke(
            {"input": query2},
            config={"callbacks": [callback]}
        )
        
        print("\n\n" + "="*100)
        print("📊 RISULTATO FINALE")
        print("="*100)
        print(f"Output: {result2.get('output', 'N/A')}")
        
        # Verifica con codice diretto
        actual_count = len(csf_df[csf_df['Function'].str.contains('Identify', case=False, na=False)])
        print(f"\n✅ Verifica con Pandas diretto: {actual_count} subcategories")
        
    except Exception as e:
        print("\n\n❌❌❌ ERRORE DURANTE ESECUZIONE ❌❌❌")
        print(f"Tipo errore: {type(e).__name__}")
        print(f"Messaggio: {e}")
        
        import traceback
        print("\nStack trace completo:")
        traceback.print_exc()


def test_direct_llm_call():
    """Test chiamata diretta all'LLM per vedere se risponde"""
    
    print("\n\n" + "="*100)
    print(" "*30 + "TEST PRELIMINARE: LLM Diretto")
    print(" "*25 + "(Verifica se il modello risponde)")
    print("="*100)
    
    api_key = os.environ.get("GPUSTACK_API_KEY", "gpustack_8a00037ab4220858_6479d07686028e2f970357b1a81200e4")
    
    llm = ChatOpenAI(
        base_url="https://gpustack.ing.unibs.it/v1",
        api_key=api_key,
        model="gpt-oss",
        temperature=0.7
    )
    
    test_message = "Rispondi con: 'Test OK'. Non aggiungere altro."
    
    print(f"\n📤 Prompt inviato: '{test_message}'")
    print("\n⏳ Attendo risposta...")
    
    try:
        callback = DetailedCallbackHandler()
        response = llm.invoke(test_message, config={"callbacks": [callback]})
        
        print("\n\n✅ RISPOSTA RICEVUTA:")
        print("-"*80)
        print(response.content if hasattr(response, 'content') else response)
        print("-"*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    
    print("\n" + "#"*100)
    print("#" + " "*98 + "#")
    print("#" + " "*25 + "PANDAS AGENT - ANALISI COMPLETA RAW OUTPUT" + " "*30 + "#")
    print("#" + " "*98 + "#")
    print("#"*100)
    
    print("\n📝 Cosa farà questo script:")
    print("   1. Testerà la connessione diretta all'LLM")
    print("   2. Creerà un Pandas Agent con callback dettagliati")
    print("   3. Invierà query semplici")
    print("   4. Mostrerà ESATTAMENTE cosa viene inviato/ricevuto")
    print("   5. Mostrerà il codice Python generato (se generato)")
    print("   6. Mostrerà errori reali (se ci sono)")
    
    print("\n⚠️  NOTA: Potrebbero esserci errori! È normale se il modello non")
    print("   supporta function calling. L'obiettivo è vedere COSA SUCCEDE.")
    
    input("\n▶️  Premi ENTER per iniziare...")
    
    # Test 1: LLM diretto
    llm_ok = test_direct_llm_call()
    
    if not llm_ok:
        print("\n⚠️  Il modello LLM non risponde correttamente.")
        print("   Continuo comunque con il test del Pandas Agent...")
        input("\n▶️  Premi ENTER per continuare...")
    
    # Test 2: Pandas Agent completo
    test_simple_query_with_raw_output()
    
    print("\n\n" + "#"*100)
    print("#" + " "*98 + "#")
    print("#" + " "*40 + "TEST COMPLETATO" + " "*43 + "#")
    print("#" + " "*98 + "#")
    print("#"*100)
    
    print("\n📊 Ora hai visto:")
    print("   ✓ I prompt ESATTI inviati all'LLM")
    print("   ✓ Le risposte RAW dell'LLM")
    print("   ✓ Il codice Python generato (se generato)")
    print("   ✓ Gli errori reali (se presenti)")
    print("   ✓ Il flusso completo di esecuzione")
    
    print("\n💡 Se hai visto errori di parsing, significa che:")
    print("   - Il modello gpt-oss potrebbe non supportare function calling")
    print("   - Serve un modello con migliore support per agents")
    print("   - Puoi usare metodi diretti Pandas invece del Pandas Agent")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrotto dall'utente!")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ ERRORE FATALE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
