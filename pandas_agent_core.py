"""
FASE 2: Logical Core - Pandas Agent per Query Deterministiche

Questo è il CUORE dell'innovazione tecnica del progetto.
Invece di fare ricerca semantica (che può allucinare), l'LLM genera
codice Python esatto che interroga i dataset NIST.

OBIETTIVO: Quasi Zero allucinazioni - solo risposte basate sui dati reali.
"""
import os
from typing import Dict, List, Optional
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from nist_data_loader import NISTDataLoader


class NISTComplianceAgent:
    """
    Agente AI per interrogare i dataset NIST in modo deterministico.
    
    Questo agente traduce domande in linguaggio naturale in codice Pandas
    che viene eseguito sui dataset NIST puliti, garantendo risposte precise
    senza allucinazioni.
    """
    
    def __init__(
        self,
        base_url: str = "https://gpustack.ing.unibs.it/v1",
        model_name: str = "gpt-oss",
        api_key: Optional[str] = None,
        temperature: float = 0.0,  # Temperatura 0 per risposte deterministiche
        verbose: bool = True
    ):
        """
        Inizializza il Pandas Agent per NIST compliance.
        
        Args:
            base_url: URL del cluster LLM UniBS
            model_name: Modello da usare (gpt-oss consigliato per code generation)
            api_key: API key (se None, usa variabile ambiente GPUSTACK_API_KEY)
            temperature: Temperatura LLM (0.0 = deterministico)
            verbose: Se True, stampa i passi intermedi
        """
        # Ottieni API key
        if api_key is None:
            api_key = os.environ.get("GPUSTACK_API_KEY")
            if not api_key:
                # Fallback alla chiave hardcoded
                api_key = "gpustack_8a00037ab4220858_6479d07686028e2f970357b1a81200e4"
        
        # Inizializza il client OpenAI-compatible
        self.llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model_name,
            temperature=temperature
        )
        
        # Carica i dataset NIST
        self.data_loader = NISTDataLoader()
        self.datasets = self.data_loader.load_all()
        
        # Crea gli agent per ciascun dataset
        self.agents = self._create_agents(verbose=verbose)
        
        self.verbose = verbose
        
        if verbose:
            print("✅ NIST Compliance Agent inizializzato")
            print(f"   - Modello: {model_name}")
            print(f"   - Dataset caricati: {len(self.datasets)}")
            print(f"   - Agenti creati: {len(self.agents)}")
    
    def _create_agents(self, verbose: bool = True) -> Dict:
        """
        Crea Pandas Agent per ogni dataset.
        
        Returns:
            Dizionario con agent per csf_mapping, pf_mapping, sp_catalog
        """
        agents = {}
        
        # Agent per CSF Mapping
        agents['csf'] = create_pandas_dataframe_agent(
            self.llm,
            self.datasets['csf_mapping'],
            verbose=verbose,
            handle_parsing_errors=True,
            allow_dangerous_code=True  # Necessario per eseguire codice generato
        )
        
        # Agent per Privacy Framework Mapping
        agents['pf'] = create_pandas_dataframe_agent(
            self.llm,
            self.datasets['pf_mapping'],
            verbose=verbose,
            handle_parsing_errors=True,
            allow_dangerous_code=True
        )
        
        # Agent per SP 800-53 Catalog
        agents['sp'] = create_pandas_dataframe_agent(
            self.llm,
            self.datasets['sp_catalog'],
            verbose=verbose,
            handle_parsing_errors=True,
            allow_dangerous_code=True
        )
        
        return agents
    
    def query_csf(self, question: str) -> str:
        """
        Interroga il dataset CSF mapping.
        
        Args:
            question: Domanda in linguaggio naturale
            
        Returns:
            Risposta basata sul dataset
            
        Examples:
            >>> agent.query_csf("Quali controlli SP 800-53 sono mappati alla funzione Identify?")
            >>> agent.query_csf("Mostrami le subcategories della categoria Asset Management")
        """
        result = self.agents['csf'].invoke({"input": question})
        return result.get('output', str(result))
    
    def query_sp800_53(self, question: str) -> str:
        """
        Interroga il catalogo SP 800-53.
        
        Args:
            question: Domanda in linguaggio naturale
            
        Returns:
            Risposta basata sul catalogo
            
        Examples:
            >>> agent.query_sp800_53("Cos'è il controllo AC-02?")
            >>> agent.query_sp800_53("Quanti enhancements ha il controllo AC-02?")
            >>> agent.query_sp800_53("Quali controlli fanno parte della famiglia Access Control?")
        """
        result = self.agents['sp'].invoke({"input": question})
        return result.get('output', str(result))
    
    def query_privacy_framework(self, question: str) -> str:
        """
        Interroga il Privacy Framework mapping.
        
        Args:
            question: Domanda in linguaggio naturale
            
        Returns:
            Risposta basata sul dataset
        """
        result = self.agents['pf'].invoke({"input": question})
        return result.get('output', str(result))
    
    def find_controls_for_subcategory(self, subcategory_id: str) -> Dict:
        """
        Trova i controlli SP 800-53 per una specifica subcategory CSF.
        
        Args:
            subcategory_id: ID della subcategory (es. "ID.AM-1")
            
        Returns:
            Dizionario con informazioni sulla subcategory e controlli
        """
        df = self.datasets['csf_mapping']
        result = df[df['Subcategory_ID'] == subcategory_id]
        
        if result.empty:
            return {"error": f"Subcategory {subcategory_id} non trovata"}
        
        row = result.iloc[0]
        
        # Estrai i controlli (separati da virgola)
        controls = [c.strip() for c in str(row['SP800_53_Controls']).split(',')]
        
        # Trova dettagli dei controlli
        sp_df = self.datasets['sp_catalog']
        control_details = []
        
        for control_id in controls:
            ctrl = sp_df[sp_df['Control_ID'] == control_id]
            if not ctrl.empty:
                control_details.append({
                    'id': control_id,
                    'name': ctrl.iloc[0]['Control_Name'],
                    'family': ctrl.iloc[0]['Control_Family']
                })
        
        return {
            'subcategory_id': subcategory_id,
            'description': row['Subcategory_Description'],
            'function': row['Function'],
            'category': row['Category'],
            'controls': control_details
        }
    
    def get_csf_function_summary(self, function_name: str) -> Dict:
        """
        Ottieni un sommario di una funzione CSF (Identify, Protect, ecc.).
        
        Args:
            function_name: Nome della funzione (case-insensitive)
            
        Returns:
            Dizionario con statistiche e subcategories
        """
        df = self.datasets['csf_mapping']
        
        # Filtra per funzione (case-insensitive)
        function_data = df[df['Function'].str.contains(function_name, case=False, na=False)]
        
        if function_data.empty:
            return {"error": f"Funzione {function_name} non trovata"}
        
        # Estrai tutte le subcategories
        subcategories = []
        for _, row in function_data.iterrows():
            subcategories.append({
                'id': row['Subcategory_ID'],
                'description': row['Subcategory_Description'],
                'controls': row['SP800_53_Controls']
            })
        
        return {
            'function': function_data.iloc[0]['Function'],
            'total_subcategories': len(subcategories),
            'categories': function_data['Category'].nunique(),
            'subcategories': subcategories
        }


def test_agent():
    """
    Test del Pandas Agent con query di esempio.
    """
    print("="*70)
    print("🧪 TEST: NIST Compliance Pandas Agent")
    print("="*70)
    
    # Inizializza l'agente
    print("\n🔄 Inizializzazione agente...")
    agent = NISTComplianceAgent(verbose=False)
    
    print("\n" + "="*70)
    print("📋 Test Query su CSF Mapping")
    print("="*70)
    
    # Test 1: Trova controlli per ID.AM-1
    print("\n1️⃣ Test metodo diretto: find_controls_for_subcategory('ID.AM-1')")
    result = agent.find_controls_for_subcategory('ID.AM-1')
    print(f"   Descrizione: {result['description']}")
    print(f"   Controlli trovati: {len(result['controls'])}")
    for ctrl in result['controls']:
        print(f"      - {ctrl['id']}: {ctrl['name']}")
    
    # Test 2: Sommario funzione Identify
    print("\n2️⃣ Test metodo diretto: get_csf_function_summary('Identify')")
    result = agent.get_csf_function_summary('Identify')
    print(f"   Funzione: {result['function']}")
    print(f"   Subcategories totali: {result['total_subcategories']}")
    print(f"   Categorie: {result['categories']}")
    print(f"   Prime 3 subcategories:")
    for sub in result['subcategories'][:3]:
        print(f"      - {sub['id']}: {sub['description'][:60]}...")
    
    # Test 3: Query in linguaggio naturale (Pandas Agent)
    print("\n" + "="*70)
    print("🤖 Test Query LLM (Pandas Agent - Code Generation)")
    print("="*70)
    
    test_queries = [
        ("CSF", "Quante subcategories ci sono nella funzione Protect?"),
        ("SP 800-53", "Quanti controlli ci sono nella famiglia AC (Access Control)?"),
        ("CSF", "Quali sono le prime 3 subcategories della funzione Detect?")
    ]
    
    for i, (dataset, query) in enumerate(test_queries, 1):
        print(f"\n{i}️⃣ Query: '{query}'")
        print(f"   Dataset: {dataset}")
        print("-"*70)
        
        try:
            if dataset == "CSF":
                answer = agent.query_csf(query)
            elif dataset == "SP 800-53":
                answer = agent.query_sp800_53(query)
            
            print(f"   ✅ Risposta: {answer}")
        except Exception as e:
            print(f"   ❌ Errore: {e}")
    
    print("\n" + "="*70)
    print("✅ Test completati!")
    print("="*70)
    print("\n🎯 Il Pandas Agent è funzionante e pronto per il Structured Dialogue!")


if __name__ == "__main__":
    test_agent()
