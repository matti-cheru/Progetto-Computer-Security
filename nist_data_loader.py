"""
NIST Data Loader - Modulo centralizzato per caricare i dataset puliti

Questo modulo fornisce funzioni per caricare i dataset NIST puliti
in modo consistente per il Pandas Agent (Fase 2).

IMPORTANTE: I dataset sono stati pre-processati e puliti da data_cleaning.py
"""
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Optional


class NISTDataLoader:
    """
    Caricatore centralizzato per i dataset NIST puliti.
    
    Garantisce accesso consistente ai dati per il Pandas Agent,
    evitando problemi di path, encoding, e inconsistenze.
    """
    
    def __init__(self, data_dir: str = "data/cleaned"):
        """
        Inizializza il data loader.
        
        Args:
            data_dir: Directory contenente i file CSV puliti
        """
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Directory dati non trovata: {self.data_dir}\n"
                f"Esegui prima data_cleaning.py per preparare i dati."
            )
        
        # Cache per evitare riletture multiple
        self._cache = {}
    
    def load_csf_mapping(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Carica il mapping CSF 1.1 → SP 800-53 Rev 5.
        
        Returns:
            DataFrame con colonne:
            - Function: Funzione CSF (Identify, Protect, Detect, Respond, Recover)
            - Category: Categoria CSF (es. Asset Management)
            - Subcategory_ID: ID subcategory (es. ID.AM-1)
            - Subcategory_Description: Descrizione della subcategory
            - SP800_53_Controls: Controlli SP 800-53 correlati (es. "CM-8, PM-5")
        """
        if use_cache and 'csf_mapping' in self._cache:
            return self._cache['csf_mapping']
        
        filepath = self.data_dir / "csf_to_sp800_53_mapping.csv"
        df = pd.read_csv(filepath)
        
        if use_cache:
            self._cache['csf_mapping'] = df
        
        return df
    
    def load_pf_mapping(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Carica il mapping Privacy Framework → SP 800-53 Rev 5.
        
        Returns:
            DataFrame con colonne:
            - Function: Funzione Privacy Framework
            - Category: Categoria
            - Subcategory_ID: ID subcategory (es. ID.IM-P1)
            - Subcategory_Description: Descrizione
            - SP800_53_Controls: Controlli SP 800-53 correlati
            - Relationship_to_CSF: Relazione con CSF
        """
        if use_cache and 'pf_mapping' in self._cache:
            return self._cache['pf_mapping']
        
        filepath = self.data_dir / "pf_to_sp800_53_mapping.csv"
        df = pd.read_csv(filepath)
        
        if use_cache:
            self._cache['pf_mapping'] = df
        
        return df
    
    def load_sp800_53_catalog(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Carica il catalogo completo SP 800-53 Rev 5.2.0.
        
        Returns:
            DataFrame con colonne:
            - Control_Family: Famiglia del controllo (AC, AT, AU, ecc.)
            - Control_ID: Identificativo controllo (es. AC-01, AC-02(01))
            - Control_Name: Nome del controllo
            - Control_Statement: Statement completo
            - Discussion: Discussione/contesto
            - Related_Controls: Controlli correlati
            - Privacy_Baseline: Baseline privacy (x se applicabile)
            - Security_Baseline_Low/Moderate/High: Baseline security
            - Reference: Riferimenti esterni
            - Is_Enhancement: Boolean (True se è un enhancement)
            - Base_Control_ID: ID del controllo base (es. AC-02 per AC-02(01))
        """
        if use_cache and 'sp_catalog' in self._cache:
            return self._cache['sp_catalog']
        
        filepath = self.data_dir / "sp800_53_catalog.csv"
        df = pd.read_csv(filepath)
        
        if use_cache:
            self._cache['sp_catalog'] = df
        
        return df
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Carica tutti i dataset in un dizionario.
        
        Returns:
            Dizionario con chiavi:
            - 'csf_mapping': CSF → SP 800-53 mapping
            - 'pf_mapping': Privacy Framework → SP 800-53 mapping
            - 'sp_catalog': Catalogo SP 800-53 completo
        """
        return {
            'csf_mapping': self.load_csf_mapping(),
            'pf_mapping': self.load_pf_mapping(),
            'sp_catalog': self.load_sp800_53_catalog()
        }
    
    def get_info(self) -> Dict[str, any]:
        """
        Restituisce informazioni sui dataset caricati.
        
        Returns:
            Dizionario con statistiche sui dataset
        """
        datasets = self.load_all()
        
        info = {
            'csf_mapping': {
                'rows': len(datasets['csf_mapping']),
                'functions': datasets['csf_mapping']['Function'].nunique(),
                'categories': datasets['csf_mapping']['Category'].nunique(),
                'subcategories': datasets['csf_mapping']['Subcategory_ID'].nunique(),
            },
            'pf_mapping': {
                'rows': len(datasets['pf_mapping']),
                'categories': datasets['pf_mapping']['Category'].nunique(),
            },
            'sp_catalog': {
                'rows': len(datasets['sp_catalog']),
                'families': datasets['sp_catalog']['Control_Family'].nunique(),
                'base_controls': (~datasets['sp_catalog']['Is_Enhancement']).sum(),
                'enhancements': datasets['sp_catalog']['Is_Enhancement'].sum(),
            }
        }
        
        return info
    
    def clear_cache(self):
        """Pulisce la cache dei dataset caricati."""
        self._cache.clear()


# Funzioni di utilità per accesso rapido
def load_csf_mapping() -> pd.DataFrame:
    """Shortcut per caricare il mapping CSF."""
    loader = NISTDataLoader()
    return loader.load_csf_mapping()

def load_pf_mapping() -> pd.DataFrame:
    """Shortcut per caricare il mapping Privacy Framework."""
    loader = NISTDataLoader()
    return loader.load_pf_mapping()

def load_sp800_53_catalog() -> pd.DataFrame:
    """Shortcut per caricare il catalogo SP 800-53."""
    loader = NISTDataLoader()
    return loader.load_sp800_53_catalog()


def load_all_nist_data() -> Dict[str, pd.DataFrame]:
    """Shortcut per caricare tutti i dataset."""
    loader = NISTDataLoader()
    return loader.load_all()


# Script di test per verificare il caricamento
if __name__ == "__main__":
    print("="*70)
    print("TEST: NIST Data Loader")
    print("="*70)
    
    try:
        loader = NISTDataLoader()
        
        print("\n📊 Informazioni sui dataset:")
        print("-"*70)
        
        info = loader.get_info()
        
        print("\n🔷 CSF Mapping:")
        print(f"   - Righe totali: {info['csf_mapping']['rows']}")
        print(f"   - Funzioni: {info['csf_mapping']['functions']}")
        print(f"   - Categorie: {info['csf_mapping']['categories']}")
        print(f"   - Subcategorie: {info['csf_mapping']['subcategories']}")
        
        print("\n🔷 Privacy Framework Mapping:")
        print(f"   - Righe totali: {info['pf_mapping']['rows']}")
        print(f"   - Categorie: {info['pf_mapping']['categories']}")
        
        print("\n🔷 SP 800-53 Catalog:")
        print(f"   - Righe totali: {info['sp_catalog']['rows']}")
        print(f"   - Control Families: {info['sp_catalog']['families']}")
        print(f"   - Base Controls: {info['sp_catalog']['base_controls']}")
        print(f"   - Enhancements: {info['sp_catalog']['enhancements']}")
        
        # Test query di esempio
        print("\n" + "="*70)
        print("🧪 Test Query Pandas")
        print("="*70)
        
        csf_df = loader.load_csf_mapping()
        
        print("\n1️⃣ Trova tutti i controlli per la funzione 'Identify':")
        identify_controls = csf_df[csf_df['Function'].str.contains('Identify', case=False)]
        print(f"   Risultati: {len(identify_controls)} subcategories")
        print(f"   Esempio: {identify_controls.iloc[0]['Subcategory_ID']} → {identify_controls.iloc[0]['SP800_53_Controls']}")
        
        pf_df = loader.load_pf_mapping()
        
        print("\n2️⃣ Trova la relazone col CSF della subcategory Privacy Framework ID.IM-P1:")
        pf_id_im_1 = pf_df[pf_df['Subcategory_ID'] == 'ID.IM-P1']
        if not pf_id_im_1.empty:
            print(f"   Relationship: {pf_id_im_1.iloc[0]['Relationship_to_CSF']}")
            print(f"   Controlli: {pf_id_im_1.iloc[0]['SP800_53_Controls']}")

        sp_df = loader.load_sp800_53_catalog()
        
        print("\n3️⃣ Trova il controllo AC-02 (SP 800-53):")
        ac02 = sp_df[sp_df['Control_ID'] == 'AC-02']
        if not ac02.empty:
            print(f"   Nome: {ac02.iloc[0]['Control_Name']}")
            print(f"   Famiglia: {ac02.iloc[0]['Control_Family']}")
        
        print("\n4️⃣ Conta enhancements del controllo AC-02:")
        ac02_enhancements = sp_df[sp_df['Base_Control_ID'] == 'AC-02']['Is_Enhancement'].sum()
        print(f"   Enhancements di AC-02: {ac02_enhancements}")
        
        print("\n" + "="*70)
        print("✅ Tutti i test completati con successo!")
        print("="*70)
        print("\n🎯 Il data loader è pronto per il Pandas Agent!")
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
