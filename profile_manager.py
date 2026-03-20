"""
Struttura del CSV generato (client_profile_state.csv):
Il file funge da "database di stato" per profilare un'organizzazione secondo il framework NIST CSF 2.0.
Combina il catalogo base delle regole ("Function", "Category", "Subcategory_ID", "Implementation_Examples") 
con le 17 colonne specifiche del "CSF 2.0 Organizational Profile Template" ufficiale.

Aggiunge inoltre una colonna finale: "Completion_Status".
- "Completion_Status" (valori: PENDING, IN_PROGRESS, DONE) viene usata per guidare 
  iterativamente la LLM senza perdere il contesto o i progressi salvati (State Machine).
"""

import pandas as pd
import os

class ProfileManager:
    def __init__(self, catalog_path='data/cleaned/csf_2_0_catalog.csv', save_dir='data/cleaned', profile_name='client_profile_state.csv'):
        self.catalog_path = catalog_path
        self.save_dir = save_dir
        self.profile_name = profile_name
        self.state_path = os.path.join(self.save_dir, self.profile_name)
        self.df = None
        self._initialize_state()

    def _initialize_state(self):
        """
        Inizializza il file di stato. Se esiste, lo carica. 
        Altrimenti, clona il catalogo CSF e aggiunge le colonne 
        necessarie per l'Organizational Profile e il tracciamento dello stato.
        """
        if os.path.exists(self.state_path):
            self.df = pd.read_csv(self.state_path)
        else:
            print("Creazione del nuovo file di stato da zero...")
            if not os.path.exists(self.catalog_path):
                raise FileNotFoundError(f"Il catalogo {self.catalog_path} non esiste.")
                
            self.df = pd.read_csv(self.catalog_path)
            
            # Aggiungiamo le colonne previste dal template e la colonna di stato
            profile_columns = [
                'Included_in_Profile',
                'Rationale',
                'Current_Priority',
                'Current_Status',
                'Current_Policies_Processes_Procedures',
                'Current_Internal_Practices',
                'Current_Roles_and_Responsibilities',
                'Current_Selected_Informative_References',
                'Current_Artifacts_and_Evidence',
                'Target_Priority',
                'Target_CSF_Tier', 
                'Target_Policies_Processes_Procedures',
                'Target_Internal_Practices',
                'Target_Roles_and_Responsibilities',
                'Target_Selected_Informative_References',
                'Notes',
                'Considerations',
                'Completion_Status' # Valori previsti: PENDING, IN_PROGRESS, DONE
            ]
            
            for col in profile_columns:
                if col == 'Completion_Status':
                    self.df[col] = 'PENDING'
                else:
                    self.df[col] = ''  # Inizializza come stringa vuota
                    
            self.save_state()

    def save_state(self):
        """Salva il DataFrame corrente nel file CSV di stato."""
        # Creiamo la cartella se non esiste
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        self.df.to_csv(self.state_path, index=False)
        print(f"Stato salvato in {self.state_path}")

    def get_next_pending(self):
        """
        Ritorna la prossima Subcategory (come dizionario) che è ancora in stato 'PENDING'.
        """
        pending_rows = self.df[self.df['Completion_Status'] == 'PENDING']
        if pending_rows.empty:
            return None
        return pending_rows.iloc[0].to_dict()

    def update_row(self, subcategory_id, updates_dict):
        """
        Aggiorna campi specifici di una riga (identificata dal subcategory_id).
        Ideale sia per compilare la prima volta, sia per le correzioni in fase di revisione.
        
        updates_dict = {'Target_Profile': '...', 'Completion_Status': 'DONE'}
        """
        if subcategory_id not in self.df['Subcategory_ID'].values:
            raise ValueError(f"Subcategory_ID '{subcategory_id}' non trovato nel database.")
            
        for col, value in updates_dict.items():
            if col in self.df.columns:
                self.df.loc[self.df['Subcategory_ID'] == subcategory_id, col] = value
            else:
                print(f"Attenzione: Colonna '{col}' non esiste nel DataFrame.")
                
        self.save_state()
        return True

    def get_progress_summary(self):
        """Ritorna un dizionario con il conteggio degli stati."""
        counts = self.df['Completion_Status'].value_counts().to_dict()
        total = len(self.df)
        completed = counts.get('DONE', 0)
        percentage = (completed / total) * 100 if total > 0 else 0
        
        return {
            'total_items': total,
            'completed': completed,
            'percentage': round(percentage, 2),
            **counts
        }

    def _test_compilazione_finta(self, max_rows=1):
        """
        Funzione di utilità per testare il salvataggio inserendo dati finti.
        Prende le prime `max_rows` in stato PENDING e le compila fittiziamente mettendole in 'DONE'.
        Non chiamata di default, usare solo per il debug manuale.
        """
        for _ in range(max_rows):
            next_task = self.get_next_pending()
            if next_task:
                sid = next_task['Subcategory_ID']
                print(f"Compilazione finta per: {sid}")
                updates = {
                    'Included_in_Profile': 'Yes',
                    'Current_Priority': 'Medium',
                    'Current_Policies_Processes_Procedures': 'Nessuna policy formale definita.',
                    'Target_Policies_Processes_Procedures': 'Creare una policy documentata e implementarla.',
                    'Completion_Status': 'DONE'
                }
                self.update_row(sid, updates)

if __name__ == "__main__":
    # Inizializza il Profile Manager. Da solo questo garantirà la creazione 
    # e/o lettura sicura del template senza alterare i valori in esso contenuti.
    manager = ProfileManager()
    
    # Stampa esclusivamente un riepilogo dello stato attuale
    print("\nRiepilogo stato corrente del Profilo Organizzativo:")
    print(manager.get_progress_summary())
    
    # manager._test_compilazione_finta(1) # Scommentare per inserire dati di prova
