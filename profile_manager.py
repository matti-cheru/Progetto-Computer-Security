"""
Structure of the generated CSV (client_profile_state.csv):
The file acts as a "state database" to profile an organization according to the NIST CSF 2.0 framework.
It combines the base rule catalog ("Function", "Category", "Subcategory_ID", "Implementation_Examples") 
with the 17 specific columns of the official "CSF 2.0 Organizational Profile Template".
.
"""

import pandas as pd
import os

class ProfileManager:
    def __init__(self, catalog_path='data/cleaned/csf_2_0_catalog.csv', save_dir='data/cleaned', profile_name='client_profile_state.csv', verbose=True):
        self.catalog_path = catalog_path
        self.save_dir = save_dir
        self.profile_name = profile_name
        self.state_path = os.path.join(self.save_dir, self.profile_name)
        self.verbose = verbose
        self.df = None
        self._initialize_state()

    def _initialize_state(self):
        """
        Initializes the state file. If it exists, it loads it.
        Otherwise, it clones the CSF catalog and adds the columns
        needed for the Organizational Profile and state tracking.
        """
        # Profile columns that must always be treated as strings (not float64)
        self._profile_columns = [
            'Included_in_Profile', 'Rationale', 'Current_Priority',
            'Current_Status', 'Current_Policies_Processes_Procedures',
            'Current_Internal_Practices', 'Current_Roles_and_Responsibilities',
            'Current_Selected_Informative_References', 'Current_Artifacts_and_Evidence',
            'Target_Priority', 'Target_CSF_Tier',
            'Target_Policies_Processes_Procedures', 'Target_Internal_Practices',
            'Target_Roles_and_Responsibilities', 'Target_Selected_Informative_References',
            'Notes', 'Considerations', 'Completion_Status',
        ]

        if os.path.exists(self.state_path):
            # Force string dtype on profile columns to prevent float64 inference
            # when columns are empty (all NaN → float64 by default)
            dtype_overrides = {col: str for col in self._profile_columns}
            self.df = pd.read_csv(self.state_path, dtype=dtype_overrides)
            # Replace any residual NaN with empty string
            self.df[self._profile_columns] = self.df[self._profile_columns].fillna('')
        else:
            print("Creating new state file from scratch...")
            if not os.path.exists(self.catalog_path):
                raise FileNotFoundError(f"The catalog {self.catalog_path} does not exist.")
                
            self.df = pd.read_csv(self.catalog_path)
            
            # Add the columns required by the template and the state column
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
                'Completion_Status' # Expected values: PENDING, IN_PROGRESS, DONE
            ]
            
            for col in profile_columns:
                if col == 'Completion_Status':
                    self.df[col] = 'PENDING'
                else:
                    self.df[col] = ''  # Initialize as empty string
                    
            self.save_state()

    def save_state(self):
        """Saves the current DataFrame to the state CSV file."""
        # Create the folder if it doesn't exist
        parent = os.path.dirname(self.state_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.df.to_csv(self.state_path, index=False)
        if self.verbose:
            print(f"State saved in {self.state_path}")

    def create_fresh_copy(self, dest_path: str):
        """
        Creates a brand-new blank profile at dest_path from the catalog,
        and switches this manager to operate on that copy.
        """
        if not os.path.exists(self.catalog_path):
            raise FileNotFoundError(f"The catalog {self.catalog_path} does not exist.")
        self.df = pd.read_csv(self.catalog_path)
        for col in self._profile_columns:
            if col == 'Completion_Status':
                self.df[col] = 'PENDING'
            else:
                self.df[col] = ''
        self.state_path = dest_path
        self.save_state()
        return self.state_path

    def get_next_pending(self):
        """
        Returns the next Subcategory (as a dictionary) that is still in 'PENDING' status.
        """
        pending_rows = self.df[self.df['Completion_Status'] == 'PENDING']
        if pending_rows.empty:
            return None
        return pending_rows.iloc[0].to_dict()

    def update_row(self, subcategory_id, updates_dict):
        """
        Updates specific fields of a row (identified by subcategory_id).
        Ideal for both initial completion and corrections during review.
        
        updates_dict = {'Target_Profile': '...', 'Completion_Status': 'DONE'}
        """
        if subcategory_id not in self.df['Subcategory_ID'].values:
            raise ValueError(f"Subcategory_ID '{subcategory_id}' not found in the database.")
            
        for col, value in updates_dict.items():
            if col in self.df.columns:
                # LLM might occasionally return a list instead of a string.
                # Pandas expects a single scalar here, so we stringify lists.
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                    
                self.df.loc[self.df['Subcategory_ID'] == subcategory_id, col] = value
            else:
                print(f"Warning: Column '{col}' does not exist in the DataFrame.")
                
        self.save_state()
        return True

    def get_progress_summary(self):
        """Returns a dictionary with the status count."""
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
        Utility function to test saving by inserting dummy data.
        Takes the first `max_rows` in PENDING status and fictitiously fills them by putting them in 'DONE'.
        Not called by default, use only for manual debugging.
        """
        for _ in range(max_rows):
            next_task = self.get_next_pending()
            if next_task:
                sid = next_task['Subcategory_ID']
                print(f"Dummy compilation for: {sid}")
                updates = {
                    'Included_in_Profile': 'Yes',
                    'Current_Priority': 'Medium',
                    'Current_Policies_Processes_Procedures': 'No formal policy defined.',
                    'Target_Policies_Processes_Procedures': 'Create a documented policy and implement it.',
                    'Completion_Status': 'DONE'
                }
                self.update_row(sid, updates)

if __name__ == "__main__":
    # Initializes the Profile Manager. This alone will guarantee the creation 
    # and/or safe reading of the template without altering the values contained in it.
    manager = ProfileManager()
    
    # Prints exclusively a summary of the current state
    print("\nCurrent state summary of the Organizational Profile:")
    print(manager.get_progress_summary())
    
    # manager._test_compilazione_finta(1) # Uncomment to insert test data
