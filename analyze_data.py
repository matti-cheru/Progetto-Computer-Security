"""
Script per analizzare la struttura dei file NIST scaricati
"""
import pandas as pd
import os

print("="*70)
print("ANALISI FILE NIST - Fase 1: Preparazione Dati")
print("="*70)

data_dir = "data"
files = [
    "csf-pf-to-sp800-53r5-mappings.xlsx",
    "cprt_SP_800_53_5_2_0_03-01-2026.xlsx",
    "CSF 2.0 Organizational Profile Template.xlsx"
]

for filename in files:
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        print(f"\n❌ File non trovato: {filename}")
        continue
    
    print(f"\n{'='*70}")
    print(f"📄 Analisi: {filename}")
    print(f"{'='*70}")
    
    try:
        # Leggi i nomi dei fogli
        excel_file = pd.ExcelFile(filepath)
        print(f"\n📋 Fogli disponibili: {excel_file.sheet_names}")
        
        # Analizza il primo foglio
        first_sheet = excel_file.sheet_names[0]
        df = pd.read_excel(filepath, sheet_name=first_sheet, nrows=5)
        
        print(f"\n📊 Primo foglio: '{first_sheet}'")
        print(f"   Dimensioni: {df.shape[0]} righe × {df.shape[1]} colonne")
        print(f"\n🔤 Colonne ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i}. {repr(col)}")
        
        print(f"\n🔍 Prime 3 righe:")
        print(df.head(3).to_string())
        
        # Controlla celle vuote
        null_counts = df.isnull().sum()
        if null_counts.any():
            print(f"\n⚠️  Celle vuote nelle prime 5 righe:")
            for col, count in null_counts[null_counts > 0].items():
                print(f"   - {col}: {count}")
        
    except Exception as e:
        print(f"\n❌ Errore nell'analisi: {e}")

print("\n" + "="*70)
print("✅ Analisi completata!")
print("="*70)
