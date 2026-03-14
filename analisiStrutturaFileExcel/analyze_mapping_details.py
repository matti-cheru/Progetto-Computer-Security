"""
Analisi dettagliata del file di mapping CSF -> SP 800-53
Questo è il file CRUCIALE per il Pandas Agent
"""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ANALISI DETTAGLIATA: File di Mapping CSF ↔ SP 800-53")
print("="*70)

# File di mapping principale
mapping_file = "data/csf-pf-to-sp800-53r5-mappings.xlsx"

# Analizza il foglio "CSF to SP 800-53r5"
print("\n" + "="*70)
print("📋 Foglio 1: 'CSF to SP 800-53r5'")
print("="*70)

df_csf = pd.read_excel(mapping_file, sheet_name='CSF to SP 800-53r5')
print(f"\n📊 Dimensioni: {df_csf.shape[0]} righe × {df_csf.shape[1]} colonne")

print(f"\n🔤 Colonne:")
for i, col in enumerate(df_csf.columns, 1):
    null_pct = (df_csf[col].isnull().sum() / len(df_csf)) * 100
    print(f"   {i}. '{col}' - {null_pct:.1f}% vuote")

print(f"\n🔍 Prime 10 righe:")
print(df_csf.head(10).to_string())

print(f"\n📈 Statistiche:")
print(f"   - Righe totali: {len(df_csf)}")
print(f"   - Righe completamente vuote: {df_csf.isnull().all(axis=1).sum()}")
print(f"   - Valori unici per colonna:")
for col in df_csf.columns:
    unique_count = df_csf[col].nunique()
    print(f"     • {col}: {unique_count}")

# Analizza il foglio "PF to SP 800-53r5"
print("\n" + "="*70)
print("📋 Foglio 2: 'PF to SP 800-53r5'")
print("="*70)

df_pf = pd.read_excel(mapping_file, sheet_name='PF to SP 800-53r5')
print(f"\n📊 Dimensioni: {df_pf.shape[0]} righe × {df_pf.shape[1]} colonne")

print(f"\n🔤 Colonne:")
for i, col in enumerate(df_pf.columns, 1):
    null_pct = (df_pf[col].isnull().sum() / len(df_pf)) * 100
    print(f"   {i}. '{col}' - {null_pct:.1f}% vuote")

print(f"\n🔍 Prime 10 righe:")
print(df_pf.head(10).to_string())

# Analizza SP 800-53 Catalog
print("\n" + "="*70)
print("📋 File: SP 800-53 Rev 5.2.0 Catalog")
print("="*70)

sp_file = "data/cprt_SP_800_53_5_2_0_03-01-2026.xlsx"
df_sp = pd.read_excel(sp_file, sheet_name='SP 800-53 Rev 5.2.0')

print(f"\n📊 Dimensioni: {df_sp.shape[0]} righe × {df_sp.shape[1]} colonne")
print(f"   - Controlli totali (non vuoti): {df_sp['Control (or Control Enhancement) Identifier'].notna().sum()}")

# Mostra alcuni esempi di controlli
print(f"\n🔍 Esempi di controlli:")
sample_controls = df_sp[df_sp['Control (or Control Enhancement) Identifier'].notna()].head(5)
for idx, row in sample_controls.iterrows():
    ctrl_id = row['Control (or Control Enhancement) Identifier']
    ctrl_name = row['Control (or Control Enhancement) Name']
    print(f"   • {ctrl_id}: {ctrl_name}")

print("\n" + "="*70)
print("✅ Analisi dettagliata completata!")
print("="*70)
