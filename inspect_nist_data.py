"""
Script per ispezionare il contenuto del dataset NIST.
Verifica perché le query restituiscono Empty DataFrame.
"""
import pandas as pd
from nist_data_loader import NISTDataLoader

print("="*80)
print("🔍 ISPEZIONE DATASET NIST")
print("="*80)

# Carica dati
loader = NISTDataLoader()
df = loader.load_csf_mapping()

print(f"\n📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"📋 Columns: {list(df.columns)}")

print("\n" + "="*80)
print("🔍 ANALISI COLONNA 'Subcategory_Description'")
print("="*80)

# Mostra prime 10 descrizioni
print("\n📝 Prime 10 descrizioni:")
print("-"*80)
for idx, desc in enumerate(df['Subcategory_Description'].head(10), 1):
    print(f"{idx:2d}. {desc}")

# Cerca "data protection" (case insensitive)
print("\n" + "="*80)
print("🔎 RICERCA: 'data protection'")
print("="*80)
result1 = df[df['Subcategory_Description'].str.contains('data protection', case=False, na=False)]
print(f"Trovate {len(result1)} righe")
if len(result1) > 0:
    print("\nRighe trovate:")
    print(result1[['Subcategory_ID', 'Subcategory_Description']].to_string())

# Cerca "access control" (case insensitive)
print("\n" + "="*80)
print("🔎 RICERCA: 'access control'")
print("="*80)
result2 = df[df['Subcategory_Description'].str.contains('access control', case=False, na=False)]
print(f"Trovate {len(result2)} righe")
if len(result2) > 0:
    print("\nRighe trovate:")
    print(result2[['Subcategory_ID', 'Subcategory_Description']].to_string())

# Cerca "data" (più generico)
print("\n" + "="*80)
print("🔎 RICERCA: 'data' (generico)")
print("="*80)
result3 = df[df['Subcategory_Description'].str.contains('data', case=False, na=False)]
print(f"Trovate {len(result3)} righe")
if len(result3) > 0:
    print("\nPrime 5 righe trovate:")
    print(result3[['Subcategory_ID', 'Subcategory_Description']].head().to_string())

# Cerca "access" (più generico)
print("\n" + "="*80)
print("🔎 RICERCA: 'access' (generico)")
print("="*80)
result4 = df[df['Subcategory_Description'].str.contains('access', case=False, na=False)]
print(f"Trovate {len(result4)} righe")
if len(result4) > 0:
    print("\nPrime 5 righe trovate:")
    print(result4[['Subcategory_ID', 'Subcategory_Description']].head().to_string())

# Mostra tutte le category uniche
print("\n" + "="*80)
print("📂 CATEGORIE UNICHE NEL DATASET")
print("="*80)
categories = df['Category'].unique()
print(f"Totale: {len(categories)} categorie\n")
for idx, cat in enumerate(categories, 1):
    count = len(df[df['Category'] == cat])
    print(f"{idx:2d}. {cat:50s} ({count:2d} subcategories)")

# Verifica query problematica originale
print("\n" + "="*80)
print("🔎 QUERY ORIGINALE PROBLEMATICA")
print("="*80)
print("Query: df[df['Subcategory_Description'].str.contains('data protection|access control', case=False)]")
result_original = df[df['Subcategory_Description'].str.contains('data protection|access control', case=False, na=False)]
print(f"Risultato: {len(result_original)} righe")
if len(result_original) > 0:
    print("\nRighe trovate:")
    print(result_original[['Subcategory_ID', 'Subcategory_Description']].to_string())
else:
    print("❌ EMPTY DATAFRAME - Nessuna riga trovata!")
    print("\n💡 SUGGERIMENTO:")
    print("   Le stringhe esatte 'data protection' e 'access control' non esistono")
    print("   nelle descrizioni NIST. Il modello deve cercare termini più generici")
    print("   come 'data' o 'access', oppure usare le Category invece che")
    print("   cercare nelle descrizioni.")

print("\n" + "="*80)
print("✅ Ispezione completata")
print("="*80)
