import pandas as pd
import os
filepath = "data\cleaned\csf_to_sp800_53_mapping.csv"
df = pd.read_csv(filepath)
results=len(df)
print(results)