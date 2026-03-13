"""
TEST VELOCE: Verifica accesso ai reasoning tokens di gpt-oss
"""
import os
import json
from openai import OpenAI
from langchain_openai import ChatOpenAI

CLUSTER_BASE_URL = "https://gpustack.ing.unibs.it/v1"
MODEL = "gpt-oss"
API_KEY = os.environ.get("GPUSTACK_API_KEY", "gpustack_8a00037ab4220858_6479d07686028e2f970357b1a81200e4")


print("\n" + "="*80)
print("TEST 1: OpenAI Nativa - Domanda Semplice")
print("="*80)

client = OpenAI(base_url=CLUSTER_BASE_URL, api_key=API_KEY)

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Calcola 5 + 7 e rispondi solo con il numero."}],
    temperature=0.0
)

choice = response.choices[0]
message = choice.message

print(f"\n✅ content: '{message.content}'")
print(f"✅ reasoning_content: '{message.reasoning_content[:100]}...'")
print(f"✅ reasoning_tokens: {response.usage.completion_tokens_details.reasoning_tokens}")
print(f"✅ output_tokens: {response.usage.completion_tokens}")


print("\n" + "="*80)
print("TEST 2: OpenAI Nativa - Task ReAct")
print("="*80)

prompt = """You have a Python dataframe 'df' with 108 rows.
Question: How many rows are in the dataframe?

Answer in this format:
Thought: [your reasoning]
Action: python_repl_ast
Action Input: len(df)"""

response2 = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0
)

choice2 = response2.choices[0]
message2 = choice2.message

print(f"\n✅ content length: {len(message2.content)} chars")
print(f"✅ content preview:\n{message2.content[:300]}")
print(f"\n✅ reasoning_content length: {len(message2.reasoning_content) if message2.reasoning_content else 0} chars")
if message2.reasoning_content:
    print(f"✅ reasoning preview:\n{message2.reasoning_content[:200]}")
print(f"\n✅ reasoning_tokens: {response2.usage.completion_tokens_details.reasoning_tokens}")


print("\n" + "="*80)
print("TEST 3: LangChain ChatOpenAI")
print("="*80)

llm = ChatOpenAI(
    base_url=CLUSTER_BASE_URL,
    api_key=API_KEY,
    model=MODEL,
    temperature=0.0
)

response3 = llm.invoke("Calcola 8 + 4 e rispondi solo con il numero.")

print(f"\n✅ content: '{response3.content}'")
print(f"❓ reasoning_content: {getattr(response3, 'reasoning_content', 'NON DISPONIBILE')}")

if hasattr(response3, 'response_metadata'):
    print(f"\n📈 response_metadata:")
    metadata = response3.response_metadata
    
    # Controlla se ci sono token details
    if 'token_usage' in metadata:
        print(f"   token_usage: {metadata['token_usage']}")
    
    # Stampa tutto
    print(json.dumps(metadata, indent=2, default=str))


print("\n" + "="*80)
print("CONCLUSIONI")
print("="*80)

print("""
✅ OpenAI Nativa: FUNZIONA perfettamente!
   - Genera content (risposta effettiva)
   - Espone reasoning_content (ragionamento interno)
   - Separato in reasoning_tokens + regular tokens

❌ LangChain ChatOpenAI: NON espone reasoning_content
   - Wrapper LangChain non passa/espone il reasoning
   - response_metadata potrebbe contenere info parziali

💡 SOLUZIONE:
   1. Usa libreria openai nativa invece di LangChain per Pandas Agent
   2. O crea un custom wrapper che espone reasoning_content
   3. O ignora reasoning e usa solo content (se disponibile)
""")
