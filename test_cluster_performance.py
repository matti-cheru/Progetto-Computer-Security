import time
import os
from openai import OpenAI
from dotenv import load_dotenv

# Carichiamo le variabili d'ambiente dal file .env
load_dotenv()

# Inizializziamo il client con la chiave fornita
client = OpenAI(
    base_url="https://gpustack.ing.unibs.it/v1",
    api_key=os.getenv("GPUSTACK_API_KEY"),
)

# Costruiamo un prompt che occupi approssimativamente 500 token
# Un token equivale a circa 3/4 di parola. Riempirò il prompt di testo tecnico per simulare
# le tue istruzioni e la descrizione della categoria dei controlli SP800-53.
dummy_context = (
    "In the context of NIST SP 800-53, Access Control (AC) focuses on limiting access to systems, "
    "equipment, and other resources to authorized users, processes, and devices. "
    "To comply with these controls, an organization must implement various security mechanisms. "
    "Some examples include role-based access control (RBAC), multifactor authentication (MFA), "
    "and continuous monitoring of user activities. Network access controls and firewalls are also critical. "
    "The environment includes a hybrid cloud infrastructure with Kubernetes clusters handling microservices. "
    "The dialogue system is responsible for retrieving relevant controls and structuring them into JSON. "
) * 8  # Ripetiamo per accumulare parole (circa 350-400 parole, ~500 token)

user_query = (
    "Please analyze the provided context regarding access control. "
    "Think step by step and provide an extremely detailed logical reasoning path before finalizing your output. "
    "I want you to consider potential edge cases where an internal actor might bypass role-based access controls "
    "due to misconfigured service accounts in the Kubernetes cluster. "
    "Provide a detailed step-by-step reasoning on how this could occur and how to mitigate it according to SP800-53."
)

prompt = f"{dummy_context}\n\n{user_query}"

messages = [
    {"role": "system", "content": "You are a senior cybersecurity expert specialized in NIST SP800-53. You must think extensively step by step before answering."},
    {"role": "user", "content": prompt}
]

models_to_test = ["qwen3", "phi4"]

for model_name in models_to_test:
    print(f"\n[{model_name}] Iniziando il test delle performance...")
    start_time = time.time()

    try:
        # Eseguiamo la richiesta
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=2048, # Uso 2048 per non aspettare iterazioni lunghissime in questo test.
        )
        end_time = time.time()
        
        duration = end_time - start_time
        usage = response.usage
        choice = response.choices[0]
        
        # Controlliamo la presenza del blocco reasoning
        reasoning_content = getattr(choice.message, 'reasoning_content', None)
        reasoning_len = len(reasoning_content) if reasoning_content else 0
        answer_len = len(choice.message.content) if choice.message.content else 0
        
        print("\n" + "="*40 + f"\nRISULTATI DEL TEST: {model_name.upper()}\n" + "="*40)
        print(f"Tempo totale: {duration:.2f} secondi")
        if usage:
            print(f"Token in ingresso (Prompt): {usage.prompt_tokens}")
            print(f"Token generati (Completion): {usage.completion_tokens}")
            print(f"Token totali: {usage.total_tokens}")
        
            if duration > 0:
                speed = usage.completion_tokens / duration
                print(f"Velocità di generazione (Token/sec): {speed:.2f} token/s")
        else:
            print("Nessuna informazione sui token restituita dal server.")
            
        print(f"Caratteri ragionamento (Reasoning): {reasoning_len}")
        print(f"Caratteri risposta finale: {answer_len}")
        print(f"Finish Reason: {choice.finish_reason}")
        print("="*40)
        
    except Exception as e:
        print(f"Errore durante la richiesta per {model_name}: {e}")
