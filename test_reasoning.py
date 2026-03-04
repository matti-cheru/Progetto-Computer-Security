import os
from openai import OpenAI

# Configuration
CLUSTER_BASE_URL = "https://gpustack.ing.unibs.it/v1"
TARGET_MODEL = "gpt-oss"  # Prova anche con: "qwen3", "phi4", "phi4-mini"

def test_reasoning():
    """
    Test specifico per verificare il reasoning del modello.
    Usa un problema che richiede ragionamento passo-passo.
    """
    print(f"--- Test Reasoning con {TARGET_MODEL} ---\n")
    
    api_key = os.environ.get("GPUSTACK_API_KEY")
    
    if not api_key:
        print("ERROR: Imposta la variabile GPUSTACK_API_KEY")
        return

    try:
        client = OpenAI(
            base_url=CLUSTER_BASE_URL,
            api_key=api_key,
        )

        print("Invio richiesta con problema logico...\n")

        # Problema che richiede ragionamento
        response = client.chat.completions.create(
            model=TARGET_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a logical reasoning assistant. Think step by step."
                },
                {
                    "role": "user", 
                    "content": """Solve this logic puzzle step by step:
                    
Alice, Bob, and Charlie are standing in a line. 
- Alice is not at the front
- Charlie is not at the back
- Bob is between Alice and Charlie

What is the order from front to back?"""
                }
            ],
            temperature=0.7,
            max_tokens=500
        )

        choice = response.choices[0]
        reasoning = getattr(choice.message, 'reasoning_content', None)

        print("=" * 60)
        
        if reasoning:
            print("✓ REASONING TROVATO!\n")
            print("--- RAGIONAMENTO INTERNO DEL MODELLO ---")
            print(reasoning)
            print("\n" + "=" * 60 + "\n")
        else:
            print("✗ Nessun reasoning_content trovato per questo modello\n")
            print("=" * 60 + "\n")
            
        print("--- RISPOSTA FINALE ---")
        print(choice.message.content)
        print("\n" + "=" * 60)

        # Info aggiuntive
        print(f"\nToken usati: {response.usage.total_tokens}")
        print(f"Modello: {TARGET_MODEL}")

    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        print("\nVerifica:")
        print("1. Sei connesso alla VPN UniBS?")
        print("2. La variabile GPUSTACK_API_KEY è impostata?")
        print("3. Il nome del modello è corretto?")

if __name__ == "__main__":
    test_reasoning()
