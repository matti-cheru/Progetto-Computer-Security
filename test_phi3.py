import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def main():
    # Configurazione del modello
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    
    print(f"--- Caricamento del modello {model_id} ---")
    print("Nota: Il primo avvio richiederà tempo per scaricare circa 7-8 GB di dati.")

    # Controllo se CUDA (GPU) è disponibile
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo in uso: {device.upper()}")
    
    if device == "cpu":
        print("ATTENZIONE: Stai usando la CPU. Sarà molto lento. Controlla l'installazione di PyTorch.")

# Caricamento del Tokenizer e del Modello
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,              
            torch_dtype=torch.float16,      
            # trust_remote_code=True,       <-- RIMOSSO: Usa l'implementazione interna stabile
            attn_implementation="eager"     # Manteniamo questo per sicurezza
        )
    except Exception as e:
        print(f"Errore durante il caricamento: {e}")
        return

    # Creazione della pipeline di generazione
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # Lista di domande per il test
    domande = [
        "Ciao! Chi sei e cosa sai fare?",
        "Spiegami in modo semplice cos'è un buco nero.",
        "Scrivi una breve funzione in Python per calcolare la sequenza di Fibonacci."
    ]

    # Configurazione parametri di generazione
    gen_kwargs = {
        "max_new_tokens": 500,      # Lunghezza massima della risposta
        "do_sample": True,          # Attiva la creatività
        "temperature": 0.7,         # Bilanciamento tra creatività e coerenza
        "top_p": 0.9,
    }

    print("\n--- Inizio Test ---\n")

    for domanda in domande:
        print(f"Utente: {domanda}")
        
        # Formattazione del prompt stile chat (importante per i modelli Instruct)
        messages = [{"role": "user", "content": domanda}]
        
        # Generazione
        output = pipe(messages, **gen_kwargs)
        
        # Estrazione della risposta
        risposta = output[0]['generated_text'][-1]['content']
        
        print(f"Phi-3: {risposta}")
        print("-" * 50)

if __name__ == "__main__":
    main()