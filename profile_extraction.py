import sys
import json
from colorama import init, Fore, Style

# IMPORT CORRETTO: Usiamo i nomi esatti che hai definito nel tuo file
from dialogue_manager import NISTComplianceDialogueManager, ConversationPhase

# Inizializza colorama per far resettare i colori automaticamente alla fine di ogni print
init(autoreset=True)

def print_header():
    """Stampa un'intestazione elegante per il terminale."""
    print(Fore.CYAN + Style.BRIGHT + "="*65)
    print(Fore.CYAN + Style.BRIGHT + "      🛡️ NIST CSF 2.0 - Compliance Officer in a Box      ")
    print(Fore.CYAN + Style.BRIGHT + "="*65)
    print(Fore.YELLOW + "Digita 'esci' o premi Ctrl+C in qualsiasi momento per uscire.\n")

def save_profile_to_file(profile_data, filename="Tailored_Company_Profile.json"):
    """Salva il profilo aziendale estratto in un file JSON."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, indent=4)
        print(Fore.GREEN + f"\n[SUCCESS] Profilo salvato con successo in: {filename}")
    except Exception as e:
        print(Fore.RED + f"\n[ERROR] Impossibile salvare il profilo: {e}")

def main():
    print_header()
    
    # 1. Inizializzazione del motore di dialogo
    print(Fore.LIGHTBLACK_EX + "[Sistema] Inizializzazione dell'Agente LLM e caricamento dati NIST in corso...")
    try:
        # Usiamo il nome esatto della tua classe
        dm = NISTComplianceDialogueManager(verbose=False)
    except Exception as e:
        print(Fore.RED + f"Errore critico durante l'avvio: {e}")
        sys.exit(1)
        
    print(Fore.GREEN + "[Sistema] Agente pronto!\n")

    # 2. Ottenere la prima domanda dall'IA
    # Il tuo codice usa start_dialogue() per rompere il ghiaccio
    response = dm.start_dialogue()
    print(Fore.MAGENTA + Style.BRIGHT + "\n🤖 Compliance Officer:\n" + Style.NORMAL + response)

    # 3. Il Loop principale (L'intervista)
    try:
        while True:
            # Controllo immediato: se l'intervista è finita (es. per limitazioni di turni), esci
            if dm.current_phase == ConversationPhase.COMPLETION:
                break

            # Input dell'utente
            user_input = input(Fore.CYAN + "\n👤 Tu: " + Style.RESET_ALL)
            
            # Condizioni di uscita manuale
            if user_input.strip().lower() in ['esci', 'exit', 'quit']:
                print(Fore.YELLOW + "\nIntervista interrotta dall'utente. Arrivederci!")
                break
                
            if not user_input.strip():
                continue

            print(Fore.LIGHTBLACK_EX + "L'agente sta ragionando e analizzando il tuo profilo...")
            
            # Elaborazione tramite LLM
            response = dm.process_user_response(user_input)
            
            # Risposta dell'IA
            print(Fore.MAGENTA + Style.BRIGHT + "\n🤖 Compliance Officer:\n" + Style.NORMAL + response)
            
            # Controllo se l'intervista è terminata DOPO la risposta
            if dm.current_phase == ConversationPhase.COMPLETION:
                print(Fore.GREEN + Style.BRIGHT + "\n" + "="*65)
                print(Fore.GREEN + Style.BRIGHT + "                 ASSESSMENT COMPLETATO!                 ")
                print(Fore.GREEN + Style.BRIGHT + "="*65)
                
                # LA RIGA CORRETTA: Estraiamo il dizionario usando la variabile giusta e il metodo to_dict()
                final_profile = dm.company_profile.to_dict()
                
                print(Fore.YELLOW + "\nEcco il tuo Profilo Aziendale Strutturato:")
                print(json.dumps(final_profile, indent=4))
                
                # Salviamo il file JSON e i log come da tuo codice originale
                save_profile_to_file(final_profile)
                dm.export_conversation_log()
                dm.export_detailed_log()
                break

    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\n[Sistema] Chiusura forzata rilevata. Arrivederci!")
        sys.exit(0)

if __name__ == "__main__":
    main()