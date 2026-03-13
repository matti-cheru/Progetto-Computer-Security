"""
TEST: Accedere ai Reasoning Tokens di gpt-oss

Obiettivo: 
- Verificare se gpt-oss espone i reasoning tokens con la libreria openai nativa
- Testare configurazioni diverse per ottenere l'output invece del solo reasoning

Confronto:
1. Libreria openai nativa (from openai import OpenAI)
2. LangChain ChatOpenAI (from langchain_openai import ChatOpenAI)
"""
import os
import json
from openai import OpenAI
from langchain_openai import ChatOpenAI

# Configurazione
CLUSTER_BASE_URL = "https://gpustack.ing.unibs.it/v1"
MODEL = "gpt-oss"
API_KEY = os.environ.get("GPUSTACK_API_KEY", "gpustack_8a00037ab4220858_6479d07686028e2f970357b1a81200e4")


def print_separator(title):
    """Helper per stampare separatori"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def test_openai_native_simple():
    """
    Test 1: Libreria OpenAI nativa con domanda semplice
    """
    print_separator("TEST 1: OpenAI Nativa - Domanda Semplice")
    
    client = OpenAI(base_url=CLUSTER_BASE_URL, api_key=API_KEY)
    
    prompt = "Calcola 5 + 7 e rispondi solo con il numero."
    print(f"📤 Prompt: {prompt}")
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    
    choice = response.choices[0]
    message = choice.message
    
    print("\n📊 RISULTATO:")
    print(f"   content: '{message.content}'")
    print(f"   reasoning_content: {getattr(message, 'reasoning_content', 'N/A')}")
    
    print("\n📈 METADATA:")
    print(f"   finish_reason: {choice.finish_reason}")
    
    if hasattr(response, 'usage'):
        usage = response.usage
        print(f"   input_tokens: {usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else usage.input_tokens if hasattr(usage, 'input_tokens') else 'N/A'}")
        print(f"   output_tokens: {usage.completion_tokens if hasattr(usage, 'completion_tokens') else usage.output_tokens if hasattr(usage, 'output_tokens') else 'N/A'}")
        
        # Controlla se ci sono dettagli sui reasoning tokens
        if hasattr(usage, 'completion_tokens_details'):
            print(f"   completion_tokens_details: {usage.completion_tokens_details}")
    
    # Dump completo dell'oggetto risposta
    print("\n🔍 DUMP COMPLETO RISPOSTA:")
    print(f"   {response.model_dump() if hasattr(response, 'model_dump') else response}")
    
    return message.content


def test_openai_native_with_reasoning_task():
    """
    Test 2: Libreria OpenAI nativa con task che richiede ragionamento
    """
    print_separator("TEST 2: OpenAI Nativa - Task con Ragionamento")
    
    client = OpenAI(base_url=CLUSTER_BASE_URL, api_key=API_KEY)
    
    prompt = """Risolvi questo problema passo-passo:
    
Ho 10 mele. Ne do 3 a Marco e 2 a Luca.
Poi ne compro altre 5.
Quante mele ho adesso?

Rispondi seguendo questo formato:
1. Situazione iniziale: ...
2. Dopo aver dato le mele: ...
3. Dopo l'acquisto: ...
4. Totale finale: X mele"""
    
    print(f"📤 Prompt: {prompt[:100]}...")
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Think step by step."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=500
    )
    
    choice = response.choices[0]
    message = choice.message
    
    print("\n📊 RISULTATO:")
    print(f"   content length: {len(message.content)} caratteri")
    print(f"   content preview: '{message.content[:200]}...'")
    
    # Cerca reasoning_content
    reasoning = getattr(message, 'reasoning_content', None)
    if reasoning:
        print(f"\n✅ REASONING TROVATO!")
        print(f"   reasoning_content length: {len(reasoning)} caratteri")
        print(f"   reasoning preview: '{reasoning[:200]}...'")
    else:
        print(f"\n❌ reasoning_content: Non trovato")
    
    # Metadata
    print("\n📈 METADATA:")
    print(f"   finish_reason: {choice.finish_reason}")
    
    if hasattr(response, 'usage'):
        usage = response.usage
        print(f"   Usage: {usage}")
    
    return message.content


def test_openai_native_react_format():
    """
    Test 3: Simuliamo un task ReAct per vedere cosa genera
    """
    print_separator("TEST 3: OpenAI Nativa - Formato ReAct Esplicito")
    
    client = OpenAI(base_url=CLUSTER_BASE_URL, api_key=API_KEY)
    
    prompt = """You have access to a Python dataframe called 'df' with these columns: ['Name', 'Age', 'City'].

Question: How many rows are in the dataframe?

Respond using this format EXACTLY:

Thought: [your reasoning about what to do]
Action: python_repl_ast
Action Input: [python code to execute]

Now respond:"""
    
    print(f"📤 Prompt (formato ReAct):\n{prompt}\n")
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a Python code generator for data analysis. Follow the ReAct format strictly."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=300
    )
    
    choice = response.choices[0]
    message = choice.message
    
    print("\n📊 RISULTATO:")
    print(f"   content: '{message.content}'")
    
    reasoning = getattr(message, 'reasoning_content', None)
    if reasoning:
        print(f"\n✅ reasoning_content trovato:")
        print(f"   {reasoning}")
    else:
        print(f"\n❌ reasoning_content: Non disponibile")
    
    # Metadata dettagliata
    print("\n📈 METADATA:")
    if hasattr(response, 'usage'):
        usage = response.usage
        usage_dict = usage.model_dump() if hasattr(usage, 'model_dump') else usage
        print(f"   {json.dumps(usage_dict, indent=2)}")
    
    return message.content


def test_langchain_wrapper():
    """
    Test 4: LangChain ChatOpenAI per confronto
    """
    print_separator("TEST 4: LangChain ChatOpenAI Wrapper")
    
    llm = ChatOpenAI(
        base_url=CLUSTER_BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        temperature=0.0
    )
    
    prompt = "Calcola 8 + 4 e rispondi solo con il numero."
    print(f"📤 Prompt: {prompt}")
    
    response = llm.invoke(prompt)
    
    print("\n📊 RISULTATO:")
    print(f"   Type: {type(response)}")
    print(f"   content: '{response.content}'")
    
    # Controlla se ci sono attributi extra
    print("\n🔍 Attributi disponibili:")
    attrs = [attr for attr in dir(response) if not attr.startswith('_')]
    for attr in attrs[:20]:  # Primi 20
        try:
            value = getattr(response, attr)
            if not callable(value):
                print(f"   {attr}: {value}")
        except:
            pass
    
    # Response metadata
    if hasattr(response, 'response_metadata'):
        print(f"\n📈 response_metadata:")
        print(f"   {json.dumps(response.response_metadata, indent=2)}")
    
    return response.content


def test_with_logprobs():
    """
    Test 5: Con logprobs attivati (alcuni modelli espongono più info)
    """
    print_separator("TEST 5: OpenAI Nativa con logprobs=True")
    
    client = OpenAI(base_url=CLUSTER_BASE_URL, api_key=API_KEY)
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "Count from 1 to 5."}
            ],
            temperature=0.0,
            logprobs=True,
            top_logprobs=3
        )
        
        choice = response.choices[0]
        
        print("\n📊 RISULTATO:")
        print(f"   content: '{choice.message.content}'")
        
        if hasattr(choice, 'logprobs') and choice.logprobs:
            print(f"\n✅ logprobs disponibili!")
            print(f"   Type: {type(choice.logprobs)}")
            if hasattr(choice.logprobs, 'content'):
                print(f"   Tokens: {len(choice.logprobs.content) if choice.logprobs.content else 0}")
        else:
            print(f"\n❌ logprobs non disponibili")
        
        # Usage
        if hasattr(response, 'usage'):
            print(f"\n📈 Usage: {response.usage}")
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        print("   (Il modello potrebbe non supportare logprobs)")


def main():
    """Main test function"""
    
    print("\n" + "#"*80)
    print("#" + " "*25 + "TEST ACCESSO REASONING TOKENS" + " "*25 + "#")
    print("#"*80)
    
    print("\n📝 Questo script testa se possiamo accedere ai reasoning tokens")
    print("   generati internamente da gpt-oss usando diverse configurazioni.\n")
    
    input("▶️  Premi ENTER per iniziare i test...")
    
    try:
        # Test 1
        result1 = test_openai_native_simple()
        input("\n▶️  Premi ENTER per test successivo...")
        
        # Test 2
        result2 = test_openai_native_with_reasoning_task()
        input("\n▶️  Premi ENTER per test successivo...")
        
        # Test 3
        result3 = test_openai_native_react_format()
        input("\n▶️  Premi ENTER per test successivo...")
        
        # Test 4
        result4 = test_langchain_wrapper()
        input("\n▶️  Premi ENTER per test successivo...")
        
        # Test 5
        test_with_logprobs()
        
        # Summary
        print("\n\n" + "#"*80)
        print("#" + " "*32 + "CONCLUSIONI" + " "*35 + "#")
        print("#"*80)
        
        print("\n📊 RISULTATI SUMMARY:")
        print(f"   Test 1 (OpenAI semplice): content = '{result1}'")
        print(f"   Test 2 (OpenAI reasoning): content length = {len(result2)} caratteri")
        print(f"   Test 3 (OpenAI ReAct): content = '{result3}'")
        print(f"   Test 4 (LangChain): content = '{result4}'")
        
        print("\n💡 ANALISI:")
        
        if all(not r for r in [result1, result2, result3, result4]):
            print("   ❌ Tutti i test restituiscono content vuoto")
            print("   📌 gpt-oss genera solo reasoning tokens interni, non testo normale")
            print("   📌 Questi reasoning tokens NON sono accessibili tramite API")
            print("\n   🔧 SOLUZIONI POSSIBILI:")
            print("      1. Usa metodi Pandas diretti (senza LLM)")
            print("      2. Prova un modello diverso (es. qwen3, phi4)")
            print("      3. Contatta amministratori cluster per configurazione modello")
        elif result1 or result2:
            print("   ✅ OpenAI nativa riesce a ottenere contenuto!")
            print("   📌 Problema potrebbe essere in LangChain wrapper")
        else:
            print("   ⚠️  Risultati misti - analisi manuale necessaria")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrotto dall'utente")
    except Exception as e:
        print(f"\n\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
