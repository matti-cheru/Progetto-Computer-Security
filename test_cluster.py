import os
import sys
from openai import OpenAI

#Configuration
# SECURITY NOTE: The documentation recommends using environment variables.
# Ideally, run: export GPUSTACK_API_KEY="your-real-key" in your terminal.
# If you cannot set env vars, replace the placeholder below.
DEFAULT_API_KEY_PLACEHOLDER = "default_key_placeholder"

# Connection Settings [cite: 222, 234]
CLUSTER_BASE_URL = "https://gpustack.ing.unibs.it/v1"
# Available models: "qwen3", "phi4-mini", "phi4", "llama3.2", "gpt-oss", "granite3.3", "gemma3" 
TARGET_MODEL = "gpt-oss" 

def get_api_key():
    """
    Retrieves the API key securely from environment variables,
    falling back to the script variable if not found.
    """
    # Check system environment variable first (Best Practice) [cite: 229]
    key = os.environ.get("GPUSTACK_API_KEY")
    
    if key:
        print("-> Using API Key from Environment Variable.")
        return key
    
    # Fallback to the variable defined above
    if DEFAULT_API_KEY_PLACEHOLDER != "default_key_placeholder":
        print("-> Using API Key hardcoded in the script.")
        return DEFAULT_API_KEY_PLACEHOLDER
        
    return None

def test_llm_connection():
    """
    Attempts to connect to the UniBS LLM Cluster and generate a response.
    """
    print(f"--- Starting Connection Test to {CLUSTER_BASE_URL} ---")
    
    api_key = get_api_key()
    
    if not api_key:
        print("ERROR: No valid API Key found.")
        print("Please set the GPUSTACK_API_KEY environment variable or update the script.")
        return

    try:
        # Initialize the OpenAI client pointing to the custom endpoint [cite: 233]
        client = OpenAI(
            base_url=CLUSTER_BASE_URL,
            api_key=api_key,
        )

        print(f"-> Sending request to model: '{TARGET_MODEL}'...")

        # Create the chat completion request [cite: 252]
        response = client.chat.completions.create(
            model=TARGET_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for a university project."},
                {"role": "user", "content": "Hello! If you can read this, please reply with 'Connection Successful' and a curiosity about AI."}
            ],
            temperature=0.7,
            max_tokens=150
        )

        # Extract the response
        choice = response.choices[0]
        message_content = choice.message.content
        
        # Check for internal reasoning/thinking (some models provide this) [cite: 266]
        # Note: The library object might have 'reasoning_content' directly or in extra_fields depending on version
        reasoning_content = getattr(choice.message, 'reasoning_content', None)

        print("\n--- TEST RESULT: SUCCESS ---")
        
        if reasoning_content:
            print(f"\n[Internal Reasoning]:\n{reasoning_content}") # [cite: 273]
            
        print(f"\n[Model Response]:\n{message_content}")

    except Exception as e:
        print(f"\n--- TEST RESULT: FAILED ---")
        print(f"An error occurred: {e}")
        print("Check your VPN connection, API Key, and if the model name is correct.") # [cite: 221]

if __name__ == "__main__":
    test_llm_connection()