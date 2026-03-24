Using the LLM Cluster at UniBS This document explains how to access and use the experimental LLM cluster running various open-source models through an OpenAI-compatible API. The service is experimental and there is no guarantee of availability, latency, or maximum throughput. Use it for coursework and experiments, not for production systems.1. Network access The cluster endpoint is only reachable from machines on the university network or from outside the university via the university VPN.If you cannot reach https://gpustack.ing.unibs.it/v1, first check that you are either on campus or connected through the official VPN.Info is available at https://www.unibs.it/it/opportunita-e-servizi/scopri-opportunita-e-servizi/servizi-digitali/reti/servizio-vpn-openvpn.2. Authentication and API key Access is controlled by an API key, referred to here as YOUR_GPUSTACK_API_KEY.Students will be given an API key by the instructor or system administrator.Do not share your key and do not commit it to Git repositories.A safe pattern is to keep the key in an environment variable, for example in a shell start-up file:Bashexport GPUSTACK_API_KEY="your-real-key-here"
Then load it in Python:Pythonimport os
from openai import OpenAI

client = OpenAI(
    base_url="https://gpustack.ing.unibs.it/v1",
    api_key=os.environ["GPUSTACK_API_KEY"],
)
3. Installing the Python client You need Python (3.9 or later recommended) and the official openai Python package:Bashpip install --upgrade openai
4. Basic usage pattern (chat completions) All LLMs on the cluster are exposed via the OpenAI-compatible chat completions API. The basic pattern is always the same; only the model name changes.Example with qwen3:Pythonfrom openai import OpenAI
import os

client = OpenAI(
    base_url="https://gpustack.ing.unibs.it/v1",
    api_key=os.environ["GPUSTACK_API_KEY"],
)

response = client.chat.completions.create(
    model="qwen3",
    messages=[
        {"role": "system", "content": "You are a helpful assistant for a computer science student."},
        {"role": "user", "content": "Explain overfitting in one paragraph."},
    ],
    temperature=0.6,
    top_p=0.95,
    max_tokens=1024,
    frequency_penalty=0,
    presence_penalty=0,
)
print(response.choices[0].message.content)
Some models exposed by the cluster return an additional field containing their internal reasoning or "thinking" process. When available, this information can be useful for debugging prompts, understanding unexpected outputs, or analysing model behaviour. After receiving a response, you can inspect both the reasoning and the final answer separately:Pythonchoice = response.choices[0]

# Print the reasoning / thinking process (if provided by the model)
print("--- THINKING ---")
print(choice.message.reasoning_content)

# Print the final answer
print("--- ANSWER ---")
print(choice.message.content)
The reasoning_content field may be None or missing for some models or configurations. In that case, only choice.message.content will be populated.5. Available models The cluster currently exposes the following models.5.1 LLMs Model nameSource (GGUF)qwen3Hugging Face/unsloth/Qwen3-4BGGUF-phi4-miniHugging Face/unsloth/Phi-4-mini-instruct-GGUFphi4Hugging Face/bartowski/phi-4-GGUFllama3.2Hugging Face/bartowski/Llama-3.2-3B-Instruct-GGUFgpt-ossHugging Face/unsloth/gpt-oss-20b-GGUFgranite3.3Hugging Face/ibm-granite/granite-3.3-2b-instruct-GGUFgemma3Hugging Face/bartowski/google_gemma-3-1b-it-GGUF(Source) 5.2 Embeddings Model nameSource (GGUF)qwen3-embeddingHugging Face/Qwen/Qwen3-Embedding-4B-GGUFnomic-embed-text-v1.5Hugging Face/nomic-ai/nomic-embed-text-v1.5-GGUF(Source) 6. Model-specific chat completion examples In all examples it is assumed you have already constructed client as shown above.6.1 Qwen3 Pythonresponse = client.chat.completions.create(
    model="qwen3",
    messages=[
        {"role": "system", "content": "You are a helpful assistant for data science coursework."},
        {"role": "user", "content": "Summarise the bias-variance tradeoff in two short paragraphs."},
    ],
    temperature=0.6,
    top_p=0.95,
    max_tokens=1024,
    frequency_penalty=0,
    presence_penalty=0,
)
print(response.choices[0].message.content)
6.2 Phi4-mini Pythonresponse = client.chat.completions.create(
    model="phi4-mini",
    messages=[
        {"role": "system", "content": "You are a concise tutor for introductory AI courses."},
        {"role": "user", "content": "Explain what a confusion matrix is, with a small example."},
    ],
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
    frequency_penalty=0,
    presence_penalty=0,
)
print(response.choices[0].message.content)
6.3 Phi4 Pythonresponse = client.chat.completions.create(
    model="phi4",
    messages=[
        {"role": "system", "content": "You are an advanced assistant for coding and research questions."},
        {"role": "user", "content": "Write a short explanation of gradient descent for lecture slides."},
    ],
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
    frequency_penalty=0,
    presence_penalty=0,
)
print(response.choices[0].message.content)
6.4 Llama3.2 Pythonresponse = client.chat.completions.create(
    model="llama3.2",
    messages=[
        {"role": "system", "content": "You help students understand algorithms."},
        {"role": "user", "content": "Describe Dijkstra's algorithm informally and mention its complexity."},
    ],
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
    frequency_penalty=0,
    presence_penalty=0,
)
print(response.choices[0].message.content)
6.5 GPT-OSS-20B Pythonresponse = client.chat.completions.create(
    model="gpt-oss",
    messages=[
        {"role": "system", "content": "You are a careful assistant for code and theory explanations."},
        {"role": "user", "content": "Given a Python function, suggest tests for edge cases."},
    ],
    temperature=0.8,
    top_p=0.95,
    max_tokens=4096,
    frequency_penalty=0,
    presence_penalty=0,
)
print(response.choices[0].message.content)
6.6 Granite3.3 Pythonresponse = client.chat.completions.create(
    model="granite3.3",
    messages=[
        {"role": "system", "content": "You provide short, clear answers."},
        {"role": "user", "content": "List key differences between supervised and unsupervised learning."},
    ],
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
    frequency_penalty=0,
    presence_penalty=0,
)
print(response.choices[0].message.content)
6.7 Gemma3 Pythonresponse = client.chat.completions.create(
    model="gemma3",
    messages=[
        {"role": "system", "content": "You answer briefly and in simple language."},
        {"role": "user", "content": "Explain what a neural network layer is to a beginner."},
    ],
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
    frequency_penalty=0,
    presence_penalty=0,
)
print(response.choices[0].message.content)
7. Embeddings (text to vectors) Embedding models convert text into vectors (lists of floating-point numbers). These vectors can be used for semantic search, clustering, recommendation, and retrieval-augmented generation (RAG).7.1 Basic embeddings example (nomic) Pythonfrom openai import OpenAI

client = OpenAI(
    base_url="https://gpustack.ing.unibs.it/v1",
    api_key="YOUR_GPUSTACK_API_KEY"
)

response = client.embeddings.create(
    model="nomic-embed-text-v1.5",
    input=[
        "What are the best cafes nearby?",
        "Can you recommend a quiet coffee shop?",
        "I'm planning a trip to Japan next month.",
        "The capital of France is Paris.",
        "I'm looking for a place to study."
    ]
)
print(response.data[0].embedding)
The returned response.data is a list with one element per input string. Each element contains an embedding field, which is the vector for that input.7.2 Embeddings example (Qwen3 embedding model) Pythonfrom openai import OpenAI
import os

client = OpenAI(
    base_url="https://gpustack.ing.unibs.it/v1",
    api_key=os.environ["GPUSTACK_API_KEY"],
)

response = client.embeddings.create(
    model="qwen3-embedding",
    input=[
        "Neural networks are function approximators.",
        "Support vector machines separate classes with margins.",
    ]
)
print(len(response.data[0].embedding))
8. Debugging with raw API calls (curl) If you need to debug issues at a lower level, you can bypass the Python client and interact directly with the HTTP API using curl. This is useful to inspect the raw request and response or to verify that network access and authentication work correctly.Example chat completion request with qwen3:Bashcurl https://gpustack.ing.unibs.it/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${YOUR_GPUSTACK_API_KEY}" \
  -d '{
    "model": "qwen3",
    "messages": [],
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 1024,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "seed": null,
    "stop": null
  }'
The JSON payload mirrors the parameters used in the Python examples. Any error returned here is a direct response from the API and can help diagnose authentication, network, or parameter issues.9. Good practices and limitations Because the cluster is experimental, you should expect occasional failures, timeouts, or slow responses.In your code, handle exceptions and retry a limited number of times.Do not send sensitive personal data or confidential information to the cluster. Treat it as a teaching and research resource.If you encounter persistent problems, record the time, model, and error message, and report them to the instructor or system administrator