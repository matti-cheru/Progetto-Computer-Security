# NIST CSF 2.0 Compliance AI Assistant

This project implements a **local AI assistant** designed to help Small-to-Medium Businesses (SMBs) navigate the **NIST Cybersecurity Framework (CSF) 2.0**.

The system utilizes local Large Language Models (LLMs) like **Microsoft Phi-3 Mini** or **Llama 3** running on consumer hardware to ensure **data privacy** and sovereignty. No sensitive compliance data is sent to external cloud providers.

## 🖥️ System Requirements

* **OS:** Windows 10/11 or Linux.
* **GPU:** NVIDIA GPU with at least **8GB VRAM** (RTX 3060/4060 or higher recommended).
* **Python:** Version 3.10 or higher.
* **Disk Space:** At least 10GB free for model weights.

## ⚙️ Installation Guide

Follow these steps precisely to ensure GPU acceleration (CUDA) works correctly.

### 1. Clone the Repository
Open your terminal (PowerShell or CMD) and run:
```bash
git clone [https://github.com/matti-cheru/Progetto-Computer-Security.git](https://github.com/matti-cheru/Progetto-Computer-Security.git)
cd Progetto-Computer-Security
2. Set up a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

Bash

python -m venv .venv
# Activate on Windows:
.\.venv\Scripts\activate
# Activate on Linux/Mac:
# source .venv/bin/activate
3. Install PyTorch with CUDA Support (CRITICAL STEP)
Do not use the standard pip install torch. To enable GPU acceleration on Windows, you must install the specific CUDA 12.1 version from the official PyTorch repository.

Run this command exactly:

Bash

pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
4. Install Project Dependencies
Once PyTorch is set up, install the remaining libraries from the requirements file:

Bash

pip install -r requirements.txt
🚀 How to Run
We have provided a test script to verify that the model loads correctly on your GPU and can answer questions.

Run the Phi-3 Test Script
Bash

python test_phi3.py
What to expect:
First Run: The script will download the model weights (~7-8 GB) from Hugging Face. This may take a few minutes depending on your connection.

Device Check: Look for the log message: Dispositivo in uso: CUDA.

Output: The model will answer 3 pre-defined questions about itself and cybersecurity concepts.

🛠️ Troubleshooting
Issue: "Flash Attention package not found"

Cause: The flash-attention library is difficult to compile on Windows.

Solution: You can safely ignore this warning. The script is configured to automatically fallback to the standard attention implementation (eager), which works perfectly on Windows.

Issue: "CUDA out of memory"

Cause: Your GPU VRAM is full.

Solution: Close other GPU-intensive applications (Video games, Browser with many tabs, Video Editors) before running the script. The Phi-3 model requires approx. 6GB of dedicated VRAM.

Issue: ImportError or ModuleNotFoundError

Solution: Ensure your virtual environment is activated (.venv) and that you ran both installation steps (Step 3 and Step 4).