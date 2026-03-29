# NIST CSF 2.0 Interview Agent

This repository contains an AI-driven interview engine designed to perform a comprehensive cybersecurity audit based on the **NIST Cybersecurity Framework (CSF) 2.0**. The agent interacts with users to gather information about their organization's current and target cybersecurity posture, automatically mapping the responses to the official NIST 2.0 subcategories.

## 🗂️ Code Structure

- **`interview_engine.py`**: The core application module. It manages the flow of the interview, generates contextual questions based on the NIST framework using an LLM, and extracts structured data (Current and Target states) from the user's free-text answers.
- **`profile_manager.py`**: Handles the underlying state machine and data persistence. It tracks the interview progress using a Pandas DataFrame and saves the state to CSV files, mapping everything exactly to the 17 official NIST Excel columns.
- **`data/`**: Directory containing the application datasets:
  - `data/cleaned/csf_2_0_catalog.csv`: The base catalog containing all 185 NIST CSF 2.0 subcategories and mapping data.
- **`.env`**: (User-created) Configuration file that securely stores necessary API keys (e.g., OpenRouter, GPUStack, GitHub).
- **`requirements.txt`**: List of Python dependencies required to run the project.

## ⚙️ Prerequisites & Setup

1. **Python**: Ensure you have Python 3.8+ installed.
2. **Virtual Environment** (Recommended): 
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Environment Variables**:
   Create a `.env` file in the root directory (ensure it is saved with UTF-8 encoding) and add your API keys:
   ```env
   OPENROUTER_API_KEY=your_key_here
   GPUSTACK_API_KEY=your_key_here
   GITHUB_TOKEN=your_token_here
   ```

## 🚀 How to Run the Application

To start the interview process, you typically run the main execution script (e.g., if you have configured the execution within `interview_engine.py` or a dedicated test script):

```bash
python interview_engine.py
```


During the execution:
- The AI will evaluate progress and pick the next pending NIST subcategory.
- It will ask you a question regarding your **Current State**.
- After your answer, it will ask a follow-up about your **Target/Future State**.
- You can type `/quit` at any point to save your progress and safely exit the application. 

## 📂 Where to find the results

As the interview progresses, the system automatically saves your profile and interview traces.

- **Profile State**: The primary result file is a CSV that tracks your comprehensive profile. Look for the `Compilazioni/` directory created in the project root. The file is automatically named using a timestamp, such as `profile_YYYYMMDD_HHMMSS.csv`.
- **Output Columns**: The resulting CSV is fully formatted to match the official NIST template, spanning 24+ columns like `Included_in_Profile`, `Current_Priority`, `Target_Policies_Processes_Procedures`, etc.
- **Logs**: Along with the profile CSV, interview logs and LLM reasoning traces are saved in a timestamped folder inside `Compilazioni/` (e.g., `logs_YYYYMMDD_HHMMSS/interview_log.json`), keeping a durable record of the exact AI prompts, user answers, and JSON extractions.
