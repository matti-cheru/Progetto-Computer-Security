"""
INTERVIEW ENGINE - Phase 1+2: Core Interview Loop + Structured Extraction

Orchestrates the NIST CSF 2.0 compliance interview by:
- Pulling the next PENDING subcategory from ProfileManager
- Building a professional, contextualized question via LLM
- Extracting structured data from the user's free-text answer
- Saving progress incrementally to client_profile_state.csv

Supports verbose and non-verbose modes with full JSON logging.
"""
import os
import json
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

import pandas as pd
from openai import OpenAI

from profile_manager import ProfileManager


class InterviewEngine:
    """
    Core engine for the NIST CSF 2.0 compliance interview.

    Connects ProfileManager (state) with the LLM to run a
    conversational loop that progressively fills the organizational profile.
    """

    # ── Subcategory scope limiter (for testing) ──────────────────────
    # Set to None to process ALL subcategories.
    SCOPE_SUBCATEGORIES: Optional[Set[str]] = {
        "ID.AM-01", "ID.AM-02", "ID.AM-03", "ID.AM-04",
        "ID.AM-05", "ID.AM-06", "ID.AM-07", "ID.AM-08",
    }

    # ── Profile columns that the LLM must fill ──────────────────────
    CURRENT_PROFILE_COLUMNS = [
        "Included_in_Profile",
        "Rationale",
        "Current_Priority",
        "Current_Status",
        "Current_Policies_Processes_Procedures",
        "Current_Internal_Practices",
        "Current_Roles_and_Responsibilities",
        "Current_Selected_Informative_References",
        "Current_Artifacts_and_Evidence",
    ]

    TARGET_PROFILE_COLUMNS = [
        "Target_Priority",
        "Target_CSF_Tier",
        "Target_Policies_Processes_Procedures",
        "Target_Internal_Practices",
        "Target_Roles_and_Responsibilities",
        "Target_Selected_Informative_References",
        "Notes",
        "Considerations",
    ]

    ALL_PROFILE_COLUMNS = CURRENT_PROFILE_COLUMNS + TARGET_PROFILE_COLUMNS

    def __init__(
        self,
        base_url: str = "https://gpustack.ing.unibs.it/v1",
        model_name: str = "gpt-oss",
        api_key: Optional[str] = None,
        verbose: bool = True,
        log_dir: Optional[str] = None,
    ):
        """
        Args:
            base_url:   LLM endpoint URL
            model_name: Model identifier
            api_key:    API key (falls back to env GPUSTACK_API_KEY)
            verbose:    If True, print full internal traces to console
            log_dir:    Directory for JSON logs (auto-created per run)
        """
        if api_key is None:
            api_key = os.environ.get(
                "GPUSTACK_API_KEY",
                "gpustack_8a00037ab4220858_6479d07686028e2f970357b1a81200e4",
            )

        self.verbose = verbose
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)

        # ProfileManager handles state persistence
        self.manager = ProfileManager()

        # Run-level logging directory
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = log_dir or os.path.join(os.getcwd(), f"interview_run_{ts}")
        os.makedirs(self.log_dir, exist_ok=True)

        # Accumulated interview log (saved at each turn)
        self._turn_logs: List[Dict[str, Any]] = []

        self._print_always(
            "\n" + "=" * 70
            + "\n🛡️  NIST CSF 2.0 — Interview Engine initialized"
            + f"\n   Model: {model_name}"
            + f"\n   Log dir: {self.log_dir}"
            + f"\n   Scope: {self.SCOPE_SUBCATEGORIES or 'ALL subcategories'}"
            + "\n" + "=" * 70
        )

    # ═══════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ═══════════════════════════════════════════════════════════════

    def start(self):
        """
        Main entry point.  Runs the interview loop until all in-scope
        subcategories are DONE, the user types /quit, or there are no
        more PENDING items.
        """
        self._print_always(
            "\n╔════════════════════════════════════════════════════════════╗"
            "\n║  NIST CSF 2.0 Organizational Profile — Interview Start   ║"
            "\n╠════════════════════════════════════════════════════════════╣"
            "\n║  Commands:                                                ║"
            "\n║   /progress  — Show completion progress                  ║"
            "\n║   /skip      — Skip the current subcategory              ║"
            "\n║   /quit      — Save and exit (resume later)              ║"
            "\n╚════════════════════════════════════════════════════════════╝"
        )

        self._show_progress()

        while True:
            row = self._get_next_in_scope()
            if row is None:
                self._print_always(
                    "\n✅ All in-scope subcategories have been completed!"
                )
                self._show_progress()
                break

            subcategory_id = row["Subcategory_ID"]

            # Mark row as IN_PROGRESS
            self.manager.update_row(
                subcategory_id, {"Completion_Status": "IN_PROGRESS"}
            )

            # Build and show question
            question, question_trace = self._build_question(row)
            self._print_always(f"\n{'─' * 60}")
            self._print_always(f"📋 Subcategory: {subcategory_id}")
            self._print_always(f"{'─' * 60}")
            self._print_always(f"\n{question}\n")

            # Get user input
            user_input = input("Your answer ▶ ").strip()

            # Handle commands
            if user_input.lower() == "/quit":
                # Revert to PENDING since we haven't processed it
                self.manager.update_row(
                    subcategory_id, {"Completion_Status": "PENDING"}
                )
                self._print_always("\n💾 Progress saved. You can resume later.")
                self._show_progress()
                break

            if user_input.lower() == "/skip":
                self.manager.update_row(
                    subcategory_id, {
                        "Completion_Status": "DONE",
                        "Included_in_Profile": "Skipped",
                        "Notes": "User skipped this subcategory.",
                    }
                )
                self._print_always(f"⏭️  Skipped {subcategory_id}")
                continue

            if user_input.lower() == "/progress":
                self._show_progress()
                # Revert to PENDING so it gets picked up again
                self.manager.update_row(
                    subcategory_id, {"Completion_Status": "PENDING"}
                )
                continue

            if not user_input:
                self._print_always("⚠️  Empty answer. Please provide a response or use /skip.")
                self.manager.update_row(
                    subcategory_id, {"Completion_Status": "PENDING"}
                )
                continue

            # Extract structured response
            extracted, extraction_trace = self._extract_response(row, user_input)

            # Show extracted data to user for transparency
            self._print_always(f"\n📊 Extracted profile data for {subcategory_id}:")
            for col, val in extracted.items():
                if col != "Completion_Status" and val:
                    self._print_always(f"   • {col}: {val}")

            # Save to profile
            extracted["Completion_Status"] = "DONE"
            self.manager.update_row(subcategory_id, extracted)
            self._print_always(f"\n✅ {subcategory_id} saved successfully.")

            # Log this turn
            turn_log = {
                "turn": len(self._turn_logs) + 1,
                "timestamp": datetime.now().isoformat(),
                "subcategory_id": subcategory_id,
                "row_context": {
                    "Function": row.get("Function", ""),
                    "Category": row.get("Category", ""),
                    "Subcategory_Description": row.get("Subcategory_Description", ""),
                    "Implementation_Examples": row.get("Implementation_Examples", ""),
                },
                "question_generation": question_trace,
                "user_answer": user_input,
                "extraction": extraction_trace,
                "extracted_data": extracted,
            }
            self._turn_logs.append(turn_log)
            self._save_run_log()

    # ═══════════════════════════════════════════════════════════════
    #  INTERNAL — Scope filtering
    # ═══════════════════════════════════════════════════════════════

    def _get_next_in_scope(self) -> Optional[Dict]:
        """
        Returns the next PENDING row that falls within SCOPE_SUBCATEGORIES.
        If SCOPE_SUBCATEGORIES is None, returns any next PENDING row.
        """
        if self.SCOPE_SUBCATEGORIES is None:
            return self.manager.get_next_pending()

        pending = self.manager.df[
            self.manager.df["Completion_Status"] == "PENDING"
        ]
        in_scope = pending[
            pending["Subcategory_ID"].isin(self.SCOPE_SUBCATEGORIES)
        ]
        if in_scope.empty:
            return None
        return in_scope.iloc[0].to_dict()

    # ═══════════════════════════════════════════════════════════════
    #  INTERNAL — Question building
    # ═══════════════════════════════════════════════════════════════

    def _build_question(self, row: Dict) -> tuple:
        """
        Uses the LLM to formulate a clear, professional interview question
        from the subcategory's catalog data.

        Returns:
            (question_text, trace_dict)
        """
        subcategory_id = row.get("Subcategory_ID", "")
        category = row.get("Category", "")
        description = row.get("Subcategory_Description", "")
        examples = row.get("Implementation_Examples", "")

        prompt = f"""You are a professional cybersecurity auditor conducting a NIST CSF 2.0 compliance interview.

Your task: formulate ONE clear, conversational question to ask the interviewee about the following NIST subcategory.

SUBCATEGORY DETAILS:
- ID: {subcategory_id}
- Category: {category}
- Description: {description}
- Implementation Examples: {examples}

INSTRUCTIONS:
1. Ask about their CURRENT state regarding this subcategory
2. Be specific but not overwhelming — ask one focused question
3. Briefly explain what this subcategory is about so the interviewee understands the context
4. Encourage them to describe: what they currently do, any policies/procedures, who is responsible, and any evidence/artifacts they have
5. Keep the tone professional but approachable
6. Write ONLY the question, no preamble or extra text

Question:"""

        messages = [
            {
                "role": "system",
                "content": "You are a cybersecurity compliance auditor. Generate clear, professional interview questions.",
            },
            {"role": "user", "content": prompt},
        ]

        trace = {
            "prompt": prompt,
            "messages": messages,
            "model": self.model_name,
        }

        self._print_verbose("\n🤖 [Question Generation] Sending prompt to LLM...")
        self._print_verbose(f"   Prompt length: {len(prompt)} chars")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=700,
            )

            content = response.choices[0].message.content or ""
            reasoning = getattr(response.choices[0].message, "reasoning_content", None)

            trace["response"] = {
                "content": content,
                "reasoning_content": reasoning,
                "tokens": {
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                },
            }

            self._print_verbose(f"   ✅ Response: {len(content)} chars")
            self._print_verbose(f"   Tokens — in: {response.usage.prompt_tokens}, out: {response.usage.completion_tokens}")
            if reasoning:
                self._print_verbose(f"   🧠 Reasoning: {reasoning[:200]}...")

            return content.strip(), trace

        except Exception as e:
            self._print_verbose(f"   ❌ LLM error: {e}")
            trace["error"] = str(e)
            # Fallback: use a generic question
            fallback = (
                f"Please describe your organization's current practices regarding "
                f"'{description}' ({subcategory_id}). "
                f"Include any relevant policies, responsibilities, and evidence."
            )
            return fallback, trace

    # ═══════════════════════════════════════════════════════════════
    #  INTERNAL — Response extraction
    # ═══════════════════════════════════════════════════════════════

    def _extract_response(self, row: Dict, user_answer: str) -> tuple:
        """
        Uses the LLM to extract structured profile data from the user's
        free-text answer, mapping it to the Current_* profile columns.

        Returns:
            (extracted_dict, trace_dict)
        """
        subcategory_id = row.get("Subcategory_ID", "")
        category = row.get("Category", "")
        description = row.get("Subcategory_Description", "")

        # Build the extraction prompt with explicit column definitions
        prompt = f"""You are a cybersecurity compliance data extraction assistant.

CONTEXT:
The user has answered an interview question about NIST CSF 2.0 subcategory:
- ID: {subcategory_id}
- Category: {category}
- Description: {description}

USER'S ANSWER:
"{user_answer}"

YOUR TASK:
Extract and organize the user's answer into the following structured fields.
For each field, extract the relevant information from the answer.
If the answer does not contain information for a field, write "Not specified" for that field.

Return ONLY a valid JSON object with these exact keys:

{{
    "Included_in_Profile": "Yes or No — whether this subcategory is relevant to the organization",
    "Rationale": "Brief justification for inclusion/exclusion",
    "Current_Priority": "High, Medium, Low, or N/A — how important this area is currently",
    "Current_Status": "Brief description of the current state of implementation",
    "Current_Policies_Processes_Procedures": "Any formal policies, processes, or procedures in place",
    "Current_Internal_Practices": "Informal or internal practices being followed",
    "Current_Roles_and_Responsibilities": "Who is responsible for this area",
    "Current_Selected_Informative_References": "Any standards, frameworks, or references currently used",
    "Current_Artifacts_and_Evidence": "Any documentation, logs, or evidence available"
}}

IMPORTANT:
- Return ONLY valid JSON, no markdown, no explanation
- Be concise but accurate
- Map the user's natural language to the correct fields
- If the user clearly indicates this area is not applicable, set Included_in_Profile to "No"
"""

        messages = [
            {
                "role": "system",
                "content": "You are a data extraction assistant. Return ONLY valid JSON, no other text.",
            },
            {"role": "user", "content": prompt},
        ]

        trace = {
            "prompt": prompt,
            "messages": messages,
            "model": self.model_name,
            "user_answer": user_answer,
        }

        self._print_verbose("\n🔍 [Response Extraction] Sending to LLM...")
        self._print_verbose(f"   User answer: {user_answer[:100]}...")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=800,
            )

            raw_content = response.choices[0].message.content or ""
            reasoning = getattr(response.choices[0].message, "reasoning_content", None)
            content = raw_content.strip()

            trace["response"] = {
                "content": raw_content,
                "reasoning_content": reasoning,
                "tokens": {
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                },
            }

            self._print_verbose(f"   ✅ Raw response: {len(content)} chars")
            self._print_verbose(f"   Tokens — in: {response.usage.prompt_tokens}, out: {response.usage.completion_tokens}")
            if reasoning:
                self._print_verbose(f"   🧠 Reasoning: {reasoning[:200]}...")
            self._print_verbose(f"   Raw content:\n{content}")

            # Clean markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            extracted = json.loads(content)
            trace["parsed"] = extracted

            # Validate: only keep known columns
            validated = {}
            for col in self.CURRENT_PROFILE_COLUMNS:
                validated[col] = extracted.get(col, "Not specified")

            return validated, trace

        except json.JSONDecodeError as e:
            self._print_verbose(f"   ❌ JSON parse error: {e}")
            trace["error"] = f"JSONDecodeError: {e}"
            # Fallback: store raw answer
            return self._fallback_extraction(user_answer), trace

        except Exception as e:
            self._print_verbose(f"   ❌ LLM error: {e}")
            trace["error"] = str(e)
            return self._fallback_extraction(user_answer), trace

    def _fallback_extraction(self, user_answer: str) -> Dict[str, str]:
        """
        Fallback when LLM extraction fails: store the raw answer
        in Current_Status so nothing is lost.
        """
        return {
            "Included_in_Profile": "Yes",
            "Rationale": "Auto-included (extraction fallback)",
            "Current_Priority": "Not specified",
            "Current_Status": user_answer,
            "Current_Policies_Processes_Procedures": "Not specified",
            "Current_Internal_Practices": "Not specified",
            "Current_Roles_and_Responsibilities": "Not specified",
            "Current_Selected_Informative_References": "Not specified",
            "Current_Artifacts_and_Evidence": "Not specified",
        }

    # ═══════════════════════════════════════════════════════════════
    #  INTERNAL — Progress & logging
    # ═══════════════════════════════════════════════════════════════

    def _show_progress(self):
        """Display current completion progress."""
        summary = self.manager.get_progress_summary()

        # Also compute in-scope progress
        if self.SCOPE_SUBCATEGORIES is not None:
            in_scope = self.manager.df[
                self.manager.df["Subcategory_ID"].isin(self.SCOPE_SUBCATEGORIES)
            ]
            # Count unique subcategory IDs (not rows)
            scope_done = in_scope[in_scope["Completion_Status"] == "DONE"][
                "Subcategory_ID"
            ].nunique()
            scope_total = in_scope["Subcategory_ID"].nunique()
            scope_pct = (scope_done / scope_total * 100) if scope_total > 0 else 0

            self._print_always(
                f"\n📊 Progress (in-scope): {scope_done}/{scope_total} "
                f"subcategories ({scope_pct:.0f}%)"
            )
        else:
            self._print_always(
                f"\n📊 Progress: {summary['completed']}/{summary['total_items']} "
                f"({summary['percentage']:.1f}%)"
            )

    def _save_run_log(self):
        """Save the full run log to a JSON file."""
        log_path = os.path.join(self.log_dir, "interview_log.json")
        log_data = {
            "run_timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "scope": list(self.SCOPE_SUBCATEGORIES) if self.SCOPE_SUBCATEGORIES else "ALL",
            "turns": self._turn_logs,
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        self._print_verbose(f"   💾 Log saved: {log_path}")

    # ═══════════════════════════════════════════════════════════════
    #  INTERNAL — Output helpers
    # ═══════════════════════════════════════════════════════════════

    def _print_always(self, message: str):
        """Print regardless of verbose setting (user-facing messages)."""
        print(message)

    def _print_verbose(self, message: str):
        """Print only if verbose mode is on (internal traces)."""
        if self.verbose:
            print(message)


# ═══════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NIST CSF 2.0 Interview Engine"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose mode (show all internal LLM traces)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss",
        help="LLM model name (default: gpt-oss)",
    )
    args = parser.parse_args()

    engine = InterviewEngine(
        verbose=args.verbose,
        model_name=args.model,
    )
    engine.start()
