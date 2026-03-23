"""
INTERVIEW ENGINE - Phase 1+2: Core Interview Loop + Structured Extraction

Orchestrates the NIST CSF 2.0 compliance interview by:
- Managing compilation sessions (new / resume) in a Compilazioni/ folder
- Pulling the next PENDING subcategory from ProfileManager
- Fetching enriched context from CSF catalog and SP800-53 mappings
- Building a professional, contextualized question via LLM
- Extracting structured data from the user's free-text answer
- Saving progress incrementally to the session's profile CSV

Supports verbose and non-verbose modes with full JSON logging.
"""
import os
import re
import json
import shutil
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

import pandas as pd
from openai import OpenAI

from profile_manager import ProfileManager
from pandas_agent_manual import ManualPandasAgent


# ═══════════════════════════════════════════════════════════════════
#  DATA PATHS
# ═══════════════════════════════════════════════════════════════════
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "cleaned")
CSF_CATALOG_PATH = os.path.join(DATA_DIR, "csf_2_0_catalog.csv")
CSF_SP800_MAPPING_PATH = os.path.join(DATA_DIR, "csf_to_sp800_53_mapping.csv")
SP800_CATALOG_PATH = os.path.join(DATA_DIR, "sp800_53_catalog.csv")


# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════

COMPILAZIONI_DIR = os.path.join(os.getcwd(), "Compilazioni")
PROFILE_FILENAME_PREFIX = "profile_"
PROFILE_FILENAME_SUFFIX = ".csv"


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

    # ── LLM parameters ──────────────────────────────────────────────
    # The extraction call needs very high max_tokens because reasoning
    # models (like gpt-oss) can consume thousands of tokens on internal
    # reasoning loops before producing the visible JSON output.
    EXTRACTION_MAX_TOKENS = 4096
    EXTRACTION_RETRIES = 3

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

        # ProfileManager will be initialized during session selection
        self.manager: Optional[ProfileManager] = None

        # Run-level logging directory (set during session selection)
        self.log_dir = log_dir

        # Accumulated interview log (saved at each turn)
        self._turn_logs: List[Dict[str, Any]] = []

        # ── Load reference data for catalog context ──────────────
        self._csf_catalog = self._load_csv_safe(CSF_CATALOG_PATH)
        self._sp800_mapping = self._load_csv_safe(CSF_SP800_MAPPING_PATH)
        self._sp800_catalog = self._load_csv_safe(SP800_CATALOG_PATH)

        # ── Pandas Agent for catalog context queries ───────────
        self._pandas_agent = ManualPandasAgent(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            temperature=0.0,
            verbose=verbose,   # follow the engine's verbosity
        )

    @staticmethod
    def _load_csv_safe(path: str) -> Optional[pd.DataFrame]:
        """Load a CSV file, returning None if it doesn't exist."""
        if os.path.exists(path):
            return pd.read_csv(path, dtype=str)
        return None

    # ═══════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ═══════════════════════════════════════════════════════════════

    def start(self):
        """
        Main entry point.
        1. Ask user: new compilation or resume existing
        2. Run the interview loop
        """
        self._print_always(
            "\n" + "=" * 62
            + "\n🛡️  NIST CSF 2.0 — Interview Engine"
            + f"\n   Model: {self.model_name}"
            + "\n" + "=" * 62
        )

        # ── Session selection ────────────────────────────────────
        session_path = self._session_selection()
        if session_path is None:
            self._print_always("\n👋 Goodbye!")
            return

        # Initialize ProfileManager on the selected session file
        # verbose=False suppresses internal "Stato salvato" messages
        self.manager = ProfileManager(
            profile_name=os.path.basename(session_path),
            save_dir=os.path.dirname(session_path),
            verbose=False,
        )

        # Setup log directory alongside the profile
        if self.log_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = os.path.join(
                os.path.dirname(session_path), f"logs_{ts}"
            )
        os.makedirs(self.log_dir, exist_ok=True)

        self._print_always(f"   Profile: {session_path}")
        self._print_always(f"   Logs:    {self.log_dir}")
        self._print_always(
            f"   Scope:   {sorted(self.SCOPE_SUBCATEGORIES) if self.SCOPE_SUBCATEGORIES else 'ALL subcategories'}"
        )

        # ── Interview loop ───────────────────────────────────────
        self._run_interview_loop()

    # ═══════════════════════════════════════════════════════════════
    #  SESSION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════

    def _session_selection(self) -> Optional[str]:
        """
        Prompts user to start a new compilation or resume an existing one.
        Returns the path to the profile CSV, or None to exit.
        """
        os.makedirs(COMPILAZIONI_DIR, exist_ok=True)

        # Find existing compilations
        existing = self._list_compilations()

        self._print_always(
            "\n╔════════════════════════════════════════════════════════════╗"
            "\n║           Select Compilation Mode                        ║"
            "\n╠════════════════════════════════════════════════════════════╣"
        )

        if existing:
            self._print_always(
                "║                                                          ║"
                "\n║  [N] Start a NEW compilation                             ║"
                "\n║  [R] Resume an existing compilation                      ║"
                "\n║  [Q] Quit                                                ║"
                "\n║                                                          ║"
                "\n╚════════════════════════════════════════════════════════════╝"
            )
        else:
            self._print_always(
                "║                                                          ║"
                "\n║  No existing compilations found.                         ║"
                "\n║  A new compilation will be created.                      ║"
                "\n║                                                          ║"
                "\n╚════════════════════════════════════════════════════════════╝"
            )
            return self._create_new_session()

        while True:
            choice = input("\nYour choice [N/R/Q] ▶ ").strip().upper()

            if choice == "Q":
                return None

            if choice == "N":
                return self._create_new_session()

            if choice == "R":
                return self._resume_session(existing)

            self._print_always("⚠️  Invalid choice. Please enter N, R, or Q.")

    def _list_compilations(self) -> List[Dict[str, Any]]:
        """Lists all existing compilation files in Compilazioni/."""
        compilations = []
        if not os.path.exists(COMPILAZIONI_DIR):
            return compilations

        for fname in sorted(os.listdir(COMPILAZIONI_DIR)):
            if fname.startswith(PROFILE_FILENAME_PREFIX) and fname.endswith(PROFILE_FILENAME_SUFFIX):
                fpath = os.path.join(COMPILAZIONI_DIR, fname)
                try:
                    # Quick peek at progress
                    df = pd.read_csv(fpath, usecols=["Completion_Status"], dtype={"Completion_Status": str})
                    total = len(df)
                    done = (df["Completion_Status"] == "DONE").sum()
                    pct = (done / total * 100) if total > 0 else 0

                    # Extract timestamp from filename
                    ts_match = re.search(r"(\d{8}_\d{6})", fname)
                    ts_str = ts_match.group(1) if ts_match else "unknown"

                    compilations.append({
                        "filename": fname,
                        "path": fpath,
                        "timestamp": ts_str,
                        "done": done,
                        "total": total,
                        "percentage": pct,
                    })
                except Exception:
                    pass  # Skip corrupt files

        return compilations

    def _create_new_session(self) -> str:
        """Creates a fresh profile CSV in Compilazioni/ with a timestamp."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{PROFILE_FILENAME_PREFIX}{ts}{PROFILE_FILENAME_SUFFIX}"
        dest_path = os.path.join(COMPILAZIONI_DIR, fname)

        # Use ProfileManager to create a pristine copy from the catalog
        temp_manager = ProfileManager(verbose=False)
        temp_manager.create_fresh_copy(dest_path)

        self._print_always(f"\n✅ New compilation created: {fname}")
        return dest_path

    def _resume_session(self, compilations: List[Dict[str, Any]]) -> Optional[str]:
        """Shows existing compilations and lets user pick one."""
        self._print_always("\n📂 Existing compilations:\n")
        for i, comp in enumerate(compilations, 1):
            dt = comp["timestamp"]
            # Format timestamp nicely
            try:
                dt_formatted = datetime.strptime(dt, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                dt_formatted = dt
            self._print_always(
                f"   [{i}] {comp['filename']}"
                f"\n       Created: {dt_formatted}"
                f"\n       Progress: {comp['done']}/{comp['total']} ({comp['percentage']:.0f}%)"
                f"\n"
            )

        while True:
            choice = input(f"Select compilation [1-{len(compilations)}] or Q to go back ▶ ").strip()

            if choice.upper() == "Q":
                return self._session_selection()

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(compilations):
                    selected = compilations[idx]
                    self._print_always(f"\n✅ Resuming: {selected['filename']}")
                    return selected["path"]
                else:
                    self._print_always(f"⚠️  Please enter a number between 1 and {len(compilations)}.")
            except ValueError:
                self._print_always("⚠️  Invalid input. Enter a number or Q.")

    # ═══════════════════════════════════════════════════════════════
    #  INTERVIEW LOOP
    # ═══════════════════════════════════════════════════════════════

    def _run_interview_loop(self):
        """
        Runs the interview loop until all in-scope subcategories are
        DONE, the user types /quit, or there are no more PENDING items.
        """
        self._print_always(
            "\n╔════════════════════════════════════════════════════════════╗"
            "\n║  Interview Started                                       ║"
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

            # Fetch enriched catalog context for this subcategory
            catalog_ctx, catalog_ctx_trace = self._fetch_catalog_context(subcategory_id)

            # Build and show question
            question, question_trace = self._build_question(row, catalog_ctx)
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

            # Extract structured response (with retry)
            extracted, extraction_trace = self._extract_response(
                row, user_input, catalog_ctx
            )

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
                "catalog_context": catalog_ctx_trace,
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
    #  INTERNAL — Catalog context enrichment
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _normalize_subcategory_id(sub_id: str) -> str:
        """
        Normalize subcategory ID between CSF 2.0 format (ID.AM-01)
        and the mapping file format (ID.AM-1), stripping leading zeros.
        Returns the version WITHOUT leading zeros (e.g. ID.AM-1).
        """
        # "ID.AM-01" → "ID.AM-1"
        return re.sub(r'-0*(\d+)$', r'-\1', sub_id)

    @staticmethod
    def _normalize_control_id(ctrl_id: str) -> str:
        """
        Normalize SP800-53 control IDs.
        Mapping uses 'CM-8', catalog uses 'CM-08'.
        Returns the version WITH leading zeros (e.g. CM-08).
        """
        # "CM-8" → "CM-08"  (pad to 2 digits)
        def _pad(m):
            return f"-{m.group(1).zfill(2)}"
        return re.sub(r'-(?!\d{2})(?!\()(\d+)', _pad, ctrl_id.strip())

    def _fetch_catalog_context(
        self, subcategory_id: str
    ) -> tuple:
        """
        Fetches enriched context for a subcategory using the Pandas Agent.
        Each lookup goes through the full ReAct pipeline so the user can
        see: prompt → reasoning → generated code → execution result.

        Steps:
        1. CSF catalog: description + examples from csf_2_0_catalog.csv
        2. SP800-53 mapping: control IDs from csf_to_sp800_53_mapping.csv
        3. SP800-53 details: control names + statements from sp800_53_catalog.csv

        Returns:
            (context_dict, trace_dict)
        """
        ctx: Dict[str, Any] = {
            "catalog_description": "",
            "catalog_examples": "",
            "sp800_controls": [],
        }
        trace: Dict[str, Any] = {
            "subcategory_id": subcategory_id,
            "pandas_agent_queries": [],
        }

        # ---- Step 1: CSF Catalog lookup via Pandas Agent ----
        if self._csf_catalog is not None:
            question_1 = (
                f"Get the Subcategory_Description and Implementation_Examples "
                f"for the row where Subcategory_ID == '{subcategory_id}'. "
                f"Return them as a string."
            )
            self._print_verbose(f"\n📚 [Pandas Agent] Step 1: CSF Catalog lookup for {subcategory_id}")

            agent_result_1 = self._pandas_agent.query(
                df=self._csf_catalog,
                question=question_1,
                max_iterations=3,
            )

            trace["pandas_agent_queries"].append({
                "step": "csf_catalog_lookup",
                "question": question_1,
                "success": agent_result_1["success"],
                "answer": agent_result_1.get("answer", ""),
                "iterations": agent_result_1["iterations"],
                "history": agent_result_1.get("history", []),
            })

            # Also do a direct lookup to populate ctx reliably
            # (the agent answer is for logging/transparency,
            #  the direct lookup ensures we always have the data)
            cat_rows = self._csf_catalog[
                self._csf_catalog["Subcategory_ID"] == subcategory_id
            ]
            if not cat_rows.empty:
                row = cat_rows.iloc[0]
                desc = row.get("Subcategory_Description", "")
                examples_raw = row.get("Implementation_Examples", "")
                if pd.isna(desc):
                    desc = ""
                if pd.isna(examples_raw):
                    examples_raw = ""
                ctx["catalog_description"] = desc
                ctx["catalog_examples"] = examples_raw

        # ---- Step 2: SP800-53 mapping lookup via Pandas Agent ----
        normalized_id = self._normalize_subcategory_id(subcategory_id)
        raw_controls = []

        if self._sp800_mapping is not None:
            question_2 = (
                f"Get the SP800_53_Controls column value for the row where "
                f"Subcategory_ID == '{normalized_id}'. Return just the value."
            )
            self._print_verbose(f"\n📚 [Pandas Agent] Step 2: SP800-53 mapping for {normalized_id}")

            agent_result_2 = self._pandas_agent.query(
                df=self._sp800_mapping,
                question=question_2,
                max_iterations=3,
            )

            trace["pandas_agent_queries"].append({
                "step": "sp800_mapping_lookup",
                "normalized_id": normalized_id,
                "question": question_2,
                "success": agent_result_2["success"],
                "answer": agent_result_2.get("answer", ""),
                "iterations": agent_result_2["iterations"],
                "history": agent_result_2.get("history", []),
            })

            # Direct lookup for reliable data extraction
            map_rows = self._sp800_mapping[
                self._sp800_mapping["Subcategory_ID"] == normalized_id
            ]
            if not map_rows.empty:
                controls_str = map_rows.iloc[0].get("SP800_53_Controls", "")
                if pd.isna(controls_str):
                    controls_str = ""
                raw_controls = [c.strip() for c in controls_str.split(",") if c.strip()]

        # ---- Step 3: SP800-53 catalog detail lookup via Pandas Agent ----
        if self._sp800_catalog is not None and raw_controls:
            # Normalize control IDs for lookup
            norm_controls = [self._normalize_control_id(c) for c in raw_controls]
            controls_filter = ", ".join(f"'{c}'" for c in norm_controls)

            question_3 = (
                f"Get Control_ID, Control_Name, and Control_Statement for rows where "
                f"Control_ID is in [{controls_filter}]. Return as a table."
            )
            self._print_verbose(f"\n📚 [Pandas Agent] Step 3: SP800-53 catalog details for {norm_controls}")

            agent_result_3 = self._pandas_agent.query(
                df=self._sp800_catalog,
                question=question_3,
                max_iterations=3,
            )

            trace["pandas_agent_queries"].append({
                "step": "sp800_catalog_lookup",
                "control_ids_queried": norm_controls,
                "question": question_3,
                "success": agent_result_3["success"],
                "answer": agent_result_3.get("answer", ""),
                "iterations": agent_result_3["iterations"],
                "history": agent_result_3.get("history", []),
            })

            # Direct lookup for reliable data
            control_details = []
            for norm_ctrl in norm_controls:
                ctrl_rows = self._sp800_catalog[
                    self._sp800_catalog["Control_ID"] == norm_ctrl
                ]
                if not ctrl_rows.empty:
                    crow = ctrl_rows.iloc[0]
                    detail = {
                        "control_id": norm_ctrl,
                        "control_name": crow.get("Control_Name", ""),
                        "control_statement": crow.get("Control_Statement", "")[:500],
                    }
                    control_details.append(detail)
            ctx["sp800_controls"] = control_details

        self._print_verbose(f"\n📚 [Catalog Context Summary] {subcategory_id}:")
        self._print_verbose(f"   Description: {ctx['catalog_description'][:100]}...")
        self._print_verbose(f"   Examples: {'Yes' if ctx['catalog_examples'] else 'No'}")
        self._print_verbose(
            f"   SP800-53 Controls: {[c['control_id'] for c in ctx['sp800_controls']]}"
        )

        return ctx, trace

    # ═══════════════════════════════════════════════════════════════
    #  INTERNAL — Question building
    # ═══════════════════════════════════════════════════════════════

    def _build_question(
        self, row: Dict, catalog_ctx: Optional[Dict] = None
    ) -> tuple:
        """
        Uses the LLM to formulate a clear, professional interview question
        from the subcategory's catalog data, enriched with SP800-53 context.

        Returns:
            (question_text, trace_dict)
        """
        subcategory_id = row.get("Subcategory_ID", "")
        category = row.get("Category", "")
        description = row.get("Subcategory_Description", "")
        examples = row.get("Implementation_Examples", "")

        # Handle NaN in examples
        if pd.isna(examples) or str(examples).strip().lower() == "nan":
            examples = "No specific implementation examples available."

        # Build SP800-53 controls context
        sp800_section = ""
        if catalog_ctx and catalog_ctx.get("sp800_controls"):
            controls_text = "\n".join(
                f"  - {c['control_id']} ({c['control_name']}): {c['control_statement'][:200]}"
                for c in catalog_ctx["sp800_controls"]
            )
            sp800_section = f"\nRELATED NIST SP 800-53 CONTROLS:\n{controls_text}\n"

        prompt = f"""You are a professional cybersecurity auditor conducting a NIST CSF 2.0 compliance interview.

Your task: formulate ONE clear, conversational question to ask the interviewee about the following NIST subcategory.

SUBCATEGORY DETAILS:
- ID: {subcategory_id}
- Category: {category}
- Description: {description}
- Implementation Examples: {examples}
{sp800_section}
INSTRUCTIONS:
1. Ask about their CURRENT state regarding this subcategory
2. Be specific but not overwhelming — ask one focused question
3. Briefly explain what this subcategory is about so the interviewee understands the context
4. The question MUST explicitly ask the interviewee to provide information about ALL of the following areas:
   a) Whether this area is applicable/relevant to their organization
   b) The PRIORITY they assign to this area (High, Medium, or Low)
   c) Their current implementation status
   d) Any formal policies, processes, or procedures in place
   e) Internal or informal practices being followed
   f) Who is responsible (roles and responsibilities)
   g) Any standards, frameworks, or informative references they follow for this area (e.g. ISO 27001, NIST SP 800-53, sector-specific regulations, or any other relevant standard for their industry/context)
   h) Any documentary evidence or artifacts they can provide
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
                max_tokens=900,
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
                f"Include: priority (High/Medium/Low), policies, practices, "
                f"responsibilities, informative references, and evidence."
            )
            return fallback, trace

    # ═══════════════════════════════════════════════════════════════
    #  INTERNAL — Response extraction (with retry)
    # ═══════════════════════════════════════════════════════════════

    def _extract_response(
        self, row: Dict, user_answer: str,
        catalog_ctx: Optional[Dict] = None,
    ) -> tuple:
        """
        Uses the LLM to extract structured profile data from the user's
        free-text answer.  Retries up to EXTRACTION_RETRIES times if
        the model returns empty content (reasoning consumed all tokens).

        Returns:
            (extracted_dict, trace_dict)
        """
        subcategory_id = row.get("Subcategory_ID", "")
        category = row.get("Category", "")
        description = row.get("Subcategory_Description", "")

        # Build SP800-53 context for extraction guidance
        sp800_hint = ""
        if catalog_ctx and catalog_ctx.get("sp800_controls"):
            ctrl_list = ", ".join(
                f"{c['control_id']} ({c['control_name']})"
                for c in catalog_ctx["sp800_controls"]
            )
            sp800_hint = (
                f"\nNOTE: The related NIST SP 800-53 controls for this subcategory are: {ctrl_list}.\n"
                f"If the user mentions any of these or other standards/frameworks, include them in "
                f"Current_Selected_Informative_References.\n"
            )

        prompt = f"""Extract structured data from this interview answer about NIST CSF 2.0 subcategory {subcategory_id} ({description}).
{sp800_hint}
ANSWER: "{user_answer}"

Return ONLY this JSON (no markdown, no explanation):

{{
    "Included_in_Profile": "Yes or No",
    "Rationale": "Brief justification",
    "Current_Priority": "High/Medium/Low, or 'Not specified' if user didn't say",
    "Current_Status": "Implementation status",
    "Current_Policies_Processes_Procedures": "Formal policies/procedures",
    "Current_Internal_Practices": "Informal practices and tools used",
    "Current_Roles_and_Responsibilities": "Who is responsible",
    "Current_Selected_Informative_References": "Only standards/frameworks/regulations the user explicitly named (e.g. ISO 27001, NIST SP 800-53). Put tools in Internal_Practices instead.",
    "Current_Artifacts_and_Evidence": "Documentation and evidence"
}}

Rules: Use 'Not specified' for missing fields. Only extract what the user actually said."""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a data extraction assistant. "
                    "Think briefly, then output ONLY valid JSON. "
                    "Do not over-analyze or repeat yourself."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        trace = {
            "prompt": prompt,
            "messages": messages,
            "model": self.model_name,
            "user_answer": user_answer,
            "attempts": [],
        }

        self._print_verbose("\n🔍 [Response Extraction] Sending to LLM...")
        self._print_verbose(f"   User answer: {user_answer[:100]}...")

        # Retry loop: reasoning models may consume all tokens on reasoning
        for attempt in range(1, self.EXTRACTION_RETRIES + 1):
            attempt_trace = {"attempt": attempt}

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=self.EXTRACTION_MAX_TOKENS,
                )

                raw_content = response.choices[0].message.content or ""
                reasoning = getattr(response.choices[0].message, "reasoning_content", None)
                content = raw_content.strip()

                attempt_trace["response"] = {
                    "content": raw_content,
                    "reasoning_content": reasoning,
                    "tokens": {
                        "input": response.usage.prompt_tokens,
                        "output": response.usage.completion_tokens,
                    },
                }

                self._print_verbose(
                    f"   [Attempt {attempt}/{self.EXTRACTION_RETRIES}] "
                    f"Raw response: {len(content)} chars, "
                    f"Tokens — in: {response.usage.prompt_tokens}, out: {response.usage.completion_tokens}"
                )
                if reasoning:
                    self._print_verbose(f"   🧠 Reasoning: {reasoning[:200]}...")

                # If content is empty, reasoning consumed all tokens → retry
                if not content:
                    self._print_verbose(
                        f"   ⚠️  Empty content (reasoning consumed all tokens). "
                        f"{'Retrying...' if attempt < self.EXTRACTION_RETRIES else 'Using fallback.'}"
                    )
                    attempt_trace["error"] = "Empty content — reasoning consumed all tokens"
                    trace["attempts"].append(attempt_trace)
                    continue

                self._print_verbose(f"   Raw content:\n{content}")

                # Clean markdown code blocks if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                extracted = json.loads(content)
                attempt_trace["parsed"] = extracted
                trace["attempts"].append(attempt_trace)

                # Validate: only keep known columns
                validated = {}
                for col in self.CURRENT_PROFILE_COLUMNS:
                    validated[col] = extracted.get(col, "Not specified")

                trace["success"] = True
                return validated, trace

            except json.JSONDecodeError as e:
                self._print_verbose(f"   ❌ JSON parse error: {e}")
                attempt_trace["error"] = f"JSONDecodeError: {e}"
                trace["attempts"].append(attempt_trace)
                # Don't retry on JSON errors — the content exists but is malformed
                break

            except Exception as e:
                self._print_verbose(f"   ❌ LLM error: {e}")
                attempt_trace["error"] = str(e)
                trace["attempts"].append(attempt_trace)
                break

        # All attempts exhausted → fallback
        self._print_verbose("   ⚠️  Using fallback extraction (raw answer saved).")
        trace["success"] = False
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
            "profile_path": self.manager.state_path,
            "scope": sorted(self.SCOPE_SUBCATEGORIES) if self.SCOPE_SUBCATEGORIES else "ALL",
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
