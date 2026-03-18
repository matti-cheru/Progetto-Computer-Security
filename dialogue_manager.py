"""
DIALOGUE MANAGER - Phase 3: Structured Dialogue System

Gestisce il dialogo strutturato per estrarre il profilo di compliance personalizzato.
Implementa:
- Decision tree basato su risposte utente
- State management (profilo aziendale, conversazione)
- Orchestrazione tra ManualPandasAgent (dinamico) e Direct Methods (veloce)
- Logging completo di tutti i dati scambiati

Questo è il CERVELLO del sistema che guida l'utente attraverso il processo compliance.
"""
import os
import json
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime

from openai import OpenAI
from nist_data_loader import NISTDataLoader
from pandas_agent_manual import ManualPandasAgent
from pandas_agent_core import NISTComplianceAgent


class ConversationPhase(Enum):
    """Fasi del dialogo strutturato"""
    WELCOME = "welcome"
    INDUSTRY_IDENTIFICATION = "industry"
    SIZE_ASSESSMENT = "size"
    DATA_SENSITIVITY = "data_sensitivity"
    INFRASTRUCTURE = "infrastructure"
    CURRENT_MATURITY = "maturity"
    CONTROL_IDENTIFICATION = "controls"
    PRIORITIZATION = "prioritization"
    RECOMMENDATIONS = "recommendations"
    COMPLETION = "completion"


@dataclass
class CompanyProfile:
    """Profilo aziendale estratto dal dialogo"""
    industry: Optional[str] = None
    size: Optional[str] = None  # small, medium, large
    employees: Optional[int] = None
    data_types: List[str] = field(default_factory=list)
    data_sensitivity: Optional[str] = None  # low, medium, high, critical
    infrastructure: List[str] = field(default_factory=list)  # on-premise, cloud, hybrid
    compliance_requirements: List[str] = field(default_factory=list)  # HIPAA, PCI-DSS, etc.
    current_maturity: Optional[str] = None  # none, basic, intermediate, advanced
    critical_assets: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return asdict(self)
    
    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ConversationTurn:
    """Un singolo turno di conversazione"""
    turn_number: int
    phase: ConversationPhase
    ai_question: str
    user_response: str
    extracted_info: Dict[str, Any]
    agent_used: str  # "manual_pandas", "direct_method", "llm_extraction"
    agent_response: Optional[Dict] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class NISTComplianceDialogueManager:
    """
    Gestisce il dialogo strutturato per compliance NIST CSF 2.0.
    
    Orchestration Pattern:
    - User input → LLM extraction → Update profile
    - Based on profile → Query NIST data (ManualPandasAgent o Direct)
    - Generate next question → Continue dialogue
    """
    
    def __init__(
        self,
        base_url: str = "https://gpustack.ing.unibs.it/v1",
        model_name: str = "gpt-oss",
        api_key: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Inizializza il Dialogue Manager.
        
        Args:
            base_url: URL cluster LLM
            model_name: Modello da usare
            api_key: API key
            verbose: Se True, stampa tutti i dettagli
        """
        self.verbose = verbose
        
        if self.verbose:
            print("\n" + "="*80)
            print("🚀 INITIALIZING NIST COMPLIANCE DIALOGUE MANAGER")
            print("="*80)
        
        # API Key
        if api_key is None:
            api_key = os.environ.get(
                "GPUSTACK_API_KEY",
                "gpustack_8a00037ab4220858_6479d07686028e2f970357b1a81200e4"
            )
        
        # Core components
        if self.verbose:
            print("\n📦 Loading components...")
        
        # 1. Data Loader (Phase 1)
        self.data_loader = NISTDataLoader()
        self.datasets = self.data_loader.load_all()
        
        if self.verbose:
            print(f"   ✅ NISTDataLoader: {len(self.datasets)} datasets loaded")
        
        # 2. Manual Pandas Agent (Phase 2 - Dynamic queries)
        self.pandas_agent = ManualPandasAgent(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            verbose=False  # Controlliamo noi il logging
        )
        
        if self.verbose:
            print(f"   ✅ ManualPandasAgent: Ready for dynamic queries")
        
        # 3. Direct Methods (Phase 2 - Fast operations)
        self.direct_methods = NISTComplianceAgent(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            verbose=False
        )
        
        if self.verbose:
            print(f"   ✅ DirectMethods: Ready for fast lookups")
        
        # 4. LLM client per extraction e generation
        self.llm_client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        
        if self.verbose:
            print(f"   ✅ LLM Client: {model_name} ready")
        
        # State
        self.current_phase = ConversationPhase.WELCOME
        self.company_profile = CompanyProfile()
        self.conversation_history: List[ConversationTurn] = []
        self.identified_controls: List[str] = []
        self.turn_counter = 0

        # Tracciamento dettagliato (prompt, reasoning, token, query pandas, ecc.)
        self.detailed_turn_logs: List[Dict[str, Any]] = []
        self._last_extraction_trace: Optional[Dict[str, Any]] = None
        
        # Setup run directories for logs
        self.timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(os.getcwd(), f"run_{self.timestamp_str}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.pandas_logs_dir = os.path.join(self.run_dir, "pandas_agent_logs")
        os.makedirs(self.pandas_logs_dir, exist_ok=True)
        
        if self.verbose:
            print(f"   📂 Created run directory: {self.run_dir}")
            print("\n✅ Dialogue Manager initialized successfully")
            print("="*80)
    
    def _log(self, message: str, level: str = "INFO"):
        """Helper per logging"""
        if self.verbose:
            prefix = {
                "INFO": "ℹ️ ",
                "QUESTION": "❓",
                "USER": "👤",
                "EXTRACT": "🔍",
                "QUERY": "🗄️ ",
                "AGENT": "🤖",
                "UPDATE": "📝",
                "RESULT": "✅"
            }.get(level, "  ")
            print(f"{prefix} {message}")
    
    def start_dialogue(self) -> str:
        """
        Inizia il dialogo di compliance.
        
        Returns:
            Messaggio di benvenuto
        """
        self.current_phase = ConversationPhase.WELCOME
        
        welcome_message = """
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║          🛡️  NIST CSF 2.0 COMPLIANCE ASSESSMENT TOOL  🛡️                  ║
║                                                                            ║
║  Welcome! I'm your AI Compliance Officer assistant.                       ║
║                                                                            ║
║  I'll guide you through a structured dialogue to:                         ║
║    ✓ Understand your organization's cybersecurity needs                   ║
║    ✓ Identify relevant NIST CSF 2.0 controls                              ║
║    ✓ Map them to SP 800-53 Rev 5 recommendations                          ║
║    ✓ Provide actionable, prioritized guidance                             ║
║                                                                            ║
║  This assessment takes approximately 10-15 minutes.                       ║
║  Please answer questions as accurately as possible.                       ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

Let's begin! 🚀
"""
        
        if self.verbose:
            print(welcome_message)
        
        # Prima domanda
        return self._next_question()
    
    def _next_question(self) -> str:
        """
        Genera la prossima domanda basandosi sulla fase corrente.
        
        Returns:
            Domanda per l'utente
        """
        phase_questions = {
            ConversationPhase.WELCOME: 
                "📊 First, tell me about your organization.\n\n"
                "What INDUSTRY does your organization operate in?\n"
                "(e.g., Healthcare, Finance, Retail, Manufacturing, Technology, Education, Government, etc.)",
            
            ConversationPhase.INDUSTRY_IDENTIFICATION:
                "👥 What is the SIZE of your organization?\n\n"
                "Please provide:\n"
                "- Approximate number of employees\n"
                "- Classification (small/medium/large)\n"
                "(e.g., 'Small company with 30 employees' or 'Enterprise with 5000+ employees')",
            
            ConversationPhase.SIZE_ASSESSMENT:
                "💾 What types of DATA does your organization handle?\n\n"
                "Examples:\n"
                "- Personal data (PII)\n"
                "- Financial data\n"
                "- Health records (PHI)\n"
                "- Payment card data\n"
                "- Intellectual property\n"
                "- Other sensitive information\n\n"
                "Please describe the main types of data you collect, process, or store.",
            
            ConversationPhase.DATA_SENSITIVITY:
                "🏢 Describe your IT INFRASTRUCTURE:\n\n"
                "- On-premise servers?\n"
                "- Cloud services (AWS, Azure, GCP)?\n"
                "- Hybrid environment?\n"
                "- Third-party services?\n"
                "- Remote work capabilities?\n\n"
                "Tell me how your IT systems are set up.",
            
            ConversationPhase.INFRASTRUCTURE:
                "📈 What is your CURRENT CYBERSECURITY MATURITY level?\n\n"
                "Choose one:\n"
                "1. None/Minimal - We have basic antivirus and passwords\n"
                "2. Basic - We have some security policies and controls\n"
                "3. Intermediate - We have documented processes and regular assessments\n"
                "4. Advanced - We have comprehensive security program with continuous monitoring\n\n"
                "Or describe your current state in your own words.",
            
            ConversationPhase.CURRENT_MATURITY:
                "🎯 Are there any SPECIFIC COMPLIANCE REQUIREMENTS or concerns?\n\n"
                "Examples:\n"
                "- Regulatory (HIPAA, PCI-DSS, GDPR, etc.)\n"
                "- Industry standards (ISO 27001, SOC 2)\n"
                "- Customer requirements\n"
                "- Recent incidents or vulnerabilities\n\n"
                "Type 'none' if no specific requirements.",
        }
        
        question = phase_questions.get(
            self.current_phase,
            "Thank you for the information. Let me analyze your requirements..."
        )
        
        return question
    
    def process_user_response(self, user_input: str) -> str:
        """
        Processa la risposta dell'utente e ritorna la prossima domanda o risultato.
        
        Questo è il CUORE del dialogue manager:
        1. Estrae informazioni dalla risposta (usando LLM)
        2. Aggiorna il profilo aziendale
        3. Interroga database NIST (con Pandas Agent o Direct)
        4. Decide la prossima fase
        5. Genera la risposta
        
        Args:
            user_input: Risposta dell'utente
            
        Returns:
            Prossima domanda o risultato finale
        """
        self.turn_counter += 1
        
        if self.verbose:
            print("\n" + "="*80)
            print(f"🔄 TURN {self.turn_counter} - Phase: {self.current_phase.value}")
            print("="*80)
        
        self._log(f"'{user_input}'", "USER")
        
        # 1. Estrazione informazioni dalla risposta
        self._log("Extracting information from response...", "EXTRACT")
        extracted_info = self._extract_information_from_response(user_input)
        extraction_trace = self._last_extraction_trace if self._last_extraction_trace else {}
        
        if self.verbose:
            print(f"\n📊 EXTRACTED INFORMATION:")
            print(json.dumps(extracted_info, indent=2))
        
        # 2. Aggiorna profilo aziendale
        self._log("Updating company profile...", "UPDATE")
        self._update_profile(extracted_info)
        
        if self.verbose:
            print(f"\n📋 UPDATED COMPANY PROFILE:")
            print(self.company_profile)
        
        # 3. Query NIST database se appropriato
        agent_response = None
        if self._should_query_nist_data():
            self._log("Querying NIST database for relevant controls...", "QUERY")
            agent_response = self._query_relevant_controls()
            
            if self.verbose and agent_response:
                print(f"\n🗄️  NIST QUERY RESULT:")
                print(json.dumps(agent_response, indent=2))
        
        # 4. Salva turn nella history
        agent_used = "none"
        if agent_response:
            agent_used = agent_response.get('agent_type', 'unknown')
        else:
            agent_used = "extraction_llm"

        turn = ConversationTurn(
            turn_number=self.turn_counter,
            phase=self.current_phase,
            ai_question=self._next_question() if self.turn_counter == 1 else self.conversation_history[-1].ai_question if self.conversation_history else "",
            user_response=user_input,
            extracted_info=extracted_info,
            agent_used=agent_used,
            agent_response=agent_response
        )
        self.conversation_history.append(turn)

        # Salva dettaglio completo del turno
        self.detailed_turn_logs.append({
            'turn': self.turn_counter,
            'phase': self.current_phase.value,
            'user_response': user_input,
            'extraction': extraction_trace,
            'profile_snapshot': self.company_profile.to_dict(),
            'agent_response': agent_response
        })
        
        # 5. Avanza alla prossima fase
        next_phase = self._determine_next_phase()
        
        if self.verbose:
            print(f"\n➡️  NEXT PHASE: {next_phase.value}")
        
        self.current_phase = next_phase
        
        # 6. Check se completato
        if self.current_phase == ConversationPhase.COMPLETION:
            return self._generate_final_report()
        
        # 7. Genera prossima domanda
        next_question = self._next_question()
        
        if self.verbose:
            print(f"\n❓ NEXT QUESTION:")
            print(next_question)
            print("="*80)
        
        return next_question
    
    def _extract_information_from_response(self, user_input: str) -> Dict[str, Any]:
        """
        Usa LLM per estrarre informazioni strutturate dalla risposta libera dell'utente.
        
        Questa è una funzionalità chiave: trasforma linguaggio naturale in dati strutturati.
        
        Args:
            user_input: Risposta in linguaggio naturale
            
        Returns:
            Dizionario con informazioni estratte
        """
        # Prompt per extraction basato sulla fase corrente
        extraction_prompts = {
            ConversationPhase.WELCOME: f"""Extract information about the INDUSTRY from this text:
"{user_input}"

Return ONLY a JSON object with this structure (no other text):
{{
    "industry": "industry name in lowercase",
    "industry_category": "healthcare|finance|retail|manufacturing|technology|education|government|other"
}}""",
            
            ConversationPhase.INDUSTRY_IDENTIFICATION: f"""Extract SIZE information:
"{user_input}"

Return ONLY JSON:
{{
    "size": "small|medium|large",
    "employees": number or null
}}""",
            
            ConversationPhase.SIZE_ASSESSMENT: f"""Extract DATA TYPES:
"{user_input}"

Return ONLY JSON:
{{
    "data_types": ["list", "of", "data", "types"],
    "data_sensitivity": "low|medium|high|critical"
}}""",
            
            ConversationPhase.DATA_SENSITIVITY: f"""Extract INFRASTRUCTURE info:
"{user_input}"

Return ONLY JSON:
{{
    "infrastructure": ["on-premise", "cloud", "hybrid"],
    "cloud_providers": ["aws", "azure", "gcp"] or [],
    "third_party_services": true or false
}}""",
            
            ConversationPhase.INFRASTRUCTURE: f"""Extract MATURITY level:
"{user_input}"

Return ONLY JSON:
{{
    "current_maturity": "none|basic|intermediate|advanced"
}}""",
            
            ConversationPhase.CURRENT_MATURITY: f"""Extract COMPLIANCE requirements:
"{user_input}"

Return ONLY JSON:
{{
    "compliance_requirements": ["HIPAA", "PCI-DSS", etc],
    "specific_concerns": ["concern1", "concern2"]
}}"""
        }
        
        # Default per fasi non gestite
        prompt = extraction_prompts.get(
            self.current_phase,
            f'Extract key information from: "{user_input}". Return as JSON.'
        )
        
        extraction_trace: Dict[str, Any] = {
            'phase': self.current_phase.value,
            'prompt': prompt,
            'llm_request': {
                'model': self.model_name,
                'temperature': 0.0,
                'max_tokens': 800,
                'messages': [
                    {"role": "system", "content": "You are a data extraction assistant. Return ONLY valid JSON, no other text."},
                    {"role": "user", "content": prompt}
                ]
            }
        }

        if self.verbose:
            print(f"\n📤 EXTRACTION PROMPT:")
            print("-"*80)
            print(prompt)
            print("-"*80)
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a data extraction assistant. Return ONLY valid JSON, no other text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=800
            )
            
            raw_content = response.choices[0].message.content
            content = raw_content.strip() if raw_content else ""
            reasoning = response.choices[0].message.reasoning_content

            extraction_trace['llm_response'] = {
                'content': raw_content,
                'reasoning_content': reasoning,
                'tokens': {
                    'input': response.usage.prompt_tokens,
                    'output': response.usage.completion_tokens,
                    'reasoning': (
                        response.usage.completion_tokens_details.reasoning_tokens
                        if hasattr(response.usage, 'completion_tokens_details') else None
                    )
                }
            }
            
            if self.verbose:
                print(f"\n📥 EXTRACTION RESPONSE:")
                print("-"*80)
                print(content)
                print("-"*80)
                
                if response.choices[0].message.reasoning_content:
                    print(f"\n🧠 REASONING:")
                    print(response.choices[0].message.reasoning_content[:200] + "...")
            
            # Parsa JSON
            # Rimuovi eventuali markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            extracted = json.loads(content)
            extraction_trace['parsed_extracted_info'] = extracted
            self._last_extraction_trace = extraction_trace
            return extracted
            
        except Exception as e:
            self._log(f"Error during extraction: {e}", "INFO")
            extraction_trace['error'] = str(e)
            self._last_extraction_trace = extraction_trace
            # Fallback: restituisci risposta grezza
            return {"raw_response": user_input, "extraction_failed": True}
    
    def _update_profile(self, extracted_info: Dict[str, Any]):
        """Aggiorna il profilo aziendale con le info estratte"""
        for key, value in extracted_info.items():
            if hasattr(self.company_profile, key) and value is not None:
                current = getattr(self.company_profile, key)
                if isinstance(current, list):
                    # Aggiungi a lista
                    if isinstance(value, list):
                        current.extend(value)
                    else:
                        current.append(value)
                else:
                    # Sovrascrivi valore
                    setattr(self.company_profile, key, value)
    
    def _should_query_nist_data(self) -> bool:
        """Decide se interrogare il database NIST in questo turn"""
        # Query NIST dopo aver raccolto info su data sensitivity o infrastructure
        return self.current_phase in [
            ConversationPhase.DATA_SENSITIVITY,
            ConversationPhase.INFRASTRUCTURE,
            ConversationPhase.CONTROL_IDENTIFICATION
        ]
    
    def _query_relevant_controls(self) -> Optional[Dict]:
        """
        Interroga database NIST per trovare controlli rilevanti.
        
        Decide automaticamente se usare ManualPandasAgent o Direct Methods.
        """
        # Costruisci query basata sul profilo
        if self.company_profile.industry in ['healthcare', 'health']:
            # Query specifica per healthcare
            self._log("Using Direct Method for healthcare baseline", "AGENT")
            
            # Usa direct method per healthcare
            controls = []
            # Identity & Access Management sono critici per healthcare
            result = self.direct_methods.find_controls_for_subcategory('PR.AC-1')
            if result:
                controls.append(result)
            
            return {
                'agent_type': 'direct_method',
                'method': 'find_controls_for_subcategory',
                'method_input': {'subcategory_id': 'PR.AC-1'},
                'results': controls
            }
        
        else:
            # Query dinamica con Pandas Agent
            self._log("Using ManualPandasAgent for dynamic query", "AGENT")
            
            # Costruisci domanda naturale
            question = self._build_dynamic_query()
            
            if self.verbose:
                print(f"\n🤖 PANDAS AGENT QUERY: {question}")

            pandas_log_file = os.path.join(
                self.pandas_logs_dir,
                f"pandas_log_turn_{self.turn_counter}_{self.current_phase.value}.json"
            )
            
            result = self.pandas_agent.query(
                df=self.datasets['csf_mapping'],
                question=question,
                max_iterations=3,
                log_to_json=pandas_log_file
            )

            pandas_log_data = None
            try:
                with open(pandas_log_file, 'r', encoding='utf-8') as f:
                    pandas_log_data = json.load(f)
            except Exception:
                pandas_log_data = None
            
            return {
                'agent_type': 'manual_pandas_agent',
                'question': question,
                'result': result,
                'pandas_log_file': pandas_log_file,
                'pandas_log': pandas_log_data
            }
    
    def _build_dynamic_query(self) -> str:
        """Costruisce una query dinamica basata sul profilo corrente"""
        # Semplice per ora
        if self.company_profile.data_sensitivity == 'high':
            return "Show me all subcategories related to data protection and access control"
        else:
            return "How many subcategories are in the Identify function?"
    
    def _determine_next_phase(self) -> ConversationPhase:
        """Determina la prossima fase del dialogo"""
        phase_sequence = [
            ConversationPhase.WELCOME,
            ConversationPhase.INDUSTRY_IDENTIFICATION,
            ConversationPhase.SIZE_ASSESSMENT,
            ConversationPhase.DATA_SENSITIVITY,
            ConversationPhase.INFRASTRUCTURE,
            ConversationPhase.CURRENT_MATURITY,
            ConversationPhase.CONTROL_IDENTIFICATION,
            ConversationPhase.COMPLETION
        ]
        
        current_index = phase_sequence.index(self.current_phase)
        if current_index < len(phase_sequence) - 1:
            return phase_sequence[current_index + 1]
        else:
            return ConversationPhase.COMPLETION
    
    def _generate_final_report(self) -> str:
        """Genera il report finale di compliance"""
        report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                     COMPLIANCE ASSESSMENT COMPLETE ✅                       ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 COMPANY PROFILE:
{json.dumps(self.company_profile.to_dict(), indent=2)}

💬 CONVERSATION TURNS: {len(self.conversation_history)}

📋 IDENTIFIED CONTROLS: {len(self.identified_controls)}

🎯 RECOMMENDATIONS:
Based on your profile, I recommend focusing on:
1. Access Control (PR.AC)
2. Data Security (PR.DS)
3. Identity Management (PR.AC.1-7)

📄 Full detailed report can be exported to PDF.

Thank you for using the NIST CSF 2.0 Compliance Tool!
"""
        return report
    
    def export_conversation_log(self, filename: Optional[str] = None):
        """Esporta log completo della conversazione"""
        if filename is None:
            filename = os.path.join(self.run_dir, "dialogue_log.json")
            
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'company_profile': self.company_profile.to_dict(),
            'conversation': [
                {
                    'turn': turn.turn_number,
                    'phase': turn.phase.value,
                    'question': turn.ai_question,
                    'response': turn.user_response,
                    'extracted': turn.extracted_info
                }
                for turn in self.conversation_history
            ],
            'identified_controls': self.identified_controls
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2)
        
        if self.verbose:
            print(f"\n✅ Conversation log exported to: {filename}")

    def export_detailed_log(self, filename: Optional[str] = None):
        """Esporta log dettagliato con prompt, reasoning, token e trace agent."""
        if filename is None:
            filename = os.path.join(self.run_dir, "dialogue_detailed_log.json")
            
        detailed_data = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'company_profile': self.company_profile.to_dict(),
            'turns_count': self.turn_counter,
            'conversation_summary': [
                {
                    'turn': turn.turn_number,
                    'phase': turn.phase.value,
                    'question': turn.ai_question,
                    'response': turn.user_response,
                    'extracted': turn.extracted_info,
                    'agent_used': turn.agent_used
                }
                for turn in self.conversation_history
            ],
            'detailed_turns': self.detailed_turn_logs,
            'identified_controls': self.identified_controls
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"\n✅ Detailed log exported to: {filename}")
