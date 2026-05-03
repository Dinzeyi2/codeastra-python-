"""
CodeAstraClient — full-featured async/sync HTTP client for the Codeastra API.

New in v1.6.0:
  - ThinkingTokens — data that thinks for itself without revealing itself
  - Thinking Executor — act on real data without the agent ever seeing it
  - OmegaTokens — unified token system with 7 paradigms
  - protect_text() — tokenize any raw text (emails, SSNs, cards in free text)
  - vault_resolve() — resolve a token to real value (executor only)
  - vault_resolve_batch() — resolve multiple tokens at once

New in v1.5.x:
  - BlindAgentMiddleware — drop-in wrapper for any agent
  - Smart Tokens — policy-bound tokens with semantic meaning
  - Blind RAG — semantic search on tokenized documents
  - K-anonymity — re-identification protection
  - Context-aware sensitivity — industry profiles

New in v1.1.0:
  - mode="cloud" / "onprem" / "hybrid"
  - zero_log=True — zero logging mode
  - Auto-register executor on init
"""
from __future__ import annotations

import re
import os
import json
import hmac
import socket
import hashlib
import asyncio
from pathlib import Path
from typing import Any, Optional, List, Dict

import httpx

TOKEN_RE    = re.compile(r'\[CVT:[A-Z]+:[A-F0-9]+\]')
CDT_RE      = re.compile(r'cdt_[A-Z]+_[bto]_[a-f0-9]+')
THT_RE      = re.compile(r'tht_[A-Z]+_[a-f0-9]+')
ANY_TOKEN   = re.compile(r'(\[CVT:[A-Z]+:[A-F0-9]+\]|cdt_[A-Z]+_[bto]_[a-f0-9]+|tht_[A-Z]+_[a-f0-9]+|tok_[A-Z]+_[a-f0-9]+)')

_DEFAULT_BASE   = "https://app.codeastra.dev"
_ONPREM_DEFAULT = "http://localhost:4000"


def _detect_environment() -> str:
    env_mode = os.environ.get("CODEASTRA_MODE", "").lower()
    if env_mode in ("cloud", "onprem", "hybrid"):
        return env_mode
    try:
        s = socket.create_connection(("localhost", 4000), timeout=1)
        s.close()
        return "onprem"
    except Exception:
        pass
    return "cloud"


def _get_base_url(mode: str, base_url: str = None) -> str:
    if base_url:
        return base_url.rstrip("/")
    if mode in ("onprem", "hybrid"):
        return os.environ.get("CODEASTRA_ONPREM_URL", _ONPREM_DEFAULT)
    return _DEFAULT_BASE


class CodeAstraClient:
    """
    Full-featured Codeastra client — v1.6.0.

    Quickstart:
        from codeastra import CodeAstraClient
        client = CodeAstraClient(api_key="sk-guard-xxx")

        # Protect any text — emails, SSNs, card numbers tokenized automatically
        result = client.protect_text("Call James at james@goldman.com SSN 234-56-7890")
        # → "Call James at [CVT:EMAIL:A1B2] [CVT:SSN:C3D4]"

        # ThinkingTokens — data that thinks for itself
        token = client.think_mint(
            real_value = "John Smith",
            data_type  = "patient",
            facts      = {"age": 67, "risk": "high", "diagnosis": "diabetes"},
        )
        matches = client.think_query("high risk patients over 65")

        # Thinking Executor — act without the agent seeing real data
        client.executor_register_integration("vapi", {...})
        client.executor_register_rule("call_high_risk", "vapi", "call", [...])
        client.executor_run_cohort("hospital_2024")
    """

    def __init__(
        self,
        api_key:      str   = None,
        base_url:     str   = None,
        agent_id:     str   = "sdk-agent",
        timeout:      float = 10.0,
        executor_url: str   = None,
        mode:         str   = "auto",
        zero_log:     bool  = False,
        onprem_dir:   str   = "./codeastra-onprem",
        verbose:      bool  = False,
    ):
        if not api_key:
            api_key = os.environ.get("CODEASTRA_API_KEY")
        if not api_key:
            api_key = self._auto_signup()

        if mode == "auto":
            mode = _detect_environment()

        self.api_key     = api_key
        self.agent_id    = agent_id
        self.mode        = mode
        self.zero_log    = zero_log
        self._verbose    = verbose
        self._timeout    = timeout
        self._onprem_dir = Path(onprem_dir)
        self.base_url    = _get_base_url(mode, base_url)

        self._headers = {
            "X-API-Key":    api_key,
            "Content-Type": "application/json",
        }
        if zero_log:
            self._headers["X-Zero-Log"] = "true"

        self._sync_client:  Optional[httpx.Client]      = None
        self._async_client: Optional[httpx.AsyncClient] = None

        if verbose:
            print(f"[CodeAstra] mode={mode} base={self.base_url} zero_log={zero_log}")

        if mode in ("onprem", "hybrid"):
            self._setup_onprem(mode)

        if executor_url:
            self._executor_url = executor_url
            try:
                self._post("/agent/executor", {
                    "execution_url": executor_url,
                    "action_type":   "*",
                    "agent_id":      agent_id,
                    "description":   f"Auto-registered by SDK agent {agent_id} ({mode})",
                })
                if verbose:
                    print(f"[CodeAstra] Executor auto-registered: {executor_url}")
            except Exception as e:
                if verbose:
                    print(f"[CodeAstra] Executor registration skipped: {e}")

    # ── Auto-signup ───────────────────────────────────────────────────────────

    def _auto_signup(self) -> str:
        creds_path = Path.home() / ".codeastra" / "credentials"
        if creds_path.exists():
            try:
                data = json.loads(creds_path.read_text())
                key  = data.get("api_key")
                if key:
                    return key
            except Exception:
                pass
        import uuid
        email    = os.environ.get("CODEASTRA_EMAIL",    f"user-{uuid.uuid4().hex[:8]}@codeastra.local")
        password = os.environ.get("CODEASTRA_PASSWORD", uuid.uuid4().hex)
        name     = os.environ.get("CODEASTRA_NAME",     f"SDK User {uuid.uuid4().hex[:6]}")
        try:
            r = httpx.post(f"{_DEFAULT_BASE}/auth/signup",
                           json={"name": name, "email": email, "password": password}, timeout=10)
            if r.is_success:
                data    = r.json()
                api_key = data.get("api_key")
                if api_key:
                    creds_path.parent.mkdir(parents=True, exist_ok=True)
                    creds_path.write_text(json.dumps(
                        {"api_key": api_key, "email": email, "password": password}))
                    print(f"[CodeAstra] Account created. Key saved to {creds_path}")
                    return api_key
        except Exception:
            pass
        raise ValueError(
            "No API key. Set CODEASTRA_API_KEY or pass api_key= "
            "or sign up at https://app.codeastra.dev"
        )

    # ── On-premise setup ──────────────────────────────────────────────────────

    def _setup_onprem(self, mode: str):
        setup_sh = self._onprem_dir / "setup.sh"
        if setup_sh.exists():
            if self._verbose:
                print(f"[CodeAstra] On-premise package at {self._onprem_dir}")
            return
        if self._verbose:
            print(f"[CodeAstra] Generating on-premise package...")
        try:
            resp  = self._post("/onprem/generate", {
                "deployment_mode": "docker", "llm_provider": "ollama",
                "llm_model": "llama3", "air_gapped": mode != "hybrid",
                "name": f"codeastra-{self.agent_id}",
            })
            files = resp.get("files", {})
            if files:
                self._onprem_dir.mkdir(parents=True, exist_ok=True)
                for filename, content in files.items():
                    fpath = self._onprem_dir / filename
                    fpath.write_text(content)
                if setup_sh.exists():
                    setup_sh.chmod(0o755)
                print(f"\n[CodeAstra] On-premise package ready: {self._onprem_dir}")
                print(f"  Run: cd {self._onprem_dir} && bash setup.sh\n")
        except Exception as e:
            if self._verbose:
                print(f"[CodeAstra] On-premise setup warning: {e} — falling back to cloud")
            self.base_url = _DEFAULT_BASE
            self.mode     = "cloud"

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _get_sync(self) -> httpx.Client:
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(headers=self._headers, timeout=self._timeout)
        return self._sync_client

    def _post(self, path: str, body: dict) -> dict:
        r = self._get_sync().post(f"{self.base_url}{path}", json=body)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str, params: dict = None) -> dict:
        r = self._get_sync().get(f"{self.base_url}{path}", params=params or {})
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str) -> dict:
        r = self._get_sync().delete(f"{self.base_url}{path}")
        r.raise_for_status()
        return r.json()

    def _get_async(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(headers=self._headers, timeout=self._timeout)
        return self._async_client

    async def _apost(self, path: str, body: dict) -> dict:
        r = await self._get_async().post(f"{self.base_url}{path}", json=body)
        r.raise_for_status()
        return r.json()

    async def _aget(self, path: str, params: dict = None) -> dict:
        r = await self._get_async().get(f"{self.base_url}{path}", params=params or {})
        r.raise_for_status()
        return r.json()

    # ══════════════════════════════════════════════════════════════════════════
    # PROTECT TEXT  (v1.6.0)
    # Tokenize any free-text string — emails, SSNs, card numbers, names.
    # This is what the agent sees instead of real data.
    # ══════════════════════════════════════════════════════════════════════════

    def protect_text(self, text: str, classification: str = "pii") -> str:
        """
        Tokenize any free-text string.
        Emails, SSNs, card numbers, names, phones — all replaced with tokens.
        Returns the protected text. Agent receives this. Never the original.

        Usage:
            safe = client.protect_text("Call James at james@goldman.com SSN 234-56-7890")
            # → "Call James at [CVT:EMAIL:A1B2] [CVT:SSN:C3D4]"
            agent.run(safe)
        """
        resp = self._post("/protect/text", {
            "text":           text,
            "classification": classification,
        })
        return resp.get("protected_text", text)

    def protect_text_full(self, text: str, classification: str = "pii") -> dict:
        """
        Tokenize text and return full response with entity list.

        Returns:
            {
                "protected_text": "...",
                "entities": [{"type": "EMAIL", "token": "[CVT:EMAIL:A1B2]", "original": "..."}],
                "count": 3
            }
        """
        return self._post("/protect/text", {
            "text":           text,
            "classification": classification,
        })

    async def aprotect_text(self, text: str, classification: str = "pii") -> str:
        """Async version of protect_text."""
        resp = await self._apost("/protect/text", {
            "text":           text,
            "classification": classification,
        })
        return resp.get("protected_text", text)

    async def aprotect_text_full(self, text: str, classification: str = "pii") -> dict:
        """Async version of protect_text_full."""
        return await self._apost("/protect/text", {
            "text":           text,
            "classification": classification,
        })

    # ══════════════════════════════════════════════════════════════════════════
    # VAULT RESOLVE  (v1.6.0)
    # Resolve tokens back to real values.
    # For authorized executors only — never call from agent context.
    # ══════════════════════════════════════════════════════════════════════════

    def vault_resolve(self, token: str) -> dict:
        """
        Resolve a token to its real value.
        For trusted executors only — never pass result to agent.

        Usage (in your executor endpoint only):
            result = client.vault_resolve("[CVT:EMAIL:A1B2]")
            if result["authorized"]:
                send_email(result["real_value"])  # real value only here
        """
        return self._post("/vault/resolve", {"token": token})

    def vault_resolve_batch(self, tokens: List[str]) -> dict:
        """
        Resolve multiple tokens at once.
        Returns {"resolved": {"token": "real_value", ...}}

        Usage:
            resolved = client.vault_resolve_batch(["[CVT:EMAIL:A1B2]", "[CVT:SSN:C3D4]"])
            emails = resolved["resolved"]["[CVT:EMAIL:A1B2]"]
        """
        return self._post("/vault/resolve-batch", {"tokens": tokens})

    async def avault_resolve(self, token: str) -> dict:
        """Async version of vault_resolve."""
        return await self._apost("/vault/resolve", {"token": token})

    async def avault_resolve_batch(self, tokens: List[str]) -> dict:
        """Async version of vault_resolve_batch."""
        return await self._apost("/vault/resolve-batch", {"tokens": tokens})

    # ══════════════════════════════════════════════════════════════════════════
    # THINKING TOKENS  (v1.6.0)
    # Data that thinks for itself without revealing itself.
    # Real value → vault. Facts → outside. Agent reasons on facts, never real value.
    # ══════════════════════════════════════════════════════════════════════════

    def think_mint(
        self,
        real_value:        str,
        data_type:         str,
        facts:             dict,
        cohort_id:         str  = None,
        signal_conditions: list = None,
        ttl_hours:         int  = 720,
    ) -> dict:
        """
        Mint a ThinkingToken. Real value goes to vault — never seen again.
        Facts written on the outside — safe for agent to reason on.

        Usage:
            token = client.think_mint(
                real_value = "John Smith | SSN:234-56-7890",
                data_type  = "patient",
                cohort_id  = "hospital_2024",
                facts      = {
                    "age":        67,
                    "risk":       "high",
                    "diagnosis":  "diabetes",
                    "region":     "atlanta",
                },
                signal_conditions = [
                    {"if": "risk == 'high'",  "signal": "needs_callback_today"},
                ],
            )
            # Returns: {"token_id": "tht_PATI_c7dd388e4f", ...}
            # Agent can query this token. Never sees "John Smith".
        """
        body = {
            "real_value": real_value,
            "data_type":  data_type,
            "facts":      facts,
            "ttl_hours":  ttl_hours,
        }
        if cohort_id:         body["cohort_id"]         = cohort_id
        if signal_conditions: body["signal_conditions"] = signal_conditions
        return self._post("/think/mint", body)

    def think_mint_batch(self, tokens: list) -> dict:
        """
        Mint up to 100 ThinkingTokens in one call.

        Usage:
            client.think_mint_batch([
                {"real_value": "John Smith", "data_type": "patient",
                 "cohort_id": "hospital_2024",
                 "facts": {"age": 67, "risk": "high"}},
                {"real_value": "Jane Doe", "data_type": "patient",
                 "cohort_id": "hospital_2024",
                 "facts": {"age": 45, "risk": "medium"}},
            ])
        """
        return self._post("/think/mint/batch", {"tokens": tokens})

    def think_query(
        self,
        query:           str,
        cohort_id:       str  = None,
        include_reasons: bool = False,
        top_k:           int  = None,
    ) -> dict:
        """
        Natural language query — tokens self-evaluate without revealing real data.
        50,000 tokens evaluated in seconds. Agent gets token IDs, never real values.

        Usage:
            results = client.think_query(
                "high risk diabetic patients over 65",
                cohort_id = "hospital_2024",
            )
            # Returns token IDs — pass to executor to act on real patients
            for match in results["matched_tokens"]:
                print(match["token_id"])  # tht_PATI_c7dd388e4f
                # Never sees real name, SSN, phone
        """
        body = {
            "query":           query,
            "include_reasons": include_reasons,
        }
        if cohort_id: body["cohort_id"] = cohort_id
        if top_k:     body["top_k"]     = top_k
        return self._post("/think/query", body)

    def think_query_cohort(self, query: str, cohort_id: str, **kwargs) -> dict:
        """Query an entire cohort. Alias for think_query with cohort_id required."""
        return self.think_query(query, cohort_id=cohort_id, **kwargs)

    def think_signal(self, cohort_id: str) -> dict:
        """
        Get tokens that are proactively signaling — no query needed.
        Tokens raise their hand when their own conditions are met.

        Usage:
            signals = client.think_signal("hospital_2024")
            # 201 patients signaled "needs_callback_today"
            # Agent never queried them. They came forward.
            for s in signals["signals"]:
                print(s["token_id"], s["signal"])
                # Pass to executor to call real patients
        """
        return self._post("/think/signal", {"cohort_id": cohort_id})

    def think_get(self, token_id: str) -> dict:
        """Get ThinkingToken metadata. Safe for agent — never returns real value."""
        return self._get(f"/think/{token_id}")

    def think_memory(self, token_id: str) -> dict:
        """
        What this token has learned from queries.
        Shows match patterns, hit counts, evolution version.
        """
        return self._get(f"/think/{token_id}/memory")

    def think_evolve(self, token_id: str) -> dict:
        """Trigger manual evolution cycle for a token."""
        return self._post(f"/think/{token_id}/evolve", {})

    def think_audit(self, token_id: str) -> dict:
        """Full query history for a ThinkingToken."""
        return self._get(f"/think/{token_id}/audit")

    def think_revoke(self, token_id: str) -> dict:
        """Revoke a ThinkingToken permanently."""
        return self._delete(f"/think/{token_id}")

    def think_stats(self) -> dict:
        """System-wide ThinkingToken statistics."""
        return self._get("/think/stats")

    def think_ollama_status(self) -> dict:
        """Check Ollama health for complex query fallback."""
        return self._get("/think/ollama/status")

    # Async ThinkingToken methods
    async def athink_mint(self, real_value: str, data_type: str, facts: dict,
                          cohort_id: str = None, signal_conditions: list = None,
                          ttl_hours: int = 720) -> dict:
        """Async version of think_mint."""
        body = {"real_value": real_value, "data_type": data_type,
                "facts": facts, "ttl_hours": ttl_hours}
        if cohort_id:         body["cohort_id"]         = cohort_id
        if signal_conditions: body["signal_conditions"] = signal_conditions
        return await self._apost("/think/mint", body)

    async def athink_mint_batch(self, tokens: list) -> dict:
        """Async version of think_mint_batch."""
        return await self._apost("/think/mint/batch", {"tokens": tokens})

    async def athink_query(self, query: str, cohort_id: str = None,
                           include_reasons: bool = False) -> dict:
        """Async version of think_query."""
        body = {"query": query, "include_reasons": include_reasons}
        if cohort_id: body["cohort_id"] = cohort_id
        return await self._apost("/think/query", body)

    async def athink_signal(self, cohort_id: str) -> dict:
        """Async version of think_signal."""
        return await self._apost("/think/signal", {"cohort_id": cohort_id})

    # ══════════════════════════════════════════════════════════════════════════
    # THINKING EXECUTOR  (v1.6.0)
    # Act on real data without the agent ever seeing it.
    # Agent → token IDs → Executor → opens vault → fires integration → locks vault
    # ══════════════════════════════════════════════════════════════════════════

    def executor_register_integration(
        self,
        name:   str,
        config: dict,
    ) -> dict:
        """
        Register an integration with the Thinking Executor.

        Supported integrations:
            vapi    — AI voice calls
            twilio  — SMS and WhatsApp
            sendgrid — Email
            stripe  — Payments
            epic    — Epic EHR
            docusign — Document signing
            salesforce — CRM
            hubspot  — CRM
            webhook  — Any HTTPS endpoint

        Usage:
            client.executor_register_integration("vapi", {
                "api_key":         "your-vapi-key",
                "phone_number_id": "your-phone-id",
                "assistant_id":    "your-assistant-id",
            })

            client.executor_register_integration("twilio", {
                "account_sid": "AC...",
                "auth_token":  "...",
                "from_number": "+14045550100",
            })
        """
        return self._post("/executor/integrations", {
            "name":   name,
            "config": config,
        })

    def executor_list_integrations(self) -> dict:
        """List all registered integrations."""
        return self._get("/executor/integrations")

    def executor_delete_integration(self, integration_id: str) -> dict:
        """Remove an integration."""
        return self._delete(f"/executor/integrations/{integration_id}")

    def executor_register_rule(
        self,
        name:            str,
        integration:     str,
        action_type:     str,
        conditions:      list,
        action_template: dict = None,
        priority:        int  = 5,
    ) -> dict:
        """
        Add a decision rule to the Thinking Executor.
        The executor reads facts and decides which action to take.
        No per-token configuration needed — rules apply to the whole cohort.

        Usage:
            # High risk → Vapi calls them immediately
            client.executor_register_rule(
                name        = "call_high_risk",
                integration = "vapi",
                action_type = "call",
                conditions  = [{"field": "risk", "op": "eq", "value": "high"}],
                action_template = {"assistant_message": "Hello {name}, this is Dr. Johnson..."},
                priority    = 10,
            )

            # Medium risk → SMS
            client.executor_register_rule(
                name        = "sms_medium_risk",
                integration = "twilio",
                action_type = "sms",
                conditions  = [{"field": "risk", "op": "eq", "value": "medium"}],
                action_template = {"body": "Hi {name}, please call us at 404-555-0100."},
                priority    = 5,
            )
        """
        body = {
            "name":        name,
            "integration": integration,
            "action_type": action_type,
            "conditions":  conditions,
            "priority":    priority,
        }
        if action_template:
            body["action_template"] = action_template
        return self._post("/executor/rules", body)

    def executor_list_rules(self) -> dict:
        """List all decision rules."""
        return self._get("/executor/rules")

    def executor_delete_rule(self, rule_id: str) -> dict:
        """Remove a decision rule."""
        return self._delete(f"/executor/rules/{rule_id}")

    def executor_run(self, token_id: str, dry_run: bool = False) -> dict:
        """
        Run the Thinking Executor on a single token.
        Executor opens vault, reads real value, fires matching integration.
        Real value never returned to agent.

        Usage:
            result = client.executor_run("tht_PATI_c7dd388e4f")
            # Vapi called patient with their real name + phone
            # result["real_value_seen_by_agent"] == False
        """
        return self._post("/executor/run", {
            "token_id": token_id,
            "dry_run":  dry_run,
        })

    def executor_run_cohort(
        self,
        cohort_id: str,
        dry_run:   bool = False,
        limit:     int  = None,
    ) -> dict:
        """
        Run the Thinking Executor on an entire cohort.
        One call. 50,000 patients processed.
        Vapi calls high risk. Twilio SMS medium risk. SendGrid emails low risk.
        Agent saw zero real values across the entire pipeline.

        Usage:
            result = client.executor_run_cohort("hospital_2024")
            print(result["total_actions"])     # 8347
            print(result["total_succeeded"])   # 8289
            print(result["real_value_seen_by_agent"])  # false
        """
        body = {"cohort_id": cohort_id, "dry_run": dry_run}
        if limit: body["limit"] = limit
        return self._post("/executor/run/cohort", body)

    def executor_log(self, limit: int = 50, token_id: str = None) -> dict:
        """Full action audit log from the executor."""
        params = {"limit": limit}
        if token_id: params["token_id"] = token_id
        return self._get("/executor/log", params)

    def executor_learning(self) -> dict:
        """What the executor has learned — which rules fire most, pattern evolution."""
        return self._get("/executor/learning")

    def executor_status(self) -> dict:
        """Integration health check — all registered integrations tested."""
        return self._get("/executor/status")

    def executor_supported(self) -> dict:
        """List all supported integrations with their required config fields."""
        return self._get("/executor/supported")

    # Async Executor methods
    async def aexecutor_run(self, token_id: str, dry_run: bool = False) -> dict:
        """Async version of executor_run."""
        return await self._apost("/executor/run", {"token_id": token_id, "dry_run": dry_run})

    async def aexecutor_run_cohort(self, cohort_id: str, dry_run: bool = False) -> dict:
        """Async version of executor_run_cohort."""
        return await self._apost("/executor/run/cohort", {"cohort_id": cohort_id, "dry_run": dry_run})

    # ══════════════════════════════════════════════════════════════════════════
    # OMEGA TOKENS  (v1.6.0)
    # Unified token system with 7 paradigms.
    # Combines BlindAgent + ThinkingTokens + SmartTokens + Executor in one.
    # ══════════════════════════════════════════════════════════════════════════

    def omega_mint(
        self,
        real_value:        str,
        data_type:         str,
        facts:             dict = None,
        allowed_actions:   list = None,
        allowed_targets:   list = None,
        signal_conditions: list = None,
        pipeline_id:       str  = None,
        ttl_hours:         int  = 720,
        max_uses:          int  = None,
    ) -> dict:
        """
        Mint an OmegaToken — the most powerful token in Codeastra.
        Combines ThinkingTokens + SmartTokens + 9-gate policy in one.

        Usage:
            token = client.omega_mint(
                real_value      = "James Dimon | james.dimon@jpmorgan.com",
                data_type       = "client",
                facts           = {"tier": "enterprise", "aum": "high", "risk": "low"},
                allowed_actions = ["send_report", "schedule_meeting"],
                allowed_targets = ["https://jpmorgan.com/portal"],
                signal_conditions = [{"if": "aum == 'high'", "signal": "priority_client"}],
            )
        """
        body = {
            "real_value": real_value,
            "data_type":  data_type,
            "ttl_hours":  ttl_hours,
        }
        if facts:             body["facts"]             = facts
        if allowed_actions:   body["allowed_actions"]   = allowed_actions
        if allowed_targets:   body["allowed_targets"]   = allowed_targets
        if signal_conditions: body["signal_conditions"] = signal_conditions
        if pipeline_id:       body["pipeline_id"]       = pipeline_id
        if max_uses:          body["max_uses"]           = max_uses
        return self._post("/omega/mint", body)

    def omega_mint_batch(self, tokens: list) -> dict:
        """Mint up to 100 OmegaTokens in one call."""
        return self._post("/omega/mint/batch", {"tokens": tokens})

    def omega_get(self, token_id: str) -> dict:
        """Get OmegaToken metadata. Safe for agent."""
        return self._get(f"/omega/{token_id}")

    def omega_execute(
        self,
        token_id:    str,
        action_type: str,
        target_url:  str = None,
        field_name:  str = None,
    ) -> dict:
        """9-gate policy execution on an OmegaToken."""
        return self._post(f"/omega/{token_id}/execute", {
            "action_type": action_type,
            "target_url":  target_url,
            "field_name":  field_name,
        })

    def omega_proof(self, token_id: str, fact: str) -> dict:
        """Prove a fact about real value without revealing it."""
        return self._get(f"/omega/{token_id}/proof", {"fact": fact})

    def omega_audit(self, token_id: str) -> dict:
        """Full audit trail for an OmegaToken."""
        return self._get(f"/omega/{token_id}/audit")

    def omega_revoke(self, token_id: str) -> dict:
        """Revoke an OmegaToken."""
        return self._delete(f"/omega/{token_id}")

    # ══════════════════════════════════════════════════════════════════════════
    # EXISTING API — preserved from v1.5.2
    # ══════════════════════════════════════════════════════════════════════════

    def tokenize(self, data: dict, classification: str = "pii", ttl_hours: int = 24) -> dict:
        resp = self._post("/vault/store", {
            "data": data, "agent_id": self.agent_id,
            "classification": classification, "ttl_hours": ttl_hours,
        })
        return resp.get("tokens", {})

    def execute(self, action_type: str, params: dict, pipeline_id: str = None) -> dict:
        body = {"agent_id": self.agent_id, "action_type": action_type, "params": params}
        if pipeline_id:
            body["pipeline_id"] = pipeline_id
            return self._post("/pipeline/action", body)
        return self._post("/agent/action", body)

    def grant(self, receiving_agent: str, tokens: list, allowed_actions: list = [],
              pipeline_id: str = None, purpose: str = None) -> dict:
        return self._post("/vault/grant", {
            "granting_agent": self.agent_id, "receiving_agent": receiving_agent,
            "tokens": tokens, "allowed_actions": allowed_actions,
            "pipeline_id": pipeline_id, "purpose": purpose,
        })

    def audit(self, pipeline_id: str = None, token: str = None) -> list:
        params = {}
        if pipeline_id: params["pipeline_id"] = pipeline_id
        if token:       params["token"]       = token
        return self._get("/pipeline/audit", params).get("audit", [])

    def stats(self) -> dict:
        return self._get("/vault/stats")

    async def atokenize(self, data: dict, classification: str = "pii", ttl_hours: int = 24) -> dict:
        resp = await self._apost("/vault/store", {
            "data": data, "agent_id": self.agent_id,
            "classification": classification, "ttl_hours": ttl_hours,
        })
        return resp.get("tokens", {})

    async def aexecute(self, action_type: str, params: dict, pipeline_id: str = None) -> dict:
        body = {"agent_id": self.agent_id, "action_type": action_type, "params": params}
        if pipeline_id:
            body["pipeline_id"] = pipeline_id
            return await self._apost("/pipeline/action", body)
        return await self._apost("/agent/action", body)

    async def agrant(self, receiving_agent: str, tokens: list,
                     allowed_actions: list = [], pipeline_id: str = None) -> dict:
        return await self._apost("/vault/grant", {
            "granting_agent": self.agent_id, "receiving_agent": receiving_agent,
            "tokens": tokens, "allowed_actions": allowed_actions, "pipeline_id": pipeline_id,
        })

    # ── Smart Tokens (v1.5.x) ─────────────────────────────────────────────────

    def smart_tokenize(self, real_value: str, data_type: str, allowed_actions: list = [],
                       allowed_targets: list = [], allowed_fields: list = [],
                       max_uses: int = 1, ttl_seconds: int = 86400,
                       semantic_label: str = None) -> dict:
        return self._post("/vault/smart-token", {
            "real_value": real_value, "data_type": data_type, "agent_id": self.agent_id,
            "allowed_actions": allowed_actions, "allowed_targets": allowed_targets,
            "allowed_fields": allowed_fields, "max_uses": max_uses,
            "ttl_seconds": ttl_seconds, "semantic_label": semantic_label,
        })

    def smart_tokenize_batch(self, tokens: list) -> list:
        return self._post("/vault/smart-token/batch", {
            "agent_id": self.agent_id, "tokens": tokens,
        }).get("tokens", [])

    def smart_token_info(self, token_id: str) -> dict:
        return self._get(f"/vault/smart-token/{token_id}")

    def smart_token_execute(self, token_id: str, action_type: str = None,
                            target_url: str = None, field_name: str = None) -> dict:
        return self._post("/vault/smart-token/execute", {
            "token_id": token_id, "action_type": action_type,
            "target_url": target_url, "field_name": field_name, "agent_id": self.agent_id,
        })

    def smart_token_revoke(self, token_id: str, reason: str = "manual") -> dict:
        try:
            return self._get(f"/vault/smart-token/{token_id}/revoke")
        except Exception:
            return self._post(f"/vault/smart-token/{token_id}/revoke", {"reason": reason})

    def smart_token_audit(self, token_id: str) -> list:
        return self._get(f"/vault/smart-token/{token_id}/audit").get("audit", [])

    def smart_token_types(self) -> list:
        return self._get("/vault/smart-token-types").get("types", [])

    async def asmart_tokenize(self, real_value: str, data_type: str,
                              allowed_actions: list = [], allowed_targets: list = [],
                              allowed_fields: list = [], max_uses: int = 1,
                              ttl_seconds: int = 86400) -> dict:
        return await self._apost("/vault/smart-token", {
            "real_value": real_value, "data_type": data_type, "agent_id": self.agent_id,
            "allowed_actions": allowed_actions, "allowed_targets": allowed_targets,
            "allowed_fields": allowed_fields, "max_uses": max_uses, "ttl_seconds": ttl_seconds,
        })

    async def asmart_token_execute(self, token_id: str, action_type: str = None,
                                   target_url: str = None, field_name: str = None) -> dict:
        return await self._apost("/vault/smart-token/execute", {
            "token_id": token_id, "action_type": action_type,
            "target_url": target_url, "field_name": field_name, "agent_id": self.agent_id,
        })

    # ── Blind RAG (v1.5.x) ────────────────────────────────────────────────────

    def rag_ingest(self, content: dict, doc_type: str, title: str = None,
                   source: str = None, classification: str = "pii") -> dict:
        return self._post("/rag/ingest", {
            "content": content, "doc_type": doc_type, "agent_id": self.agent_id,
            "title": title, "source": source, "classification": classification,
        })

    def rag_ingest_batch(self, documents: list) -> dict:
        return self._post("/rag/ingest/batch", {"agent_id": self.agent_id, "documents": documents})

    def rag_search(self, query: str, doc_type: str = None,
                   top_k: int = 5, min_score: float = 0.3) -> dict:
        body = {"query": query, "top_k": top_k, "min_score": min_score}
        if doc_type: body["doc_type"] = doc_type
        return self._post("/rag/search", body)

    def rag_delete(self, doc_id: str) -> dict:
        return self._delete(f"/rag/document/{doc_id}")

    def rag_stats(self) -> dict:
        return self._get("/rag/stats")

    async def arag_ingest(self, content: dict, doc_type: str,
                          title: str = None, classification: str = "pii") -> dict:
        return await self._apost("/rag/ingest", {
            "content": content, "doc_type": doc_type,
            "agent_id": self.agent_id, "title": title, "classification": classification,
        })

    async def arag_search(self, query: str, doc_type: str = None,
                          top_k: int = 5, min_score: float = 0.3) -> dict:
        body = {"query": query, "top_k": top_k, "min_score": min_score}
        if doc_type: body["doc_type"] = doc_type
        return await self._apost("/rag/search", body)

    # ── Policy (v1.5.x) ───────────────────────────────────────────────────────

    def register_sensitive_type(self, fields: list, prefixes: list = [],
                                 doc_types: list = []) -> dict:
        return self._post("/policy/sensitivity/fields", {
            "fields": fields, "prefixes": prefixes, "doc_types": doc_types,
        })

    def set_sensitivity_policy(self, sensitive_fields: list = None,
                               sensitive_prefixes: list = None,
                               sensitive_doc_types: list = None,
                               field_classifications: dict = None,
                               strict_mode: bool = None) -> dict:
        body = {}
        if sensitive_fields      is not None: body["sensitive_fields"]      = sensitive_fields
        if sensitive_prefixes    is not None: body["sensitive_prefixes"]    = sensitive_prefixes
        if sensitive_doc_types   is not None: body["sensitive_doc_types"]   = sensitive_doc_types
        if field_classifications is not None: body["field_classifications"] = field_classifications
        if strict_mode           is not None: body["strict_mode"]           = strict_mode
        return self._post("/policy/sensitivity", body)

    def get_sensitivity_policy(self) -> dict:
        return self._get("/policy/sensitivity")

    def test_sensitivity(self, content: dict, field_policy: dict = {},
                         sensitive_fields: list = [], tokenize_all: bool = False) -> dict:
        return self._post("/policy/sensitivity/test", {
            "content": content, "field_policy": field_policy,
            "sensitive_fields": sensitive_fields, "tokenize_all": tokenize_all,
        })

    def set_context(self, industry: str = None, data_scope: str = None,
                    classification_level: str = None, extra_sensitive_fields: list = [],
                    safe_fields: list = [], strict_mode: bool = False) -> dict:
        return self._post("/policy/context", {
            "industry": industry, "data_scope": data_scope,
            "classification_level": classification_level,
            "extra_sensitive_fields": extra_sensitive_fields,
            "safe_fields": safe_fields, "context_strict_mode": strict_mode,
        })

    def set_anonymity(self, k_minimum: int = 5, suppress_singleton: bool = True,
                      auto_bucket: bool = True, detect_narrowing: bool = True,
                      quasi_identifiers: list = None) -> dict:
        body = {"k_minimum": k_minimum, "suppress_singleton": suppress_singleton,
                "auto_bucket": auto_bucket, "detect_narrowing": detect_narrowing}
        if quasi_identifiers is not None:
            body["quasi_identifiers"] = quasi_identifiers
        return self._post("/policy/anonymity", body)

    def test_context(self, content: dict, context: dict, field_policy: dict = {}) -> dict:
        return self._post("/policy/context/test", {
            "content": content, "context": context, "field_policy": field_policy,
        })

    def smart_ingest(self, content: dict, doc_type: str, field_policy: dict = {},
                     sensitive_fields: list = [], tokenize_all: bool = False,
                     title: str = None, classification: str = "pii") -> dict:
        return self._post("/rag/ingest", {
            "content": content, "doc_type": doc_type, "agent_id": self.agent_id,
            "title": title, "classification": classification,
            "field_policy": field_policy, "sensitive_fields": sensitive_fields,
            "tokenize_all": tokenize_all,
        })

    def smart_ingest_with_context(self, content: dict, doc_type: str,
                                   context: dict = {}, field_policy: dict = {},
                                   sensitive_fields: list = [], tokenize_all: bool = False,
                                   title: str = None) -> dict:
        return self._post("/rag/ingest", {
            "content": content, "doc_type": doc_type, "agent_id": self.agent_id,
            "title": title, "context": context, "field_policy": field_policy,
            "sensitive_fields": sensitive_fields, "tokenize_all": tokenize_all,
        })

    # ── Audit (v1.5.x) ────────────────────────────────────────────────────────

    def verify_audit(self) -> dict:
        try:
            return self._get("/audit/secure/verify")
        except Exception as e:
            return {"verified": False, "error": str(e)}

    def export_audit(self, output_path: str = "audit_report.json") -> str:
        try:
            data = self._get("/audit/secure/export")
            Path(output_path).write_text(json.dumps(data, indent=2))
            return output_path
        except Exception as e:
            return str(e)

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def extract_tokens(obj: Any) -> list:
        """Extract all tokens (CVT, CDT, THT, tok_) from any text or object."""
        text = json.dumps(obj) if not isinstance(obj, str) else obj
        return ANY_TOKEN.findall(text)

    @staticmethod
    def contains_token(val: Any) -> bool:
        """Check if a value contains any Codeastra token."""
        text = json.dumps(val) if not isinstance(val, str) else str(val)
        return bool(ANY_TOKEN.search(text))

    @staticmethod
    def is_token(val: str) -> bool:
        """Check if a string is exactly a Codeastra token."""
        return bool(ANY_TOKEN.fullmatch(val.strip()))

    @staticmethod
    def verify_executor_call(payload: str, signature: str, secret: str) -> bool:
        """Verify an incoming executor call is genuinely from Codeastra."""
        expected = "sha256=" + hmac.new(
            secret.encode(),
            payload.encode() if isinstance(payload, str) else payload,
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    def set_zero_log(self, enabled: bool = True):
        self.zero_log = enabled
        if enabled:
            self._headers["X-Zero-Log"] = "true"
        else:
            self._headers.pop("X-Zero-Log", None)
        self._sync_client  = None
        self._async_client = None

    def info(self) -> dict:
        return {
            "version":  "1.6.0",
            "mode":     self.mode,
            "base_url": self.base_url,
            "agent_id": self.agent_id,
            "zero_log": self.zero_log,
        }

    def close(self):
        if self._sync_client:  self._sync_client.close()

    async def aclose(self):
        if self._async_client: await self._async_client.aclose()

    def __enter__(self):  return self
    def __exit__(self, *_): self.close()
    async def __aenter__(self): return self
    async def __aexit__(self, *_): await self.aclose()

    def __repr__(self):
        return f"CodeAstraClient(v1.6.0, mode={self.mode!r}, agent_id={self.agent_id!r})"
