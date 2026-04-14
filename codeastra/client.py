"""
CodeAstraClient — full-featured async/sync HTTP client for the Codeastra API.

New in v1.1.0:
  - mode="cloud"   — default, uses app.codeastra.dev
  - mode="onprem"  — pulls deployment package, runs vault locally
  - mode="hybrid"  — local vault + cloud LLM (best for enterprise)
  - zero_log=True  — zero logging mode, max privacy
  - Auto-register executor on init
  - Auto-detect environment
  - Auto-generate on-premise package
  - HMAC verification of executor calls
  - Tamper-proof audit verification
  - Auto-signup on first use
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
from typing import Any, Optional

import httpx

TOKEN_RE = re.compile(r'\[CVT:[A-Z]+:[A-F0-9]+\]')

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
    Full-featured Codeastra client.

    Modes:
        cloud   — default. Uses app.codeastra.dev
        onprem  — local vault. Auto-generates deployment package on first use.
        hybrid  — local vault + cloud LLM. Best for enterprise.

    Usage:
        # Cloud (default — zero config)
        client = CodeAstraClient(api_key="sk-guard-xxx")

        # On-premise (auto-generates docker-compose + setup.sh)
        client = CodeAstraClient(api_key="sk-guard-xxx", mode="onprem")

        # Hybrid (local vault, cloud LLM)
        client = CodeAstraClient(api_key="sk-guard-xxx", mode="hybrid")

        # Zero logging
        client = CodeAstraClient(api_key="sk-guard-xxx", zero_log=True)

        # With executor auto-registered
        client = CodeAstraClient(api_key="sk-guard-xxx",
                                  executor_url="https://your-app.com/execute")

        # No API key — auto-signup
        client = CodeAstraClient()
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
        # Auto-signup if no API key
        if not api_key:
            api_key = os.environ.get("CODEASTRA_API_KEY")
        if not api_key:
            api_key = self._auto_signup()

        # Auto-detect mode
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

        # On-premise: auto-generate deployment package
        if mode in ("onprem", "hybrid"):
            self._setup_onprem(mode)

        # Auto-register executor if provided
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
        """Auto-create account on first use. Saves key to ~/.codeastra/credentials."""
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
            r = httpx.post(f"{_DEFAULT_BASE}/auth/signup", json={
                "name": name, "email": email, "password": password,
            }, timeout=10)
            if r.is_success:
                data    = r.json()
                api_key = data.get("api_key")
                if api_key:
                    creds_path.parent.mkdir(parents=True, exist_ok=True)
                    creds_path.write_text(json.dumps({
                        "api_key": api_key, "email": email, "password": password,
                    }))
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
        """Auto-generate on-premise deployment package if not already present."""
        setup_sh = self._onprem_dir / "setup.sh"
        if setup_sh.exists():
            if self._verbose:
                print(f"[CodeAstra] On-premise package at {self._onprem_dir}")
            return

        if self._verbose:
            print(f"[CodeAstra] Generating on-premise package...")

        try:
            resp = self._post("/onprem/generate", {
                "deployment_mode": "docker",
                "llm_provider":    "ollama",
                "llm_model":       "llama3",
                "air_gapped":      mode != "hybrid",
                "name":            f"codeastra-{self.agent_id}",
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

    # ── HMAC verification ─────────────────────────────────────────────────────

    @staticmethod
    def verify_executor_call(payload: str, signature: str, secret: str) -> bool:
        """
        Verify an incoming executor call is genuinely from Codeastra.
        Use in your executor endpoint to reject forged requests.

        Usage:
            @app.post("/execute")
            def execute(request):
                if not CodeAstraClient.verify_executor_call(
                    request.body, request.headers["X-Codeastra-Signature"], YOUR_SECRET
                ):
                    raise HTTPException(401)
        """
        expected = "sha256=" + hmac.new(
            secret.encode(),
            payload.encode() if isinstance(payload, str) else payload,
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    # ── Audit verification ────────────────────────────────────────────────────

    def verify_audit(self) -> dict:
        """Verify tamper-proof audit chain integrity."""
        try:
            return self._get("/audit/secure/verify")
        except Exception as e:
            return {"verified": False, "error": str(e)}

    def export_audit(self, output_path: str = "audit_report.json") -> str:
        """Export full compliance audit report."""
        try:
            data = self._get("/audit/secure/export")
            Path(output_path).write_text(json.dumps(data, indent=2))
            return output_path
        except Exception as e:
            return str(e)

    # ── Zero-log mode ─────────────────────────────────────────────────────────

    def set_zero_log(self, enabled: bool = True):
        """Enable/disable zero-logging mode."""
        self.zero_log = enabled
        if enabled:
            self._headers["X-Zero-Log"] = "true"
        else:
            self._headers.pop("X-Zero-Log", None)
        self._sync_client  = None
        self._async_client = None

    # ── sync helpers ──────────────────────────────────────────────────────────

    def _get_sync(self) -> httpx.Client:
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(
                headers=self._headers, timeout=self._timeout)
        return self._sync_client

    def _post(self, path: str, body: dict) -> dict:
        r = self._get_sync().post(f"{self.base_url}{path}", json=body)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str, params: dict = None) -> dict:
        r = self._get_sync().get(f"{self.base_url}{path}", params=params or {})
        r.raise_for_status()
        return r.json()

    # ── async helpers ─────────────────────────────────────────────────────────

    def _get_async(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                headers=self._headers, timeout=self._timeout)
        return self._async_client

    async def _apost(self, path: str, body: dict) -> dict:
        r = await self._get_async().post(f"{self.base_url}{path}", json=body)
        r.raise_for_status()
        return r.json()

    async def _aget(self, path: str, params: dict = None) -> dict:
        r = await self._get_async().get(
            f"{self.base_url}{path}", params=params or {})
        r.raise_for_status()
        return r.json()

    # ── public sync API ───────────────────────────────────────────────────────

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

    # ── public async API ──────────────────────────────────────────────────────

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

    # ── utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def extract_tokens(obj: Any) -> list:
        text = json.dumps(obj) if not isinstance(obj, str) else obj
        return TOKEN_RE.findall(text)

    @staticmethod
    def contains_token(val: Any) -> bool:
        text = json.dumps(val) if not isinstance(val, str) else str(val)
        return bool(TOKEN_RE.search(text))

    @staticmethod
    def is_token(val: str) -> bool:
        return bool(TOKEN_RE.fullmatch(val.strip()))


    # ── smart tokens (v4.2) ───────────────────────────────────────────────────

    def smart_tokenize(
        self,
        real_value:      str,
        data_type:       str,
        allowed_actions: list = [],
        allowed_targets: list = [],
        allowed_fields:  list = [],
        max_uses:        int  = 1,
        ttl_seconds:     int  = 86400,
        semantic_label:  str  = None,
    ) -> dict:
        """
        Mint a smart token — policy-bound and semantically meaningful.

        The agent receives meaning (what the data is, where it can go).
        The real value is vault-protected forever.
        The trusted executor reveals it only at the last mile.

        Usage:
            token = client.smart_tokenize(
                real_value      = "John Smith",
                data_type       = "patient_name",
                allowed_actions = ["fill_form"],
                allowed_fields  = ["first_name"],
                max_uses        = 1,
                ttl_seconds     = 30,
            )
            # → {"token_id": "tok_PATI_a1b2c3", "data_type": "patient_name", ...}
            # Agent gets this. Never sees "John Smith".
        """
        return self._post("/vault/smart-token", {
            "real_value":      real_value,
            "data_type":       data_type,
            "agent_id":        self.agent_id,
            "allowed_actions": allowed_actions,
            "allowed_targets": allowed_targets,
            "allowed_fields":  allowed_fields,
            "max_uses":        max_uses,
            "ttl_seconds":     ttl_seconds,
            "semantic_label":  semantic_label,
        })

    def smart_tokenize_batch(self, tokens: list) -> list:
        """Mint multiple smart tokens in one call."""
        resp = self._post("/vault/smart-token/batch", {
            "agent_id": self.agent_id,
            "tokens":   tokens,
        })
        return resp.get("tokens", [])

    def smart_token_info(self, token_id: str) -> dict:
        """Get smart token metadata. Safe for agent — never returns real value."""
        return self._get(f"/vault/smart-token/{token_id}")

    def smart_token_execute(
        self,
        token_id:    str,
        action_type: str = None,
        target_url:  str = None,
        field_name:  str = None,
    ) -> dict:
        """
        Policy-gated JIT reveal. Called by trusted executor — NEVER by agent.

        Runs all 5 policy gates. If all pass, returns real value.
        Token auto-revokes after max_uses reached.

        Usage (in your executor endpoint):
            result = client.smart_token_execute(
                token_id    = "tok_PATI_a1b2c3",
                action_type = "fill_form",
                target_url  = "https://hospital.com/intake",
                field_name  = "first_name",
            )
            if result["authorized"]:
                form.fill(field_name, result["real_value"])
                # real value used here — agent never saw it
        """
        return self._post("/vault/smart-token/execute", {
            "token_id":    token_id,
            "action_type": action_type,
            "target_url":  target_url,
            "field_name":  field_name,
            "agent_id":    self.agent_id,
        })

    def smart_token_revoke(self, token_id: str, reason: str = "manual") -> dict:
        """Immediately revoke a smart token."""
        try:
            return self._get(f"/vault/smart-token/{token_id}/revoke")
        except Exception:
            return self._post(f"/vault/smart-token/{token_id}/revoke", {"reason": reason})

    def smart_token_audit(self, token_id: str) -> list:
        """Full reveal audit trail for a token."""
        return self._get(f"/vault/smart-token/{token_id}/audit").get("audit", [])

    def smart_token_types(self) -> list:
        """List all supported data types for smart tokens."""
        return self._get("/vault/smart-token-types").get("types", [])

    async def asmart_tokenize(
        self,
        real_value:      str,
        data_type:       str,
        allowed_actions: list = [],
        allowed_targets: list = [],
        allowed_fields:  list = [],
        max_uses:        int  = 1,
        ttl_seconds:     int  = 86400,
    ) -> dict:
        return await self._apost("/vault/smart-token", {
            "real_value":      real_value,
            "data_type":       data_type,
            "agent_id":        self.agent_id,
            "allowed_actions": allowed_actions,
            "allowed_targets": allowed_targets,
            "allowed_fields":  allowed_fields,
            "max_uses":        max_uses,
            "ttl_seconds":     ttl_seconds,
        })

    async def asmart_token_execute(
        self,
        token_id:    str,
        action_type: str = None,
        target_url:  str = None,
        field_name:  str = None,
    ) -> dict:
        return await self._apost("/vault/smart-token/execute", {
            "token_id":    token_id,
            "action_type": action_type,
            "target_url":  target_url,
            "field_name":  field_name,
            "agent_id":    self.agent_id,
        })


    # ── blind RAG (v4.3) ──────────────────────────────────────────────────────

    def rag_ingest(
        self,
        content:        dict,
        doc_type:       str,
        title:          str  = None,
        source:         str  = None,
        classification: str  = "pii",
    ) -> dict:
        """
        Tokenize a document and index it for blind semantic search.

        Real values tokenized before indexing.
        Agent can search and find — never sees real values.

        Usage:
            client.rag_ingest(
                content  = {"name": "John Smith", "age": "67",
                             "diagnosis": "diabetes", "risk": "high"},
                doc_type = "patient_record",
            )
        """
        return self._post("/rag/ingest", {
            "content":        content,
            "doc_type":       doc_type,
            "agent_id":       self.agent_id,
            "title":          title,
            "source":         source,
            "classification": classification,
        })

    def rag_ingest_batch(self, documents: list) -> dict:
        """Ingest multiple documents. Max 50 per call."""
        return self._post("/rag/ingest/batch", {
            "agent_id":  self.agent_id,
            "documents": documents,
        })

    def rag_search(
        self,
        query:     str,
        doc_type:  str   = None,
        top_k:     int   = 5,
        min_score: float = 0.3,
    ) -> dict:
        """
        Semantic search over tokenized documents.
        Returns token references — never real values.

        Usage:
            results = client.rag_search(
                "find diabetic patients over 65 with high risk"
            )
            for r in results["results"]:
                tokens = r["tokens"]  # ["[CVT:NAME:A1B2]", ...]
                # Pass tokens to executor to notify real patients
        """
        body = {"query": query, "top_k": top_k, "min_score": min_score}
        if doc_type: body["doc_type"] = doc_type
        return self._post("/rag/search", body)

    def rag_delete(self, doc_id: str) -> dict:
        """Delete a document from the blind RAG index."""
        r = self._get_sync().delete(f"{self.base_url}/rag/document/{doc_id}")
        r.raise_for_status()
        return r.json()

    def rag_stats(self) -> dict:
        """Vault RAG statistics."""
        return self._get("/rag/stats")

    async def arag_ingest(
        self,
        content:        dict,
        doc_type:       str,
        title:          str = None,
        classification: str = "pii",
    ) -> dict:
        return await self._apost("/rag/ingest", {
            "content": content, "doc_type": doc_type,
            "agent_id": self.agent_id, "title": title,
            "classification": classification,
        })

    async def arag_search(
        self,
        query:    str,
        doc_type: str   = None,
        top_k:    int   = 5,
        min_score: float = 0.3,
    ) -> dict:
        body = {"query": query, "top_k": top_k, "min_score": min_score}
        if doc_type: body["doc_type"] = doc_type
        return await self._apost("/rag/search", body)


    # ── policy-driven sensitivity (v4.4) ─────────────────────────────────────

    def register_sensitive_type(
        self,
        fields:    list,
        prefixes:  list = [],
        doc_types: list = [],
    ) -> dict:
        """
        Register custom sensitive field names for your tenant.
        Once registered, these are ALWAYS tokenized automatically.

        Usage:
            client.register_sensitive_type(
                fields    = ["employee_badge", "case_ref", "policy_number"],
                prefixes  = ["EMP-", "LEGAL-", "POL-"],
                doc_types = ["hr_record", "legal_filing"],
            )
            # Now employee_badge is always tokenized — no per-request config needed
        """
        return self._post("/policy/sensitivity/fields", {
            "fields":    fields,
            "prefixes":  prefixes,
            "doc_types": doc_types,
        })

    def set_sensitivity_policy(
        self,
        sensitive_fields:      list = None,
        sensitive_prefixes:    list = None,
        sensitive_doc_types:   list = None,
        field_classifications: dict = None,
        strict_mode:           bool = None,
    ) -> dict:
        """
        Set full sensitivity policy.

        field_classifications: {
            "employee_badge": "restricted",   # always tokenize
            "department":     "internal",     # keep but don't export
            "office_floor":   "public",       # never tokenize
        }
        """
        body = {}
        if sensitive_fields       is not None: body["sensitive_fields"]       = sensitive_fields
        if sensitive_prefixes     is not None: body["sensitive_prefixes"]     = sensitive_prefixes
        if sensitive_doc_types    is not None: body["sensitive_doc_types"]    = sensitive_doc_types
        if field_classifications  is not None: body["field_classifications"]  = field_classifications
        if strict_mode            is not None: body["strict_mode"]            = strict_mode
        return self._post("/policy/sensitivity", body)

    def get_sensitivity_policy(self) -> dict:
        """Get current sensitivity policy."""
        return self._get("/policy/sensitivity")

    def test_sensitivity(
        self,
        content:          dict,
        field_policy:     dict = {},
        sensitive_fields: list = [],
        tokenize_all:     bool = False,
    ) -> dict:
        """
        Test how your policy classifies fields — without actually tokenizing.
        Shows exactly what would be tokenized vs kept.

        Usage:
            result = client.test_sensitivity({
                "employee_badge": "EMP-77291",
                "name":           "John Smith",
                "department":     "Oncology",
                "age_range":      "65-75",
            })
            print(result["would_tokenize"])  # employee_badge, name
            print(result["would_keep"])      # department, age_range
        """
        return self._post("/policy/sensitivity/test", {
            "content":          content,
            "field_policy":     field_policy,
            "sensitive_fields": sensitive_fields,
            "tokenize_all":     tokenize_all,
        })

    def smart_ingest(
        self,
        content:          dict,
        doc_type:         str,
        field_policy:     dict = {},
        sensitive_fields: list = [],
        tokenize_all:     bool = False,
        title:            str  = None,
        classification:   str  = "pii",
    ) -> dict:
        """
        Ingest a document with full policy-driven sensitivity.
        Combines RAG ingest + policy resolution in one call.

        All three layers applied automatically:
          - Built-in: known field names + patterns
          - Per-request: field_policy + sensitive_fields
          - Tenant policy: registered via register_sensitive_type()

        Usage:
            # Simple — built-in detection handles it
            client.smart_ingest({"name": "John", "age": "67"}, "patient_record")

            # With custom fields
            client.smart_ingest(
                content          = {"employee_badge": "EMP-77291", "dept": "HR"},
                doc_type         = "hr_record",
                sensitive_fields = ["employee_badge"],
            )

            # With field-level classification
            client.smart_ingest(
                content      = {"badge": "EMP-77291", "floor": "3rd"},
                doc_type     = "employee_record",
                field_policy = {"badge": "tokenize", "floor": "public"},
            )
        """
        return self._post("/rag/ingest", {
            "content":          content,
            "doc_type":         doc_type,
            "agent_id":         self.agent_id,
            "title":            title,
            "classification":   classification,
            "field_policy":     field_policy,
            "sensitive_fields": sensitive_fields,
            "tokenize_all":     tokenize_all,
        })


    # ── context-aware + k-anonymity (v4.5) ───────────────────────────────────

    def set_context(
        self,
        industry:               str  = None,
        data_scope:             str  = None,
        classification_level:   str  = None,
        extra_sensitive_fields: list = [],
        safe_fields:            list = [],
        strict_mode:            bool = False,
    ) -> dict:
        """
        Register context-aware sensitivity rules.

        Industry profiles auto-applied:
            healthcare → diagnosis, medication, lab_result tokenized
            fintech    → salary, credit_score, transaction tokenized
            legal      → case_number, privilege, settlement tokenized
            government → clearance_level, classification tokenized
            hr         → salary, performance_rating tokenized

        Usage:
            client.set_context(industry="healthcare", data_scope="patient_records")
            # Now diagnosis, medication, lab_result etc. are always tokenized
            # Even if not in built-in detection
        """
        return self._post("/policy/context", {
            "industry":               industry,
            "data_scope":             data_scope,
            "classification_level":   classification_level,
            "extra_sensitive_fields": extra_sensitive_fields,
            "safe_fields":            safe_fields,
            "context_strict_mode":    strict_mode,
        })

    def set_anonymity(
        self,
        k_minimum:           int  = 5,
        suppress_singleton:  bool = True,
        auto_bucket:         bool = True,
        detect_narrowing:    bool = True,
        quasi_identifiers:   list = None,
    ) -> dict:
        """
        Configure k-anonymity protection.

        Protects against re-identification even when names are tokenized.
        "67yo diabetic in zip 30314" → 1 result → suppressed (below k=5)
        age:67 → auto-bucketed to 65-74
        zip:30314 → auto-bucketed to 303xxx
        Narrowing attacks → detected and blocked

        Usage:
            client.set_anonymity(k_minimum=5, auto_bucket=True)
        """
        body = {
            "k_minimum":          k_minimum,
            "suppress_singleton": suppress_singleton,
            "auto_bucket":        auto_bucket,
            "detect_narrowing":   detect_narrowing,
        }
        if quasi_identifiers is not None:
            body["quasi_identifiers"] = quasi_identifiers
        return self._post("/policy/anonymity", body)

    def test_context(
        self,
        content:  dict,
        context:  dict,
        field_policy: dict = {},
    ) -> dict:
        """
        Test context-aware classification without ingesting.

        Usage:
            result = client.test_context(
                content = {"diagnosis": "diabetes", "age": "67", "dept": "cardiology"},
                context = {"industry": "healthcare"},
            )
            # would_tokenize: diagnosis (context-sensitive in healthcare)
            # would_keep: dept (not sensitive)
        """
        return self._post("/policy/context/test", {
            "content":      content,
            "context":      context,
            "field_policy": field_policy,
        })

    def smart_ingest_with_context(
        self,
        content:   dict,
        doc_type:  str,
        context:   dict = {},
        field_policy: dict = {},
        sensitive_fields: list = [],
        tokenize_all: bool = False,
        title:     str  = None,
    ) -> dict:
        """
        Full pipeline ingest — policy + context + k-anonymity protection.

        Usage:
            client.smart_ingest_with_context(
                content  = {"name": "John", "diagnosis": "diabetes", "age": "67"},
                doc_type = "patient_record",
                context  = {"industry": "healthcare", "data_scope": "patient_records"},
            )
        """
        return self._post("/rag/ingest", {
            "content":          content,
            "doc_type":         doc_type,
            "agent_id":         self.agent_id,
            "title":            title,
            "context":          context,
            "field_policy":     field_policy,
            "sensitive_fields": sensitive_fields,
            "tokenize_all":     tokenize_all,
        })

    def info(self) -> dict:
        return {
            "mode":      self.mode,
            "base_url":  self.base_url,
            "agent_id":  self.agent_id,
            "zero_log":  self.zero_log,
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
        return f"CodeAstraClient(mode={self.mode!r}, agent_id={self.agent_id!r}, zero_log={self.zero_log})"
