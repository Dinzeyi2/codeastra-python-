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
