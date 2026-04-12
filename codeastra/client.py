"""
CodeAstraClient — low-level async/sync HTTP client for the Codeastra API.
All SDK components use this. Customers can also use it directly.
"""
from __future__ import annotations

import re
import json
import asyncio
import threading
from typing import Any, Optional

import httpx

TOKEN_RE = re.compile(r'\[CVT:[A-Z]+:[A-F0-9]+\]')

_DEFAULT_BASE = "https://app.codeastra.dev"


class CodeAstraClient:
    """
    Thin wrapper around the Codeastra REST API.

    Usage:
        client = CodeAstraClient(api_key="sk-guard-xxx")
        tokens = client.tokenize({"name": "John Smith", "ssn": "123-45-6789"})
        # → {"name": "[CVT:NAME:A1B2]", "ssn": "[CVT:SSN:C3D4]"}
    """

    def __init__(
        self,
        api_key:   str,
        base_url:  str = _DEFAULT_BASE,
        agent_id:  str = "sdk-agent",
        timeout:   float = 10.0,
        executor_url: str = None,   # optional: bring your own executor
    ):
        self.api_key  = api_key
        self.base_url = base_url.rstrip("/")
        self.agent_id = agent_id
        self._headers = {
            "X-API-Key":    api_key,
            "Content-Type": "application/json",
        }
        self._timeout = timeout
        self._executor_url = executor_url
        # Sync client (lazy)
        self._sync_client: Optional[httpx.Client] = None
        # Async client (lazy)
        self._async_client: Optional[httpx.AsyncClient] = None
        # Auto-register executor if provided
        if executor_url:
            try:
                self._post("/agent/executor", {
                    "execution_url": executor_url,
                    "action_type": "*",
                    "agent_id": agent_id,
                    "description": f"Auto-registered by SDK agent {agent_id}",
                })
            except Exception:
                pass  # non-fatal — zero-config mode still works

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

    def tokenize(
        self,
        data:           dict,
        classification: str = "pii",
        ttl_hours:      int = 24,
    ) -> dict:
        """
        Store real data in vault. Returns token map.
        {"name": "John"} → {"name": "[CVT:NAME:A1B2]"}
        """
        resp = self._post("/vault/store", {
            "data":           data,
            "agent_id":       self.agent_id,
            "classification": classification,
            "ttl_hours":      ttl_hours,
        })
        return resp.get("tokens", {})

    def execute(
        self,
        action_type: str,
        params:      dict,
        pipeline_id: str = None,
    ) -> dict:
        """
        Submit an action with token params.
        Codeastra resolves tokens → real values → POSTs to your executor.
        Agent never sees real values.
        """
        body = {
            "agent_id":    self.agent_id,
            "action_type": action_type,
            "params":      params,
        }
        if pipeline_id:
            body["pipeline_id"] = pipeline_id
            return self._post("/pipeline/action", body)
        return self._post("/agent/action", body)

    def grant(
        self,
        receiving_agent: str,
        tokens:          list[str],
        allowed_actions: list[str] = [],
        pipeline_id:     str = None,
        purpose:         str = None,
    ) -> dict:
        """Grant tokens to another agent in a pipeline."""
        return self._post("/vault/grant", {
            "granting_agent":  self.agent_id,
            "receiving_agent": receiving_agent,
            "tokens":          tokens,
            "allowed_actions": allowed_actions,
            "pipeline_id":     pipeline_id,
            "purpose":         purpose,
        })

    def audit(self, pipeline_id: str = None, token: str = None) -> list:
        """Get chain of custody for a pipeline or token."""
        params = {}
        if pipeline_id: params["pipeline_id"] = pipeline_id
        if token:       params["token"]       = token
        return self._get("/pipeline/audit", params).get("audit", [])

    # ── public async API ──────────────────────────────────────────────────────

    async def atokenize(
        self,
        data:           dict,
        classification: str = "pii",
        ttl_hours:      int = 24,
    ) -> dict:
        resp = await self._apost("/vault/store", {
            "data":           data,
            "agent_id":       self.agent_id,
            "classification": classification,
            "ttl_hours":      ttl_hours,
        })
        return resp.get("tokens", {})

    async def aexecute(
        self,
        action_type: str,
        params:      dict,
        pipeline_id: str = None,
    ) -> dict:
        body = {
            "agent_id":    self.agent_id,
            "action_type": action_type,
            "params":      params,
        }
        if pipeline_id:
            body["pipeline_id"] = pipeline_id
            return await self._apost("/pipeline/action", body)
        return await self._apost("/agent/action", body)

    async def agrant(
        self,
        receiving_agent: str,
        tokens:          list[str],
        allowed_actions: list[str] = [],
        pipeline_id:     str = None,
    ) -> dict:
        return await self._apost("/vault/grant", {
            "granting_agent":  self.agent_id,
            "receiving_agent": receiving_agent,
            "tokens":          tokens,
            "allowed_actions": allowed_actions,
            "pipeline_id":     pipeline_id,
        })

    # ── utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def extract_tokens(obj: Any) -> list[str]:
        """Extract all vault tokens from any string/dict/list."""
        text = json.dumps(obj) if not isinstance(obj, str) else obj
        return TOKEN_RE.findall(text)

    @staticmethod
    def contains_token(val: Any) -> bool:
        text = json.dumps(val) if not isinstance(val, str) else str(val)
        return bool(TOKEN_RE.search(text))

    @staticmethod
    def is_token(val: str) -> bool:
        return bool(TOKEN_RE.fullmatch(val.strip()))

    def close(self):
        if self._sync_client:  self._sync_client.close()

    async def aclose(self):
        if self._async_client: await self._async_client.aclose()

    def __enter__(self):  return self
    def __exit__(self, *_): self.close()
    async def __aenter__(self): return self
    async def __aexit__(self, *_): await self.aclose()
