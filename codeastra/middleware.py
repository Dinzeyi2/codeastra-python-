"""
BlindAgentMiddleware — drop-in middleware for LangChain, CrewAI, AutoGPT.

Two lines. Any agent becomes blind.

    from codeastra import BlindAgentMiddleware
    agent = BlindAgentMiddleware(your_langchain_agent, api_key="sk-guard-xxx")

How it works:
  1. Intercepts every tool call before the agent sees the result
  2. Scans the result for PII/PHI/PCI fields
  3. Tokenizes detected fields → stores real values in Codeastra vault
  4. Returns tokens to the agent — agent reasons on tokens, never real data
  5. When agent submits a final action, intercepts it, resolves tokens → executes

Supports: LangChain AgentExecutor, CrewAI Agent, AutoGPT-style run() agents,
          any object with .run() / .invoke() / .chat() / .step()
"""
from __future__ import annotations

import re
import json
import inspect
import functools
from typing import Any, Callable, Optional

from .client import CodeAstraClient, TOKEN_RE

# Fields that trigger automatic tokenization when found in tool output
_PII_FIELDS = {
    "name", "first_name", "last_name", "full_name",
    "email", "email_address",
    "phone", "phone_number", "mobile",
    "ssn", "social_security", "social_security_number",
    "dob", "date_of_birth", "birthday",
    "address", "street", "zip", "postal_code",
    "credit_card", "card_number", "cvv", "expiry",
    "mrn", "patient_id", "npi",
    "account_number", "routing_number", "iban",
    "passport", "license", "drivers_license",
    "ip", "ip_address", "mac_address",
    "username", "user_id", "employee_id",
}

_PHI_FIELDS = {
    "diagnosis", "icd_code", "medication", "prescription",
    "allergy", "lab_result", "test_result", "condition",
    "treatment", "procedure", "insurance_id", "member_id",
}

_PCI_FIELDS = {
    "card_number", "credit_card", "cvv", "expiry",
    "account_number", "routing_number",
}


def _classify(fields: set) -> str:
    if fields & _PCI_FIELDS: return "pci"
    if fields & _PHI_FIELDS: return "phi"
    return "pii"


def _extract_sensitive(obj: Any) -> dict:
    """
    Walk a dict/str/list and extract fields that look sensitive.
    Returns flat dict of {field: value} pairs to tokenize.
    """
    found = {}

    def _walk(o, prefix=""):
        if isinstance(o, dict):
            for k, v in o.items():
                key = k.lower().replace(" ", "_").replace("-", "_")
                if key in (_PII_FIELDS | _PHI_FIELDS | _PCI_FIELDS):
                    if isinstance(v, str) and v and not TOKEN_RE.fullmatch(v.strip()):
                        found[k] = v
                else:
                    _walk(v, prefix=k)
        elif isinstance(o, list):
            for item in o:
                _walk(item, prefix)

    if isinstance(obj, dict):
        _walk(obj)
    elif isinstance(obj, str):
        # Try JSON parse
        try:
            _walk(json.loads(obj))
        except Exception:
            pass
    return found


def _tokenize_in_place(obj: Any, token_map: dict) -> Any:
    """
    Replace real values with tokens throughout a nested object.
    token_map: {real_value: token}
    """
    if isinstance(obj, str):
        for real, token in token_map.items():
            obj = obj.replace(str(real), token)
        return obj
    elif isinstance(obj, dict):
        return {k: _tokenize_in_place(v, token_map) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_tokenize_in_place(i, token_map) for i in obj]
    return obj


class BlindAgentMiddleware:
    """
    Drop-in middleware that makes any agent framework blind to real data.

    Works with:
      - LangChain:  AgentExecutor, RunnableAgent, Chain
      - CrewAI:     Agent, Crew
      - AutoGPT:    any object with .run() / .step()
      - Generic:    anything with .run() / .invoke() / .chat()

    Usage:
        # LangChain
        from codeastra import BlindAgentMiddleware
        agent = BlindAgentMiddleware(langchain_executor, api_key="sk-guard-xxx")
        result = agent.invoke({"input": "Schedule appointment for patient"})

        # CrewAI
        crew   = BlindAgentMiddleware(my_crew, api_key="sk-guard-xxx")
        result = crew.run()

        # With pipeline (multi-agent)
        agent_a = BlindAgentMiddleware(intake_agent,      api_key="sk-guard-xxx", agent_id="intake")
        agent_b = BlindAgentMiddleware(scheduling_agent,  api_key="sk-guard-xxx", agent_id="scheduling")

    Args:
        agent:          The underlying agent object to wrap
        api_key:        Your Codeastra API key (sk-guard-xxx)
        agent_id:       Unique ID for this agent in the pipeline (default: "sdk-agent")
        base_url:       Codeastra API base URL (default: https://app.codeastra.dev)
        classification: Default data classification: "pii", "phi", or "pci"
        pipeline_id:    Optional pipeline ID for multi-agent tracking
        on_tokenize:    Optional callback(field, token) called when data is tokenized
        verbose:        Print tokenization events to stdout (default: False)
    """

    def __init__(
        self,
        agent:          Any,
        api_key:        str,
        agent_id:       str = "sdk-agent",
        base_url:       str = "https://app.codeastra.dev",
        classification: str = "pii",
        pipeline_id:    Optional[str] = None,
        on_tokenize:    Optional[Callable] = None,
        verbose:        bool = False,
    ):
        self._agent          = agent
        self._client         = CodeAstraClient(api_key, base_url, agent_id)
        self._classification = classification
        self._pipeline_id    = pipeline_id
        self._on_tokenize    = on_tokenize
        self._verbose        = verbose

        # Track tokens minted this session: {field_key: token}
        self._session_tokens: dict = {}
        # Reverse map: {real_value: token}
        self._value_to_token: dict = {}

        # Patch agent's tool call mechanism
        self._patch_tools()

    # ── tool patching ─────────────────────────────────────────────────────────

    def _patch_tools(self):
        """
        Intercept tool calls on the underlying agent.
        Supports LangChain tools, CrewAI tools, generic callables.
        """
        agent = self._agent

        # LangChain: AgentExecutor has .tools list
        if hasattr(agent, "tools") and isinstance(agent.tools, list):
            for i, tool in enumerate(agent.tools):
                agent.tools[i] = self._wrap_tool(tool)
            if self._verbose:
                print(f"[CodeAstra] Patched {len(agent.tools)} LangChain tools")

        # LangChain: RunnableAgent / chain with .steps
        if hasattr(agent, "steps"):
            for step in agent.steps:
                if hasattr(step, "tool"):
                    step.tool = self._wrap_tool(step.tool)

        # CrewAI: Agent has .tools
        if hasattr(agent, "agent") and hasattr(agent.agent, "tools"):
            tools = agent.agent.tools
            for i, tool in enumerate(tools):
                tools[i] = self._wrap_tool(tool)

    def _wrap_tool(self, tool: Any) -> Any:
        """
        Wrap a single tool so its output is tokenized before the agent sees it.
        Works with LangChain BaseTool, CrewAI tools, and plain callables.
        """
        # LangChain BaseTool — has ._run and .run
        if hasattr(tool, "_run"):
            original_run  = tool._run
            original_arun = getattr(tool, "_arun", None)

            @functools.wraps(original_run)
            def patched_run(*args, **kwargs):
                result = original_run(*args, **kwargs)
                return self._blind_output(result)

            tool._run = patched_run

            if original_arun:
                @functools.wraps(original_arun)
                async def patched_arun(*args, **kwargs):
                    result = await original_arun(*args, **kwargs)
                    return self._blind_output(result)
                tool._arun = patched_arun

            return tool

        # Plain callable
        if callable(tool):
            @functools.wraps(tool)
            def wrapped(*args, **kwargs):
                result = tool(*args, **kwargs)
                return self._blind_output(result)
            return wrapped

        return tool

    # ── core blindness logic ──────────────────────────────────────────────────

    def _blind_output(self, output: Any) -> Any:
        """
        Given any tool output, tokenize all sensitive fields.
        Returns the same structure with real values replaced by tokens.
        """
        sensitive = _extract_sensitive(output)
        if not sensitive:
            return output

        classification = _classify(set(k.lower() for k in sensitive))
        try:
            tokens = self._client.tokenize(sensitive, classification=classification)
        except Exception as e:
            if self._verbose:
                print(f"[CodeAstra] Warning: tokenization failed: {e}")
            return output

        # Build reverse map for replacement
        for field, token in tokens.items():
            real_val = sensitive.get(field)
            if real_val:
                self._value_to_token[str(real_val)] = token
            self._session_tokens[field] = token

        if self._on_tokenize:
            for field, token in tokens.items():
                try: self._on_tokenize(field, token)
                except Exception: pass

        if self._verbose:
            print(f"[CodeAstra] Tokenized {len(tokens)} field(s): {list(tokens.keys())}")

        return _tokenize_in_place(output, self._value_to_token)

    async def _ablind_output(self, output: Any) -> Any:
        """Async version of _blind_output."""
        sensitive = _extract_sensitive(output)
        if not sensitive:
            return output

        classification = _classify(set(k.lower() for k in sensitive))
        try:
            tokens = await self._client.atokenize(sensitive, classification=classification)
        except Exception as e:
            if self._verbose:
                print(f"[CodeAstra] Warning: async tokenization failed: {e}")
            return output

        for field, token in tokens.items():
            real_val = sensitive.get(field)
            if real_val:
                self._value_to_token[str(real_val)] = token
            self._session_tokens[field] = token

        if self._verbose:
            print(f"[CodeAstra] Tokenized {len(tokens)} field(s): {list(tokens.keys())}")

        return _tokenize_in_place(output, self._value_to_token)

    # ── pipeline: grant tokens to next agent ─────────────────────────────────

    def grant_to(
        self,
        next_agent_id:   str,
        allowed_actions: list[str] = [],
        purpose:         str = None,
    ) -> dict:
        """
        Grant all tokens minted this session to the next agent in the pipeline.

        agent_a.run(input)
        grant = agent_a.grant_to("scheduling-agent", ["schedule_appointment"])
        # Now scheduling-agent can use agent_a's tokens
        """
        tokens = list(self._session_tokens.values())
        if not tokens:
            return {"granted": False, "error": "No tokens minted this session"}
        return self._client.grant(
            receiving_agent = next_agent_id,
            tokens          = tokens,
            allowed_actions = allowed_actions,
            pipeline_id     = self._pipeline_id,
            purpose         = purpose,
        )

    async def agrant_to(
        self,
        next_agent_id:   str,
        allowed_actions: list[str] = [],
    ) -> dict:
        tokens = list(self._session_tokens.values())
        if not tokens:
            return {"granted": False, "error": "No tokens minted this session"}
        return await self._client.agrant(
            receiving_agent = next_agent_id,
            tokens          = tokens,
            allowed_actions = allowed_actions,
            pipeline_id     = self._pipeline_id,
        )

    # ── execute action with tokens ────────────────────────────────────────────

    def execute(self, action_type: str, params: dict) -> dict:
        """Submit a final action. Tokens in params are resolved by Codeastra."""
        return self._client.execute(action_type, params, self._pipeline_id)

    async def aexecute(self, action_type: str, params: dict) -> dict:
        return await self._client.aexecute(action_type, params, self._pipeline_id)

    # ── proxy all agent methods ───────────────────────────────────────────────

    def run(self, *args, **kwargs):
        """Proxy .run() — used by CrewAI, AutoGPT, generic agents."""
        result = self._agent.run(*args, **kwargs)
        return self._blind_output(result)

    def invoke(self, *args, **kwargs):
        """Proxy .invoke() — used by LangChain LCEL chains."""
        result = self._agent.invoke(*args, **kwargs)
        if isinstance(result, dict) and "output" in result:
            result["output"] = self._blind_output(result["output"])
            return result
        return self._blind_output(result)

    def chat(self, *args, **kwargs):
        """Proxy .chat() — used by various chat-style agents."""
        result = self._agent.chat(*args, **kwargs)
        return self._blind_output(result)

    async def arun(self, *args, **kwargs):
        result = await self._agent.arun(*args, **kwargs)
        return await self._ablind_output(result)

    async def ainvoke(self, *args, **kwargs):
        result = await self._agent.ainvoke(*args, **kwargs)
        if isinstance(result, dict) and "output" in result:
            result["output"] = await self._ablind_output(result["output"])
            return result
        return await self._ablind_output(result)

    # ── session info ──────────────────────────────────────────────────────────

    @property
    def tokens(self) -> dict:
        """All tokens minted this session. {field: token}"""
        return dict(self._session_tokens)

    @property
    def token_count(self) -> int:
        return len(self._session_tokens)

    def audit(self) -> list:
        """Get chain of custody for this session's pipeline."""
        return self._client.audit(pipeline_id=self._pipeline_id)

    # ── pass-through attribute access to underlying agent ────────────────────

    def __getattr__(self, name: str):
        """Fall through to the underlying agent for any unpatched attribute."""
        return getattr(self._agent, name)

    def __repr__(self):
        return (f"BlindAgentMiddleware(agent={type(self._agent).__name__}, "
                f"agent_id={self._client.agent_id!r}, "
                f"tokens_minted={self.token_count})")

    def close(self):
        self._client.close()

    async def aclose(self):
        await self._client.aclose()

    def __enter__(self):  return self
    def __exit__(self, *_): self.close()
    async def __aenter__(self): return self
    async def __aexit__(self, *_): await self.aclose()
