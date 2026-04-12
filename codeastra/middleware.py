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
        api_key:        str  = None,
        agent_id:       str  = "sdk-agent",
        base_url:       str  = None,
        classification: str  = "pii",
        pipeline_id:    Optional[str] = None,
        on_tokenize:    Optional[Callable] = None,
        verbose:        bool = False,
        mode:           str  = "auto",        # auto | cloud | onprem | hybrid
        zero_log:       bool = False,
        executor_url:   str  = None,
        onprem_dir:     str  = "./codeastra-onprem",
    ):
        self._agent          = agent
        self._client         = CodeAstraClient(
            api_key      = api_key,
            base_url     = base_url,
            agent_id     = agent_id,
            mode         = mode,
            zero_log     = zero_log,
            executor_url = executor_url,
            onprem_dir   = onprem_dir,
            verbose      = verbose,
        )
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


# ══════════════════════════════════════════════════════════════════════════════
# INPUT SCANNER — scans prompt text for raw PII/PHI/PCI before agent sees it
# OUTPUT SCANNER — scans agent response for any leaked real values
# ══════════════════════════════════════════════════════════════════════════════

import re as _re

# Regex patterns for detecting raw sensitive data in free text
_PATTERNS = {
    # SSN: 123-45-6789 or 123456789
    "ssn": _re.compile(
        r'\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b'
    ),
    # Credit card: 13-19 digits, passes Luhn
    "credit_card": _re.compile(
        r'\b(?:4[0-9]{12}(?:[0-9]{3})?'       # Visa
        r'|5[1-5][0-9]{14}'                    # Mastercard
        r'|3[47][0-9]{13}'                     # Amex
        r'|6(?:011|5[0-9]{2})[0-9]{12}'       # Discover
        r'|(?:2131|1800|35\d{3})\d{11})\b'    # JCB
    ),
    # Email
    "email": _re.compile(
        r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'
    ),
    # Phone: various formats
    "phone": _re.compile(
        r'\b(?:\+?1[-.\s]?)?'
        r'(?:\(?\d{3}\)?[-.\s]?)'
        r'\d{3}[-.\s]?\d{4}\b'
    ),
    # DOB: MM/DD/YYYY or YYYY-MM-DD
    "dob": _re.compile(
        r'\b(?:0[1-9]|1[0-2])[\/\-](?:0[1-9]|[12]\d|3[01])[\/\-](?:19|20)\d{2}\b'
        r'|\b(?:19|20)\d{2}[\/\-](?:0[1-9]|1[0-2])[\/\-](?:0[1-9]|[12]\d|3[01])\b'
    ),
    # MRN: MRN- or MRN: followed by digits
    "mrn": _re.compile(
        r'\bMRN[-:\s]*\s*[A-Z0-9]{4,12}\b', _re.IGNORECASE
    ),
    # IP address
    "ip_address": _re.compile(
        r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
    ),
}


def _luhn_check(number: str) -> bool:
    """Validate credit card number with Luhn algorithm."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _scan_text_for_pii(text: str) -> dict:
    """
    Scan free text for raw PII/PHI/PCI patterns.
    Returns {synthetic_field_key: matched_value} for tokenization.

    Example:
        "Patient John Smith SSN 123-45-6789 email john@hospital.org"
        → {"ssn_0": "123-45-6789", "email_0": "john@hospital.org"}
    """
    found = {}
    if not isinstance(text, str):
        return found

    for field, pattern in _PATTERNS.items():
        matches = pattern.findall(text)
        for i, match in enumerate(matches):
            val = match.strip() if isinstance(match, str) else match[0].strip()
            if not val or TOKEN_RE.search(val):
                continue
            # Extra validation for credit cards
            if field == "credit_card":
                digits_only = _re.sub(r'\D', '', val)
                if not _luhn_check(digits_only):
                    continue
            key = f"{field}_{i}" if i > 0 else field
            found[key] = val

    return found


def _scan_obj_for_pii(obj: Any) -> dict:
    """Scan any object (str, dict, list) for raw PII in free text."""
    if isinstance(obj, str):
        return _scan_text_for_pii(obj)
    elif isinstance(obj, dict):
        combined = {}
        for v in obj.values():
            combined.update(_scan_obj_for_pii(v))
        return combined
    elif isinstance(obj, list):
        combined = {}
        for item in obj:
            combined.update(_scan_obj_for_pii(item))
        return combined
    return {}


def _blind_text(text: str, token_map: dict) -> str:
    """Replace all known real values in text with their tokens."""
    if not isinstance(text, str):
        return text
    for real, token in token_map.items():
        if real and real in text:
            text = text.replace(real, token)
    return text


def _blind_any(obj: Any, token_map: dict) -> Any:
    """Replace real values anywhere in obj with tokens."""
    if isinstance(obj, str):
        return _blind_text(obj, token_map)
    elif isinstance(obj, dict):
        return {k: _blind_any(v, token_map) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_blind_any(i, token_map) for i in obj]
    return obj


# ── Patch BlindAgentMiddleware with input + output scanning ──────────────────

_orig_run    = BlindAgentMiddleware.run
_orig_invoke = BlindAgentMiddleware.invoke
_orig_chat   = BlindAgentMiddleware.chat
_orig_arun   = BlindAgentMiddleware.arun
_orig_ainvoke = BlindAgentMiddleware.ainvoke


def _scan_and_blind_input(self, *args, **kwargs):
    """
    Scan all input args/kwargs for raw PII/PHI/PCI.
    Tokenize any found values before passing to the agent.
    Returns (new_args, new_kwargs).
    """
    # Collect all text from args and kwargs
    all_text = json.dumps(list(args)) + json.dumps(kwargs)
    raw_pii  = _scan_obj_for_pii(all_text)

    if not raw_pii:
        return args, kwargs

    # Tokenize detected values
    try:
        classification = _classify(set(k.split("_")[0] for k in raw_pii))
        minted = self._client.tokenize(raw_pii, classification=classification)
        # Build replacement map: {real_value: token}
        for field, token in minted.items():
            real_val = raw_pii.get(field)
            if real_val:
                self._value_to_token[real_val] = token
            self._session_tokens[field] = token

        if self._verbose:
            print(f"[CodeAstra] Input scan: tokenized {len(minted)} value(s) in prompt: {list(minted.keys())}")

        # Replace real values in args and kwargs
        new_args   = tuple(_blind_any(a, self._value_to_token) for a in args)
        new_kwargs = {k: _blind_any(v, self._value_to_token) for k, v in kwargs.items()}
        return new_args, new_kwargs

    except Exception as e:
        if self._verbose:
            print(f"[CodeAstra] Input scan warning: {e}")
        return args, kwargs


def _scan_output(self, result: Any) -> Any:
    """
    Scan agent output for any real values that leaked through.
    Replace with tokens using session's value_to_token map.
    Also scan output text for any NEW raw PII not yet tokenized.
    """
    # Step 1: replace known real values with existing tokens
    if self._value_to_token:
        result = _blind_any(result, self._value_to_token)

    # Step 2: scan output for any new raw PII that leaked
    new_pii = _scan_obj_for_pii(result)
    if new_pii:
        try:
            classification = _classify(set(k.split("_")[0] for k in new_pii))
            minted = self._client.tokenize(new_pii, classification=classification)
            for field, token in minted.items():
                real_val = new_pii.get(field)
                if real_val:
                    self._value_to_token[real_val] = token
                self._session_tokens[field] = token

            result = _blind_any(result, self._value_to_token)

            if self._verbose:
                print(f"[CodeAstra] Output gate: caught {len(minted)} leaked value(s): {list(minted.keys())}")
        except Exception as e:
            if self._verbose:
                print(f"[CodeAstra] Output gate warning: {e}")

    return result


async def _ascan_output(self, result: Any) -> Any:
    """Async version of _scan_output."""
    if self._value_to_token:
        result = _blind_any(result, self._value_to_token)

    new_pii = _scan_obj_for_pii(result)
    if new_pii:
        try:
            classification = _classify(set(k.split("_")[0] for k in new_pii))
            minted = await self._client.atokenize(new_pii, classification=classification)
            for field, token in minted.items():
                real_val = new_pii.get(field)
                if real_val:
                    self._value_to_token[real_val] = token
                self._session_tokens[field] = token
            result = _blind_any(result, self._value_to_token)
            if self._verbose:
                print(f"[CodeAstra] Output gate (async): caught {len(minted)} leaked value(s)")
        except Exception as e:
            if self._verbose:
                print(f"[CodeAstra] Output gate warning: {e}")
    return result


# ── Monkey-patch all proxy methods with input + output scanning ───────────────

def _patched_run(self, *args, **kwargs):
    args, kwargs = _scan_and_blind_input(self, *args, **kwargs)
    result = self._agent.run(*args, **kwargs)
    result = self._blind_output(result)       # tool output scan (existing)
    return _scan_output(self, result)         # output gate scan (new)

def _patched_invoke(self, *args, **kwargs):
    args, kwargs = _scan_and_blind_input(self, *args, **kwargs)
    result = self._agent.invoke(*args, **kwargs)
    if isinstance(result, dict) and "output" in result:
        result["output"] = self._blind_output(result["output"])
        result["output"] = _scan_output(self, result["output"])
        return result
    result = self._blind_output(result)
    return _scan_output(self, result)

def _patched_chat(self, *args, **kwargs):
    args, kwargs = _scan_and_blind_input(self, *args, **kwargs)
    result = self._agent.chat(*args, **kwargs)
    result = self._blind_output(result)
    return _scan_output(self, result)

async def _patched_arun(self, *args, **kwargs):
    args, kwargs = _scan_and_blind_input(self, *args, **kwargs)
    result = await self._agent.arun(*args, **kwargs)
    result = await self._ablind_output(result)
    return await _ascan_output(self, result)

async def _patched_ainvoke(self, *args, **kwargs):
    args, kwargs = _scan_and_blind_input(self, *args, **kwargs)
    result = await self._agent.ainvoke(*args, **kwargs)
    if isinstance(result, dict) and "output" in result:
        result["output"] = await self._ablind_output(result["output"])
        result["output"] = await _ascan_output(self, result["output"])
        return result
    result = await self._ablind_output(result)
    return await _ascan_output(self, result)


# Apply patches
BlindAgentMiddleware.run              = _patched_run
BlindAgentMiddleware.invoke           = _patched_invoke
BlindAgentMiddleware.chat             = _patched_chat
BlindAgentMiddleware.arun             = _patched_arun
BlindAgentMiddleware.ainvoke          = _patched_ainvoke

# Expose scanner functions for direct use
BlindAgentMiddleware._scan_input      = _scan_and_blind_input
BlindAgentMiddleware._scan_output     = _scan_output
BlindAgentMiddleware.scan_text        = staticmethod(_scan_text_for_pii)
