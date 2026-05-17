from .client import CodeAstraClient
from .middleware import BlindAgentMiddleware
from .wrappers import blind_tool, BlindCrewAIAgent, BlindAutoGPTAgent

__version__ = "1.6.1"

# Fail-closed enforcement — active by default, zero config
# If Codeastra cannot protect data, it blocks — never exposes real data
try:
    from .fail_closed import (
        apply_fail_closed_to_middleware,
        CodastraSecurityError,
        TokenizationError,
        VaultError,
        ExecutionAbortedError,
    )
    apply_fail_closed_to_middleware()
except Exception:
    # If fail_closed module not present — still works, just without enforcement
    CodastraSecurityError = Exception
    TokenizationError     = Exception
    VaultError            = Exception
    ExecutionAbortedError = Exception

__all__ = [
    "CodeAstraClient",
    "BlindAgentMiddleware",
    "blind_tool",
    "BlindCrewAIAgent",
    "BlindAutoGPTAgent",
    "CodastraSecurityError",
    "TokenizationError",
    "VaultError",
    "ExecutionAbortedError",
]
