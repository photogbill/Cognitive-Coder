# SPDX-License-Identifier: Apache-2.0
"""Cognitive Coder — a host-agnostic engine for writing, building and fixing code.

**This module is the public API.** Nothing else in the package is stable
(C9, M8). What is re-exported here — the Ports of `ports.py` and the shared
types of `types.py` — is frozen at 1.0 under semver; breaking either is a
major version.

Two rules govern this file, both from §1.2 / M50, because a host may vendor
this repo as a git submodule and `sys.path`-insert it with no install at all:

  * **No import-time side effects.** Importing this package must not create a
    directory, open a database, read a config file, or probe a toolchain.
  * **No package-metadata reads.** The version comes from `version.py`, never
    from `importlib.metadata`, which raises for a vendored copy.

Ten-line embedding example — it runs anywhere, with no model and no network:

    from cognitive_coder import (AutoApprove, Host, MemoryFileSystem,
                                 ScriptedLLM, Session)

    host = Host(llm=ScriptedLLM(["def main():\\n    return 0\\n"]),
                fs=MemoryFileSystem({"src/main.py": b""}),
                approval=AutoApprove())
    session = Session(host)
    session.run("make src/main.py return zero")
    print(session.report())

The meta-lesson, worth knowing before you tune anything (Appendix D): with a
frontier model you improve results by improving the prompt. With a small model
you improve results by improving the **loop**.
"""

from __future__ import annotations

# -- the engine ------------------------------------------------------------
from . import (
               context,
               diagnostics,
               guard,
               langs,
               patcher,
               redact,
               review,
               runner,
               textio,
)
from .codemap import CodeMap

# -- errors (C6: every one of these carries an operator-facing sentence) ----
from .errors import (
               BudgetExceeded,
               Cancelled,
               CognitiveCoderError,
               ConfigurationError,
               GuardRefusal,
               NoModelLoadedError,
               PathEscape,
               PortError,
               TransactionError,
)
from .journal import Journal
from .loop import Loop, LoopConfig
from .patcher import Patcher, Transaction
from .personas import (
               Persona,
               PromptBuilder,
               detect_commentary,
               strip_commentary,
               strip_think,
)
from .planner import Planner
from .ports import (
               ApprovalPort,
               AutoApprove,
               Cancel,
               CancelToken,
               DenyAll,
               EventPort,
               ExecPort,
               FileSystemPort,
               Host,
               LLMPort,
               LocalFileSystem,
               MemoryFileSystem,
               MemoryStorage,
               NeverCancelled,
               NullLLM,
               RecordingEvents,
               ScriptedLLM,
               SilentEvents,
               StoragePort,
               SubprocessExec,
)
from .providers import RemoteGate, available_providers, make_provider
from .redact import Budget, RedactionReport
from .session import Session, SessionConfig

# -- the contract (§5) -----------------------------------------------------
from .types import (
               EVENT_KINDS,
               FINISH_REASONS,
               JOURNAL_EVENTS,
               ROLES,
               AttemptRecord,
               CodemapStats,
               Completion,
               Diagnostic,
               Edit,
               EditResult,
               GuardFinding,
               JournalEvent,
               Message,
               ModelCapabilities,
               PhaseResult,
               Plan,
               ProcResult,
               RunResult,
               Symbol,
               Task,
               TaskOutcome,
               ToolCall,
               ToolSpec,
               TransactionRecord,
)
from .version import API_VERSION, IMPLEMENTED_PHASES, __version__

__all__ = [
    # version
    "__version__", "API_VERSION", "IMPLEMENTED_PHASES",
    # errors
    "CognitiveCoderError", "PortError", "ConfigurationError",
    "NoModelLoadedError", "GuardRefusal", "PathEscape", "TransactionError",
    "Cancelled", "BudgetExceeded",
    # shared types — half the frozen contract
    "ToolSpec", "ToolCall", "Message", "Completion", "ModelCapabilities",
    "ProcResult", "Diagnostic", "GuardFinding", "PhaseResult", "RunResult",
    "Task", "Plan", "Edit", "EditResult", "TransactionRecord", "Symbol",
    "CodemapStats", "JournalEvent", "AttemptRecord", "TaskOutcome",
    "ROLES", "FINISH_REASONS", "EVENT_KINDS", "JOURNAL_EVENTS",
    # ports — the other half
    "LLMPort", "FileSystemPort", "ExecPort", "StoragePort", "EventPort",
    "ApprovalPort", "CancelToken", "Cancel", "NeverCancelled", "Host",
    # null implementations, so the engine runs hostless (M20)
    "NullLLM", "ScriptedLLM", "MemoryFileSystem", "LocalFileSystem",
    "SubprocessExec", "MemoryStorage", "SilentEvents", "RecordingEvents",
    "AutoApprove", "DenyAll",
    # engine
    "langs", "diagnostics", "guard", "runner", "patcher", "textio",
    "context", "redact", "review", "Budget", "RedactionReport",
    "CodeMap", "Journal", "Loop", "LoopConfig", "Patcher",
    "Transaction", "Planner", "Persona", "PromptBuilder",
    "detect_commentary", "strip_commentary", "strip_think",
    "Session", "SessionConfig", "RemoteGate", "make_provider",
    "available_providers",
]
