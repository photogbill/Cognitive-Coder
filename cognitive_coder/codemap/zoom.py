# SPDX-License-Identifier: Apache-2.0
"""Semantic zoom — and the reconciliation of freshness with prefix caching.

TWO PROBLEMS, ONE SEAM.

**Problem one (§6.7): what does the model need to see about the rest of the
codebase?** Not everything. Immediate dependencies of the file being written
get full signatures, parameters and docstrings; distant architecture gets
class names and paths, or nothing. A dependency costs ~30 tokens as an
interface and ~800 as a file (F9), and that ratio is what makes multi-file
work possible on a local model at all.

**Problem two (G.7): llama.cpp caches the KV state of a prompt PREFIX.** If
the beginning of the prompt is byte-identical to last time, those tokens are
not reprocessed — the difference between 3 seconds and minutes, on every call.
But §6.7 demands a fresh codemap and every write updates it, which taken
naively says "rebuild the prefix after every write and never benefit from the
cache".

**The seam is the same seam.** The zoom's split between distant architecture
and immediate dependencies is exactly the split between slow-changing and
fast-changing content:

    CACHED PREFIX  1. system prompt / persona          never changes
                   2. project conventions, style        never changes
                   3. DISTANT architecture, low-res     changes by EPOCH
    ─────────────── cache boundary ───────────────
    VOLATILE TAIL  4. immediate dependencies, high-res  (F9 interfaces)
                   5. same-codebase examples            (F3)
                   6. THE TASK
                   7. staleness note + last diagnostics
                   8. output contract                   short; last for recency

And the resolution that makes it honest: **the injected map is a HINT; the
tool call is GROUND TRUTH.** `search_codemap()` queries live SQLite every
time, so the queryable interface is never stale (M30), and staleness in the
injected text costs at most one extra tool call — never a wrong answer.

Restated as the rule everything here obeys: *the query interface must never be
stale; the cached hint may lag, provided the model can always check.*

**Never put a timestamp, a session id, or a randomised preamble anywhere in
the prefix.** One varying token at position 40 silently discards 30k tokens of
cached work, and nothing will tell you it happened except `prompt_ms` in the
journal (M55).
"""

from __future__ import annotations

from typing import Any

from ..types import CodemapStats

# How many files may change before the architecture snapshot is rebuilt
# (G.7.2). Five is a starting point to measure, not a law.
EPOCH_FILE_THRESHOLD = 5


def architecture_prefix(store: Any, *, target: str = "",
                        max_files: int = 60) -> str:
    """The LOW-RESOLUTION architecture block. Goes in the CACHED PREFIX.

    Stable for the whole session between epochs, and therefore byte-identical
    call to call — which is the entire point. Nothing time-varying,
    nothing target-varying beyond the epoch, nothing sorted by anything but a
    stable key.

    ``target`` is accepted and deliberately unused for ordering: sorting the
    architecture by relevance to the current file would change the prefix
    bytes every time the target changed, discarding the cache to save a few
    hundred tokens. The high-resolution part is where relevance belongs.
    """
    files = store.files()
    lines = ["# PROJECT ARCHITECTURE (low resolution — names and paths only)"]
    shown = sorted(files, key=lambda f: f["path"])[:max_files]
    for row in shown:
        syms = store.symbols_in(row["path"])
        heads = [s["name"] for s in syms
                 if s["kind"] in ("class", "struct", "trait", "interface")]
        funcs = [s["name"] for s in syms if s["kind"] == "function"]
        summary = ", ".join(heads[:4] + funcs[:4]) or "(no symbols indexed)"
        approx = " ~approx" if row.get("approximate") else ""
        lines.append(f"  {row['path']}{approx}: {summary}")
    if len(files) > len(shown):
        lines.append(f"  … and {len(files) - len(shown)} more files not "
                     f"listed here. Use search_codemap to look any of them "
                     f"up.")
    return "\n".join(lines)


def dependency_interfaces(store: Any, target: str, *,
                          budget_tokens: int = 1200,
                          count_tokens=None) -> str:
    """HIGH-RESOLUTION interfaces for what `target` depends on. VOLATILE TAIL.

    F9, and on the target machine the highest-value item in that appendix.
    The model cannot hallucinate around a signature it has been handed
    verbatim (D4), and the cost of handing it over is small and predictable.
    """
    deps = _direct_dependencies(store, target)
    if not deps:
        return ""
    lines = ["# INTERFACES YOU MAY CALL (exact signatures — use these, do "
             "not guess)"]
    used = 0
    dropped: list[str] = []
    for path in deps:
        block = [f"## {path}"]
        for s in store.symbols_in(path):
            if s["name"].split(".")[-1].startswith("_"):
                continue
            doc = f"  # {s['docstring']}" if s["docstring"] else ""
            block.append(f"{s['signature'] or s['name']}{doc}")
        text = "\n".join(block)
        cost = count_tokens(text) if count_tokens else len(text) // 4
        if used + cost > budget_tokens and used:
            dropped.append(path)
            continue
        lines.append(text)
        used += cost
    if dropped:
        lines.append(f"# NOT SHOWN (no room): {', '.join(dropped)} — call "
                     f"search_codemap if you need them.")
    return "\n".join(lines)


def _direct_dependencies(store: Any, target: str) -> list[str]:
    """Files the target CALLS INTO. Callees only, deliberately.

    Not callers. The block this feeds is headed "interfaces you may call",
    and a caller is not something you call — putting them here spends the
    F9 budget on the wrong files and teaches the model that its dependants
    are its dependencies. Callers are a different question with a different
    answer: `store.blast_radius`, asked when a signature changes.
    """
    out: list[str] = []
    for row in store.db.execute(
            "SELECT DISTINCT f2.path AS path "
            "FROM files f1 "
            "JOIN symbols s1 ON s1.file_id = f1.id "
            "JOIN edges e ON e.src_symbol_id = s1.id "
            "JOIN symbols s2 ON s2.id = e.dst_symbol_id "
            "JOIN files f2 ON f2.id = s2.file_id "
            "WHERE f1.path = ? AND f2.path != ?", (target, target)):
        if row["path"] not in out:
            out.append(row["path"])
    return out


def similar_examples(store: Any, target: str, *, limit: int = 2) -> str:
    """One or two functions from THIS codebase that look like the job (F3).

    Local models have weak priors on your specific idioms — your error
    handling, your logging, your naming. Two examples buy real consistency at
    24B, which is good enough to imitate style faithfully.

    A different use of the codemap from dependency injection, and both should
    run: **dependencies tell it what EXISTS; examples tell it what GOOD LOOKS
    LIKE HERE.** Similarity stays dumb — same directory, similar name — which
    is enough and costs nothing.
    """
    folder = target.replace("\\", "/").rsplit("/", 1)[0] if "/" in target \
        else ""
    picks: list[dict] = []
    for row in store.files():
        path = row["path"]
        if path == target:
            continue
        same_dir = path.startswith(folder + "/") if folder else True
        if not same_dir:
            continue
        for s in store.symbols_in(path):
            if s["kind"] == "function" and s["docstring"]:
                picks.append({**s, "path": path})
        if len(picks) >= limit * 3:
            break
    if not picks:
        return ""
    lines = ["# HOW THIS CODEBASE DOES THINGS (existing code, for style — "
             "not part of the task)"]
    for s in picks[:limit]:
        lines.append(f"# from {s['path']}")
        lines.append(f"{s['signature']}  # {s['docstring']}")
    return "\n".join(lines)


def staleness_note(store: Any) -> str:
    """The lag declaration. **Goes in the TAIL, never the prefix** (G.7.3).

    A note in the prefix would change the prefix bytes and invalidate the
    very cache it describes. The tail is reprocessed anyway, so putting it
    there is free.
    """
    changed = store.changed_since_epoch()
    epoch = store.epoch
    if not changed:
        return (f"Architecture snapshot: epoch {epoch}, up to date.")
    listed = ", ".join(f"`{p}`" for p in changed[:6])
    more = "" if len(changed) <= 6 else f" and {len(changed) - 6} more"
    return (f"Architecture snapshot: epoch {epoch}. "
            f"{len(changed)} file(s) have changed since it was taken "
            f"({listed}{more}). For anything you are about to touch, call "
            f"search_codemap rather than trusting the summary above.")


def should_bump_epoch(store: Any, *, target: str = "",
                      replanned: bool = False, model_changed: bool = False,
                      operator_asked: bool = False) -> tuple[bool, str]:
    """Whether to rebuild the cached prefix, and why (G.7.2).

    The reasons are a closed list on purpose. Bumping for anything else means
    paying 30–60 s of prompt reprocessing for a benefit nobody measured.
    """
    if operator_asked:
        return True, "the operator asked for a refresh"
    if model_changed:
        # A model change killed the KV cache and the prompt-prefix state
        # anyway (§0.1 consequence 2) — there is nothing left to preserve.
        return True, "the loaded model changed, so the cached prefix died"
    if replanned:
        return True, "the plan was revised"
    changed = store.changed_since_epoch()
    if len(changed) >= EPOCH_FILE_THRESHOLD:
        return True, (f"{len(changed)} files have changed since the last "
                      f"snapshot")
    if target and target in changed:
        return True, (f"the file being worked on ({target}) has changed "
                      f"since the snapshot")
    return False, ""


def stats_line(stats: CodemapStats) -> str:
    """The resolution rate, said out loud (§6.7).

    A call graph that silently drops what it could not bind looks complete
    and isn't. This number is how anyone finds out.
    """
    line = stats.one_line()
    if stats.unresolved and stats.resolution_rate < 0.7:
        line += ("  — a large share of calls could not be bound to a "
                 "definition, so blast-radius answers are incomplete")
    return line


def generate_architecture_context(store: Any, target: str, *,
                                  max_tokens: int = 4096,
                                  count_tokens=None) -> str:
    """The whole zoomed view for one file, under a budget, declaring omissions.

    This is the §6.7 acceptance function: the context for a chosen file must
    fit a 4,096-token budget and NAME WHAT IT DROPPED. The omission block is
    not politeness — a model that doesn't know something was withheld will
    invent its contents.
    """
    def cost(text: str) -> int:
        return count_tokens(text) if count_tokens else max(1, len(text) // 4)

    dropped: list[str] = []
    pieces: list[str] = []
    remaining = max_tokens

    interfaces = dependency_interfaces(store, target,
                                       budget_tokens=int(max_tokens * 0.45),
                                       count_tokens=count_tokens)
    if interfaces:
        pieces.append(interfaces)
        remaining -= cost(interfaces)

    examples = similar_examples(store, target)
    if examples and cost(examples) < remaining * 0.3:
        pieces.append(examples)
        remaining -= cost(examples)
    elif examples:
        dropped.append("style examples from this codebase")

    arch = architecture_prefix(store, target=target)
    if cost(arch) <= remaining:
        pieces.insert(0, arch)
        remaining -= cost(arch)
    else:
        trimmed = "\n".join(arch.splitlines()[:max(3, remaining // 12)])
        pieces.insert(0, trimmed + "\n  … (architecture list truncated)")
        dropped.append("the full file list — use search_codemap to look "
                       "anything up")

    pieces.append(staleness_note(store))
    body = "\n\n".join(p for p in pieces if p)
    body += "\n\n--- NOT INCLUDED ---\n"
    if dropped:
        body += ("Left out to fit the context window: "
                 + "; ".join(dropped)
                 + ".\nCall search_codemap or read_slice instead of guessing "
                   "at anything above.\n")
    else:
        body += ("Nothing was left out of the architecture summary.\n")
    if count_tokens is None:
        body += ("Sizes here were estimated from character counts, not "
                 "counted exactly.\n")
    return body
