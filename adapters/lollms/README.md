# A LoLLMs adapter — deliberately not built here

This directory is a placeholder, and that is a decision rather than an
omission (§8).

**ParisNeo gets the specification and decides for himself.** Whether LoLLMs
wants this as a personality, an MCP plugin, a service, or not at all is not
this project's call, and a half-guessed adapter written by someone who does
not use LoLLMs would be worse than none: it would be an obligation to
maintain, and a wrong shape to argue with.

## What has already been done for you

Everything that makes the adapter easy is in the core, and it is the reason
the core looks the way it does:

- **`cognitive_coder/**` imports no GUI and no host, ever.** Not
  conditionally, not inside functions. There is a test that walks the tree
  and fails the build otherwise. FastAPI, Starlette and Flask are on the
  forbidden list alongside Qt — so nothing here can quietly assume a web
  framework either.
- **Six small Protocols** are the entire integration surface. Implement them
  structurally; there is nothing to inherit and nothing of ours to import.
- **`tests/port_conformance.py` is shipped as part of the product.** Run it
  against your implementations and you will know whether they are right
  without reading the core. That is what it is for.
- **Zero required runtime dependencies.** No version arguments with whatever
  LoLLMs already has installed.
- **The vendoring path is tested**: `sys.path`-insert the repo and import it,
  with no install and no package metadata. A git submodule works.

## The shape it would take

```python
from cognitive_coder import Host, Session, SessionConfig

class LollmsLLM:
    """Wrap the binding LoLLMs already has loaded. Do not load a second one."""
    def complete(self, messages, **kw): ...
    def stream(self, messages, **kw): ...
    def capabilities(self): ...
    def count_tokens(self, text): ...

class LollmsEvents:
    """Push events to the client — SSE, websocket, whatever LoLLMs uses."""
    def event(self, kind, message, data=None): ...

host = Host(llm=LollmsLLM(binding), fs=..., events=LollmsEvents(), ...)
Session(host, config=SessionConfig()).run(request)
```

The `EventPort` vocabulary is a closed set precisely so a web client can
switch on it without worrying about tomorrow's addition. See
`docs/PORTS.md`.

## The one thing worth reading first

`docs/EMBEDDING.md`, then `examples/tiny_host.py`. The example drives a whole
session with no model and no network, in about fifty lines, and every line of
it is a line a host author writes.

If something in the Ports is awkward for a FastAPI host, that is worth
knowing before 1.0 freezes the surface — please say so.
