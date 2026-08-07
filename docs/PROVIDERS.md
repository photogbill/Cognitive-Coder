# Providers

## The rule that comes before all the others

**Offline is the default. The network is an explicit, visible choice.**

No remote call happens unless you have enabled a remote provider *for this
session*, and remote mode is shown whenever it is on. There is no
"helpfully falls back to the cloud", and there is no environment variable
that turns it on. **A key is not consent.**

This exists because one host of this engine is an air-gapped, zero-telemetry
tool used where an outbound connection is a safety problem rather than an
inconvenience. A coding module that silently phoned home would break that
promise on its host's behalf.

Enforcement, not intention:

- `make_provider()` refuses to construct anything remote without a
  `RemoteGate` that has been explicitly enabled for that specific provider;
- `ApprovalPort.approve_remote` is called before the first remote call, with
  the byte count;
- a `remote` event fires and keeps firing, so a host can keep an indicator on
  screen;
- a test drives a **full scripted session with every socket entry point
  monkeypatched to raise**, and fails if anything so much as tries.

---

## What is built

### `openai_compatible` — the one that matters

One adapter covers **llama.cpp server, Ollama, LM Studio, vLLM and LiteLLM**.
This is the highest-value provider in the project: it is local, and it is
what most self-hosters actually run.

```python
from cognitive_coder import make_provider

llm = make_provider("openai_compatible",
                    base_url="http://127.0.0.1:8080",
                    model="devstral-small-2-24b-instruct")
```

Built on `urllib` from the standard library — no `requests`, no `httpx`, no
argument about versions when you embed this next to something else.

`detect()` probes the conventional ports and tells you which are answering,
so a host can offer a list instead of a text box:

| Server | Port |
|---|---|
| llama.cpp server | 8080 |
| Ollama | 11434 |
| LM Studio | 1234 |
| vLLM | 8000 |
| LiteLLM | 4000 |

**A local endpoint is not a remote one.** `127.0.0.1`, `localhost`, and
private ranges like `192.168.x.x` and `10.x.x.x` are LAN addresses; talking
to them is not "the network" in the sense that matters, and the provider
reports `is_remote=False` for them. A public host **is** remote, says so, and
needs the gate. Treating a local llama.cpp server as remote would make the
warning meaningless through overuse — which is how a safety indicator stops
being read.

Where the server offers more, it is used: llama.cpp's `/props` reveals GBNF
grammar support, `/tokenize` gives exact token counts, and its `timings`
field gives real prompt-processing time.

### `local_llamacpp` — a GGUF in-process

```python
llm = make_provider("local_llamacpp",
                    model_path="models/Devstral-Small-2-24B-Q4_K_M.gguf",
                    n_ctx=16384, type_k="q8_0", type_v="q8_0")
```

Needs the `[llamacpp]` extra. `llama_cpp` is imported inside the constructor,
never at module level, so a machine without it gets a **sentence explaining
what is missing and what it costs** rather than an ImportError from
`import cognitive_coder`.

Usually you do not want this. If your host already has a model loaded, wrap
it in your own `LLMPort` instead — cheaper, and correct, because the host
owns loading and unloading.

`save_state()` / `load_state()` are exposed for hosts that want to preserve a
KV prefix across their own model-swap button. Offered, never driven.

---

## What is not built

Remote providers — Anthropic, Google Gemini, Mistral, OpenRouter, OpenAI —
are specified and **not present in this version.** They arrive with
`redact.py` and budget enforcement, which are the things that make them safe
to have.

`available_providers()` names them and says why they are absent, rather than
offering a name that fails at call time:

```python
{'openai_compatible': {'built': True,  'remote': False, ...},
 'local_llamacpp':    {'built': True,  'remote': False, ...},
 'anthropic':         {'built': False, 'remote': True,
                       'description': 'remote provider — arrives with '
                                      'redaction and budgets (phase 8)'}}
```

**The gate they must come through was built first, on purpose**, so that
adding a provider later cannot accidentally route around it. When they land,
every one of them must:

1. be **disabled unless explicitly enabled for the session** — no env-var
   auto-detection;
2. route **every outbound message** through redaction first — the initial
   payload, every subsequent turn, and **every tool-result message**, because
   a file slice returned to a remote model is outbound context like any
   other;
3. call `approve_remote` before the first call of a session;
4. enforce a token and spend **budget that halts** rather than silently
   continuing;
5. record provider, model, token counts and cost in the journal;
6. keep a persistent "REMOTE MODE — data leaves this machine" indicator up.

**Keys never touch the journal, the codemap, or any log.**

A note for the implementer of phase 8: OpenAI is in the provider list for the
benefit of users who want it. The project owner does not use it and this is a
settled preference — implement it, don't advocate it, and don't make it a
default anywhere.

---

## Model notes

The engine is calibrated against **Devstral Small 2 24B**
(`mistralai/Devstral-Small-2-24B-Instruct-2512`): 256k context, Apache-2.0,
native tool calling, multimodal, recommended temperature **0.15**. Nothing
about the design is specific to it — where the document reasons about "a 7B",
it is describing the harder case this design also survives.

**Layer the model's own system prompt; do not replace it.** Devstral ships
`CHAT_SYSTEM_PROMPT.txt` and was tuned with it; replacing it wholesale
discards that tuning. Pass it as `SessionConfig.model_system_prompt` and the
engine adds project conventions, execution constraints and the output
contract on top.

**Reasoning models:** `<think>` blocks are stripped before model output is
used for anything. Reasoning in a source file is not a style problem, it is a
broken file. Budget for the tax too — `context.measure_budget(llm,
reasoning=True)` reserves room for thinking that produces no answer.

**Tool calling** is used natively where `capabilities().supports_tools` is
true. Where it is not, the text-marker fallback takes over: several syntaxes
accepted, a hard cap of three lookups per generation, and the syntax
corrected once and only once. The fallback also **forces an epoch per write**
— without live tools a lagging architecture summary has no safety net, so it
is not allowed to lag.

**Fill-in-the-middle**, where the model supports it, is structurally
incapable of touching code outside the hole — which is the clean answer to a
model "helpfully" rewriting code it was not asked to touch.
`supports_fim()` recognises the usual code models.
