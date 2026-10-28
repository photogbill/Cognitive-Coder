# FIM mode — holes for an ordinary code model

*Built 2026-10-27. Modules: `cognitive_coder/fim.py` (the rules),
`loop.Loop._fim_generate` (the repair), `session.Session._probe_fim` (the
probe), `ports.FimLLM` / `types.FimRequest` / `types.FimResult` (the
contract), `providers/openai_compatible.py` and `providers/local_llamacpp.py`
(two hosts), ATK's `ccoder_host.ATKLLM.fim_fill` (a third).*

## What it is

Fill-in-the-middle: the model is given the code before a hole and the
code after it, in the FIM format it was trained on, and writes only what
goes between. It structurally cannot touch code outside the hole — the
cure for D6 (the surrounding function "helpfully" rewritten) that Appendix
G.4 called "PROVEN, and underused", and that every provider advertised
(`supports_fim`) while nothing read it.

It is the autoregressive cousin of the canvas (`docs/DIFFUSION_MODE.md`):
the same decision about WHERE the hole goes, the same cleaning and splicing
of what comes back, a different model behind it. Qwen3-Coder supports FIM
in every version; Qwen2.5-Coder, DeepSeek-Coder, StarCoder, CodeGemma and
Codestral have it too. For all of them a located error used to be repaired
by the whole file (~2,500 tokens, the dropped-code guard watching) or a
narrowed edit (one definition, a fence, a contract to follow).

| | whole file | narrowed edit | canvas | FIM hole |
|---|---|---|---|---|
| what comes back | the file | named definitions | the hole | the hole |
| can drop code elsewhere | yes (guarded) | no | no | no |
| needs a contract the model must follow | yes | yes | no | no |
| size limit | the reply budget | — | 256 tokens | the context |
| model | any | any | DiffusionGemma | one with FIM tokens |
| tokens for a 20-line body | ~2,500 | ~250 | ~200 | ~200 |

## The probe decides, not the name

`supports_fim` is a hint from the model's name. Mistral documents FIM for
Codestral and not for Devstral; a merge may have the tokens and not the
training. So at the model check (and after every swap) the session asks
the host to fill ONE known hole:

```python
def add(a, b):
    """Return the sum of a and b."""
    ▮                                     ← the hole

def sub(a, b):
    """Return a minus b."""
    return a - b
```

`return a + b` (or `b + a`), parsing when spliced, under seven lines, is
the pass. Anything else — an error from the host ("the model has no FIM
tokens"), `return a - b`, prose, a fill that does not parse — turns the
mode off for the session with the reason in BUILD_LOG.txt (`fim off — …`)
and the journal (`fim`, `kind: probe`). A canvas model is never probed:
its holes are canvases.

`SessionConfig.fim`: `"auto"` (probe a real port), `"on"` (probe even a
scripted double — the tests), `"off"`.

## The repair

For a repair attempt of a Python file that parses, before a narrowed edit
and before the whole file (never on the ladder's fresh start):

1. **Where** — the body of the function the located error is in (up to
   `fim.BODY_MAX_LINES`, 120; longer, a window around the line); when the
   failure names no line of this file, the body of a function the failing
   test calls; then the body the engine doubts most (`Loop.doubt`, which
   the resample's disagreement map fills for an ordinary model — the
   canvas's confidence readout for a diffusion one).
2. **The prompt** — the file up to the hole is the prefix, ending in a
   short comment at the hole's indentation: the error's one line, up to
   four rows of the values at the failure, and "write the corrected body
   below; the signature above and everything after it stay as they are".
   A FIM model takes no instruction; a comment right before the code it
   writes is the one place it reads one. The comment is in the prompt
   only. The file after the hole is the suffix. The `# INTERFACES YOU MAY
   CALL` block goes along as a repo-level chunk for a model trained on
   that format (Qwen's `<|repo_name|>` / `<|file_sep|>`; llama-server's
   `input_extra`).
3. **Candidates** — `fim_candidates` (2): the first at the attempt's
   temperature and seed, the second warmer with a new seed, and the
   second is asked only when the first came back unusable.
4. **Judging** — `canvas.clean_fill` in the "open" style (a reply that
   runs on past the hole is cut where it dedents out of the body, when
   the cut parses), echoed FIM tokens stripped, the spliced file must
   parse, the old lines must not come back, no name may be defined twice.
5. **The same hole twice** — the same function with the same failure's
   words after a fill says the fault is elsewhere; that attempt goes to
   the other paths.

The journal gets a `fim` event per repair (the hole, every candidate with
its tokens and timings, which was kept) and the `generate` event carries
`via: fim`; BUILD_LOG.txt gets the hint, every fill and the hole as kept.
The host gets a `status` event with the fill for a Live tab.

## The contract

```python
class FimLLM(Protocol):
    def fim_fill(self, request: FimRequest, *, cancel=None) -> FimResult: ...
```

`FimRequest`: `prefix`, `suffix`, `path`, `extra` ((path, text) chunks),
`max_tokens`, `temperature`, `seed`, `stop`, and the profile's `order`
(`psm` — prefix, suffix, middle — or `spm`, Mistral's suffix-first) and
`tokens` (the token texts) for a host that builds the raw prompt itself.
`FimResult`: `text` (the middle only), `finish_reason`, `error`, the token
counts and timings. Never raises; a model without FIM tokens is a sentence.

Hosts in this repo:

* `openai_compatible` — llama-server's `/infill` (it knows its model's
  own FIM tokens and refuses a model without them) when the server
  answered `/props`; otherwise `/v1/completions` with `suffix` (vLLM,
  Ollama). Local only.
* `local_llamacpp` — the prompt from TOKEN IDS: the vocabulary's FIM
  tokens where the binding exposes them (`llama_vocab_fim_pre` …), else
  the profile's token texts tokenized as specials — each must be ONE
  token, or the model has none — around the prefix and suffix tokenized
  as plain text, in the profile's order. `fim_prompt_ids` is the shared
  builder; ATK's `ATKLLM.fim_fill` uses it under the engine's lock
  (`LLMEngine.run_on_model`).

## What it does not do

* Write a new file body by body: the first attempt of a new file is where
  a whole-file reply does well, and stays whole-file.
* Replace the canvas: a diffusion model's hole sees both sides inside the
  canvas and reads its own confidence out; FIM has neither. A canvas model
  keeps diffusion mode.
* Trust a name. The probe is the whole of the evidence.
