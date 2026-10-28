# Speed — where the minutes go, and the one lever the engine cannot pull

*2026-10-27. The engine measures speed (the probe at the model check, the
`generate` lines' prefill and decode, `ccoder postmortem`) and reduces
the tokens a build needs (narrowed edits, FIM holes, the dropped-code
guard, resampling before repair). What it cannot do is make the model
decode faster. That is the host's inference server, and this page is the
engine's advice to it.*

## What the journals say

On the 2026-10-16 Devstral build **76 % of the wall clock was the model
decoding** at 3–5 tok/s — a 24B Q4_K_M spilling out of a 16 GB card into
system RAM. The speed probe now says so at the start of every build
(`SessionConfig.speed_probe`; under 8 tok/s is the warning). The first
answer is always the same: a quantization, a context length and a KV
cache type that fit in VRAM. A Q4 24B at 16k context with a q8_0 KV cache
fits a 16 GB card; the same model at 32k with an f16 cache does not.

## Draftless speculative decoding

Most of what a build asks a model to write is text the model has just
been shown: a whole-file repair copies the file with one function changed;
a continuation copies the tail it was given; a narrowed edit copies a
signature and a docstring. Speculative decoding guesses the next tokens
and lets the model verify several at once; the **n-gram** modes guess
them from the prompt itself, so they need no draft model, no second GGUF
and no VRAM — and they are at their best exactly on rewrites of code the
model was shown.

llama.cpp's own documentation describes the draftless modes and says of
`ngram-simple` that it "fits rewriting source code"
([docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)).

```
llama-server -m model.gguf -c 16384 -fa on --jinja \
             --spec-type ngram-simple
```

`--spec-default` turns on `ngram-mod`, the mode whose hash pool is shared
across the server's slots. The speed probe and the `generate` lines show
the before and after; the cost of a bad guess is a few verified tokens
thrown away, never a different answer — the output distribution is the
model's own.

The engine's `openai_compatible` provider sends `cache_prompt: true` on
every request, so a server that keeps the prefix also skips the prefill
of everything the prompt shares with the last one (the cached prefix is
what `prefix_for` in `personas.py` is ordered for).

## ATK's in-process binding

ATK runs chat models through llama-cpp-python, which has its own
prompt-lookup drafting (`LlamaPromptLookupDecoding`); ATK's `_draft_guard`
turns it off on the binding versions whose logits buffer cannot hold it
(the `could not broadcast input array` failure of 2026-09-26). Until a
binding sizes that buffer correctly, the draftless modes are a
`llama-server` feature: ATK already runs a server process for
DiffusionGemma, and a chat model served the same way, pointed at through
`ccoder build --url http://127.0.0.1:8080`, gets `--spec-type` and the
`/infill` endpoint (FIM mode) for free.

## What the engine does on its side

| | |
|---|---|
| the speed probe | tok/s and prefill after the model check and every swap; a warning under 8 tok/s |
| `ccoder doctor` | names the draftless flag when a llama.cpp endpoint is answering |
| narrowed edits, FIM holes | a repair returns one definition or one body, not a file (`docs/NARROWED_EDITS.md`, `docs/FIM_MODE.md`) |
| resampling before repair | a second sample of the same prompt costs one decode; the prefix is already cached |
| the size rule | smaller files and functions make every one of the above cheaper (`sizing.py`) |
| `ccoder postmortem` | the journals read back, ranked by the minutes each harness fault cost |
