# chinchilla-llm-bench

Concurrency benchmark for a remote **vLLM** server, with a live **agent-swarm**
terminal UI: every concurrent request gets its own card showing what it is
doing (prompt, streaming output, token count) while the run is in progress.

```
chinchilla-bench --base-url http://127.0.0.1:1235/v1 \
                 --model qwen3.8-27b-nvfp4 --pp 200 --tg 128 --c 1 2 3 4
```

## What it measures

For every concurrency level `c` in `--c`, two tests are run (each with `--n`
requests, at most `c` in flight at once):

| test | what it does |
|------|--------------|
| `pp<pp>` (prefill) | prompt of `pp` tokens, `max_tokens=1` → measures prompt-processing speed and time-to-first-token |
| `tg<tg>` (decode)  | short prompt, generate `tg` tokens → measures generation throughput |

### Report columns

| column | meaning |
|--------|---------|
| `t/s (total)` | aggregate tokens/s measured during each request's active window (all agents combined), mean ± std over requests. For `c=1` this equals `t/s (req)` |
| `t/s (req)` | per-request throughput (pp: prompt tokens / ttfr · tg: generated tokens / duration), mean ± std |
| `peak t/s` | peak of the aggregate token stream during each request's window, mean ± std (tg only) |
| `peak t/s (req)` | peak per-request rate over a 0.5 s sliding window, mean ± std (tg only) |
| `ttfr (ms)` | time to first token, mean ± std (pp only) |
| `est_ppt (ms)` | estimated pure prompt-processing time = `ttfr − baseline`, where baseline is the ttfr of a 1-token prompt measured before the sweep (pp only) |
| `e2e_ttft (ms)` | end-to-end time to first token as seen by the client (pp only) |

Prompt length is made exact using vLLM's native `POST /tokenize` endpoint when
available (falls back to a ~1.3 tok/word estimate if the endpoint is missing).

## The live UI

While running, the terminal shows a swarm grid — one card per agent:

- **border color** = status: green working · cyan done · red error · dim idle
- **title** = agent id + role (`A3  tester`)
- **subtitle** = status + tokens generated so far
- **body** = the prompt (dim) and the live streaming output
- **top bar** = current phase, active concurrency level, `AGENTS LIVE`,
  `TOKENS/SEC`, `TOKENS TOTAL`, `ELAPSED`

Keys while running: `q` or `s` = stop (reports what finished), `l` = toggle loop.

## Setup (uv)

```bash
uv sync          # creates .venv and installs deps
uv run chinchilla-bench --help
```

## Usage

```bash
uv run chinchilla-bench \
  --base-url http://127.0.0.1:1235/v1 \
  --model qwen3.8-27b-nvfp4 \
  --pp 200 --tg 128 \
  --c 1 2 3 4
```

| flag | default | description |
|------|---------|-------------|
| `--base-url` | — | OpenAI-compatible base URL (required) |
| `--model` | — | served model name (required) |
| `--pp` | `200` | prompt tokens for the prefill test |
| `--tg` | `128` | tokens to generate for the decode test |
| `--c` | — | concurrency levels to sweep, e.g. `--c 1 2 3 4` (required) |
| `--n` | `5` | requests per test |
| `--temperature` | `0.0` | sampling temperature |
| `--timeout` | `300` | per-request timeout (s) |
| `--seed` | `1337` | prompt-generation seed |
| `--loop` | off | repeat the whole sweep until stopped |
| `--output` | `model_result_<model>.txt` | where to write the markdown report |
| `--no-ui` | off | plain progress lines instead of the swarm UI (good for logs/CI) |

The run ends with the settings summary and the results table printed to the
terminal, plus a markdown copy written to `model_result_<model>.txt`:

```
| model             |       test |       t/s (total) |         t/s (req) |      peak t/s |   peak t/s (req) |       ttfr (ms) |    est_ppt (ms) |   e2e_ttft (ms) |
|:------------------|-----------:|------------------:|------------------:|--------------:|-----------------:|----------------:|----------------:|----------------:|
| qwen3.8-27b-nvfp4 | pp200 (c1) | 4448.64 ± 1603.51 | 4448.64 ± 1603.51 |               |                  |  144.27 ± 19.48 |   40.72 ± 19.48 |  144.27 ± 19.48 |
| qwen3.8-27b-nvfp4 | tg128 (c1) |      66.83 ± 5.13 |      66.83 ± 5.13 |  71.00 ± 4.24 |     71.00 ± 4.24 |                 |                 |                 |
```

## Project layout

```
src/chinchilla_llm_bench/
├── cli.py      # argparse CLI + entry point
├── config.py   # BenchConfig (all settings)
├── client.py   # OpenAI-compatible streaming client + /tokenize probe
├── prompt.py   # builds prompts of ~pp tokens (exact via vLLM tokenizer)
├── agent.py    # one worker thread = one swarm card
├── ui.py       # rich Live swarm grid + top bar + key handling
├── runner.py   # sweep orchestration: preflight, baseline, pp/tg phases
├── stats.py    # mean±std, sliding-window peak, phase aggregation
└── report.py   # rich table (console) + markdown table (file)
tests/          # pytest suite with a mock vLLM server
```

## Tests

```bash
uv run pytest
```

The suite runs a real HTTP server (`tests/mock_server.py`) that speaks just
enough of the vLLM/OpenAI API (SSE streaming, usage chunks, `/models`,
`/tokenize`) to exercise the client, the full sweep, and the report end to end.

## Notes

- Any OpenAI-compatible endpoint works, but `est_ppt` and exact prompt sizing
  are best with vLLM (it serves `/tokenize`).
- `--n` should be ≥ the highest `--c` so every agent stays busy.
- Results depend on server load; run the sweep twice to check stability.
