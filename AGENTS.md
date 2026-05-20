# NVIDIA NIM Agent Extension

Pi coding agent extension registering NVIDIA NIM (`build.nvidia.com`) as a custom LLM provider. Single-file TypeScript (`index.ts`), no build step.

## Commands

```bash
npm run check   # tsc --noEmit (typecheck only, no emit)
npm test        # node --test test/*.test.mjs
```

Run `check` before `test`. No lint or formatter configured.

**Note**: `npm run check` requires `typescript` installed (`npm i -D typescript`). It is not listed in devDependencies — the repo relies on pi-coding-agent pulling it in transitively or a global install.

## Architecture

- **Entry point**: `index.ts` — default export receives `ExtensionAPI` from pi
- **Provider name**: `nvidia-nim`, API type `openai-completions`
- **Base URL**: `https://integrate.api.nvidia.com/v1`
- **No build**: `"type": "module"`, `tsconfig.json` has `noEmit: true`. Pi loads `.ts` directly.

## Thinking / Reasoning Gotchas

NVIDIA NIM does **not** support the standard OpenAI `reasoning_effort` parameter. Thinking is enabled via `chat_template_kwargs`, which differs per model family:

| Family | enable kwargs |
|--------|--------------|
| DeepSeek V4 | `{ thinking: true, reasoning_effort: "high"\|"max" }` |
| DeepSeek V3.x, R1 distills | `{ thinking: true }` |
| GLM-5, GLM-4.7 | `{ enable_thinking: true, clear_thinking: false }` |
| Kimi K2.6 | `{ thinking: true }` + `sendReasoningEffort: true` |
| Qwen3, QwQ | `{ enable_thinking: true }` |

- `minimal` → `low`, `xhigh` → `high` (or `max` for DeepSeek V4)
- GLM models think **by default** — must explicitly disable when thinking is off
- **Do not send `developer` role with `chat_template_kwargs`** — causes 500 errors on NIM. The extension sets `supportsDeveloperRole: false` globally.

## API Key Resolution Order

1. pi's resolved credential (CLI override, `auth.json` literal, or shell command starting with `!`)
2. `NVIDIA_NIM_API_KEY` env var
3. `NVIDIA_API_KEY` env var (fallback)

Keys starting with `!` in `auth.json` are executed as shell commands. Empty resolved values throw before any fetch.

## Model Discovery

On `session_start`, the extension fetches `GET /v1/models` and re-registers any new models. **Critical**: discovered models must use `ctx.modelRegistry.registerProvider()`, NOT `pi.registerProvider()`. The latter only works during initial extension load.

Discovery is silenced on 429 (rate limit) or missing credentials — shows a warning only on 401/403.

## Mistral Model Compat

Mistral models on NIM need extra flags: `requiresToolResultName`, `requiresThinkingAsText`, `requiresMistralToolIds`.

## Content Normalization

The stream normalizes `[{type:"text", text:"..."}]` content arrays to plain strings. Many older/smaller NIM backends reject the array format.

## Auth Tests

Tests in `test/nvidia-nim-auth.test.mjs` use `node:test` and mock `globalThis.fetch`. They set `PI_CODING_AGENT_DIR` to a temp directory to isolate `auth.json`. Delete `NVIDIA_NIM_API_KEY`/`NVIDIA_API_KEY` from `process.env` at the top of the file to prevent host credentials from leaking into assertions.

## CI

Push to `main` triggers `.github/workflows/publish.yml`: auto-bumps patch if version already exists on npm, then publishes. Requires `NPM_TOKEN` secret.
