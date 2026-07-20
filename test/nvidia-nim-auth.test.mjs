import assert from "node:assert/strict";
import test from "node:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
	discoverAndLoadExtensions,
	ModelRegistry,
	ModelRuntime,
} from "@earendil-works/pi-coding-agent";
import extension from "../index.ts";

// Keep host NVIDIA credentials from affecting missing-env assertions.
delete process.env.NVIDIA_NIM_API_KEY;
delete process.env.NVIDIA_API_KEY;

const EXPLICIT_NVIDIA_NIM_API_KEY_REF = "$NVIDIA_NIM_API_KEY";

function createUserContext() {
	return {
		messages: [
			{
				role: "user",
				content: [{ type: "text", text: "hello" }],
			},
		],
		systemPrompt: "You are a test.",
		tools: [],
	};
}

async function createModelRuntime(authPath, tempDir) {
	return ModelRuntime.create({
		authPath,
		modelsPath: join(tempDir, "models.json"),
		modelsStorePath: join(tempDir, "models-store.json"),
		allowModelNetwork: false,
	});
}

async function createModelRegistry(authPath, tempDir) {
	return new ModelRegistry(await createModelRuntime(authPath, tempDir));
}

async function getModelRequestApiKey(modelRegistry, model) {
	const auth = await modelRegistry.getApiKeyAndHeaders(model);
	assert.ok(auth.ok, auth.ok ? undefined : auth.error);
	if (!auth.ok) {
		throw new Error(auth.error);
	}
	return auth.apiKey;
}

function usePiAgentDir(t, agentDir) {
	const originalAgentDir = process.env.PI_CODING_AGENT_DIR;
	process.env.PI_CODING_AGENT_DIR = agentDir;
	t.after(() => {
		if (originalAgentDir === undefined) {
			delete process.env.PI_CODING_AGENT_DIR;
		} else {
			process.env.PI_CODING_AGENT_DIR = originalAgentDir;
		}
	});
}

test("registers provider with an explicit NVIDIA_NIM_API_KEY env reference", () => {
	let providerConfig;

	extension({
		registerProvider(name, config) {
			assert.equal(name, "nvidia-nim");
			providerConfig = config;
		},
		on() {},
	});

	assert.equal(providerConfig?.apiKey, EXPLICIT_NVIDIA_NIM_API_KEY_REF);
});

test("loads through the current pi extension loader", async (t) => {
	const tempDir = mkdtempSync(join(tmpdir(), "pi-nvidia-nim-loader-"));
	t.after(() => rmSync(tempDir, { recursive: true, force: true }));

	const extensionPath = join(import.meta.dirname, "..", "index.ts");
	const result = await discoverAndLoadExtensions(
		[extensionPath],
		join(import.meta.dirname, ".."),
		tempDir,
	);

	assert.deepEqual(result.errors, []);
	assert.equal(result.extensions.length, 1);
	const registration = result.runtime.pendingProviderRegistrations.find(
		({ name }) => name === "nvidia-nim",
	);
	assert.equal(registration?.config.api, "openai-completions");
	assert.equal(registration?.config.models?.length, 44);
	const inkling = registration?.config.models?.find(({ id }) => id === "thinkingmachines/inkling");
	assert.deepEqual(
		inkling && {
			reasoning: inkling.reasoning,
			thinkingLevelMap: inkling.thinkingLevelMap,
			input: inkling.input,
			contextWindow: inkling.contextWindow,
			maxTokens: inkling.maxTokens,
			thinkingFormat: inkling.compat?.thinkingFormat,
			chatTemplateKwargs: inkling.compat?.chatTemplateKwargs,
		},
		{
			reasoning: true,
			thinkingLevelMap: { off: "none", xhigh: "max", max: "max" },
			input: ["text", "image"],
			contextWindow: 1_048_576,
			maxTokens: 16_384,
			thinkingFormat: "chat-template",
			chatTemplateKwargs: {
				reasoning_effort: { $var: "thinking.effort" },
			},
		},
	);
});

test("sends Inkling's full output budget and maps pi thinking levels to its chat template", async (t) => {
	const tempDir = mkdtempSync(join(tmpdir(), "pi-nvidia-nim-inkling-"));
	t.after(() => rmSync(tempDir, { recursive: true, force: true }));

	const modelRegistry = await createModelRegistry(join(tempDir, "auth.json"), tempDir);
	let providerConfig;
	extension({
		registerProvider(name, config) {
			providerConfig = config;
			modelRegistry.registerProvider(name, config);
		},
		on() {},
	});

	const model = modelRegistry.find("nvidia-nim", "thinkingmachines/inkling");
	assert.ok(model, "expected thinkingmachines/inkling to be registered");
	assert.ok(providerConfig?.streamSimple, "extension should register a custom streamSimple");

	const originalFetch = globalThis.fetch;
	const requestBodies = [];
	globalThis.fetch = async (_url, init) => {
		requestBodies.push(JSON.parse(String(init?.body)));
		return new Response("unauthorized", {
			status: 401,
			headers: { "content-type": "text/plain" },
		});
	};

	t.after(() => {
		globalThis.fetch = originalFetch;
	});

	for (const reasoning of [undefined, "minimal", "xhigh", "max"]) {
		const stream = providerConfig.streamSimple(model, createUserContext(), {
			apiKey: "ABC123",
			reasoning,
		});
		for await (const _event of stream) {
			// Consume the stream so the mocked request completes.
		}
	}

	assert.equal(
		requestBodies.some((body) => Object.hasOwn(body, "reasoning_effort")),
		false,
		"Inkling reasoning effort should only be sent through chat_template_kwargs",
	);

	assert.deepEqual(
		requestBodies.map(({ max_tokens, chat_template_kwargs }) => ({
			max_tokens,
			chat_template_kwargs,
		})),
		[
			{
				max_tokens: 16_384,
				chat_template_kwargs: { reasoning_effort: "none" },
			},
			{
				max_tokens: 16_384,
				chat_template_kwargs: { reasoning_effort: "minimal" },
			},
			{
				max_tokens: 16_384,
				chat_template_kwargs: { reasoning_effort: "max" },
			},
			{
				max_tokens: 16_384,
				chat_template_kwargs: { reasoning_effort: "max" },
			},
		],
	);
});

test("registers MiniMax M3 with native reasoning modes and a usable output budget", async (t) => {
	const tempDir = mkdtempSync(join(tmpdir(), "pi-nvidia-nim-minimax-m3-"));
	t.after(() => rmSync(tempDir, { recursive: true, force: true }));

	const modelRegistry = await createModelRegistry(join(tempDir, "auth.json"), tempDir);
	let providerConfig;
	extension({
		registerProvider(name, config) {
			providerConfig = config;
			modelRegistry.registerProvider(name, config);
		},
		on() {},
	});

	const model = modelRegistry.find("nvidia-nim", "minimaxai/minimax-m3");
	assert.ok(model, "expected minimaxai/minimax-m3 to be registered");
	assert.ok(providerConfig?.streamSimple, "extension should register a custom streamSimple");
	assert.deepEqual(
		{
			reasoning: model.reasoning,
			thinkingLevelMap: model.thinkingLevelMap,
			input: model.input,
			contextWindow: model.contextWindow,
			maxTokens: model.maxTokens,
			thinkingFormat: model.compat?.thinkingFormat,
			chatTemplateKwargs: model.compat?.chatTemplateKwargs,
		},
		{
			reasoning: true,
			thinkingLevelMap: {
				off: "disabled",
				minimal: "adaptive",
				low: "adaptive",
				medium: "adaptive",
				high: "enabled",
				xhigh: "enabled",
				max: "enabled",
			},
			input: ["text", "image"],
			contextWindow: 1_048_576,
			maxTokens: 16_384,
			thinkingFormat: "chat-template",
			chatTemplateKwargs: {
				thinking_mode: { $var: "thinking.effort" },
			},
		},
	);

	const originalFetch = globalThis.fetch;
	const requestBodies = [];
	globalThis.fetch = async (_url, init) => {
		requestBodies.push(JSON.parse(String(init?.body)));
		return new Response("unauthorized", {
			status: 401,
			headers: { "content-type": "text/plain" },
		});
	};

	t.after(() => {
		globalThis.fetch = originalFetch;
	});

	for (const reasoning of [undefined, "minimal", "low", "medium", "high", "xhigh", "max"]) {
		const stream = providerConfig.streamSimple(model, createUserContext(), {
			apiKey: "ABC123",
			reasoning,
		});
		for await (const _event of stream) {
			// Consume the stream so the mocked request completes.
		}
	}

	assert.equal(
		requestBodies.some((body) => Object.hasOwn(body, "reasoning_effort")),
		false,
		"MiniMax M3 reasoning mode should only be sent through chat_template_kwargs",
	);
	assert.deepEqual(
		requestBodies.map(({ max_tokens, chat_template_kwargs }) => ({
			max_tokens,
			chat_template_kwargs,
		})),
		[
			{ max_tokens: 16_384, chat_template_kwargs: { thinking_mode: "disabled" } },
			{ max_tokens: 16_384, chat_template_kwargs: { thinking_mode: "adaptive" } },
			{ max_tokens: 16_384, chat_template_kwargs: { thinking_mode: "adaptive" } },
			{ max_tokens: 16_384, chat_template_kwargs: { thinking_mode: "adaptive" } },
			{ max_tokens: 16_384, chat_template_kwargs: { thinking_mode: "enabled" } },
			{ max_tokens: 16_384, chat_template_kwargs: { thinking_mode: "enabled" } },
			{ max_tokens: 16_384, chat_template_kwargs: { thinking_mode: "enabled" } },
		],
	);
});

test("uses auth.json literal identifier-shaped NVIDIA NIM credentials for provider requests", async (t) => {
	const tempDir = mkdtempSync(join(tmpdir(), "pi-nvidia-nim-"));
	t.after(() => rmSync(tempDir, { recursive: true, force: true }));
	usePiAgentDir(t, tempDir);

	const authPath = join(tempDir, "auth.json");
	writeFileSync(
		authPath,
		JSON.stringify(
			{
				"nvidia-nim": { type: "api_key", key: "ABC123" },
			},
			null,
			2,
		),
	);

	const modelRegistry = await createModelRegistry(authPath, tempDir);

	let providerConfig;
	extension({
		registerProvider(name, config) {
			providerConfig = config;
			modelRegistry.registerProvider(name, config);
		},
		on() {},
	});

	assert.ok(providerConfig?.streamSimple, "extension should register a custom streamSimple");

	const model = modelRegistry.find("nvidia-nim", "deepseek-ai/deepseek-v3.2");
	assert.ok(model, "expected deepseek-ai/deepseek-v3.2 to be registered");

	const apiKey = await getModelRequestApiKey(modelRegistry, model);
	assert.equal(apiKey, "ABC123");

	const originalFetch = globalThis.fetch;
	const originalEnv = process.env.NVIDIA_NIM_API_KEY;
	delete process.env.NVIDIA_NIM_API_KEY;

	let requestUrl;
	let authorizationHeader;

	globalThis.fetch = async (url, init) => {
		requestUrl = String(url);
		authorizationHeader = new Headers(init?.headers).get("authorization");
		return new Response("unauthorized", {
			status: 401,
			headers: { "content-type": "text/plain" },
		});
	};

	t.after(() => {
		globalThis.fetch = originalFetch;
		if (originalEnv === undefined) {
			delete process.env.NVIDIA_NIM_API_KEY;
		} else {
			process.env.NVIDIA_NIM_API_KEY = originalEnv;
		}
	});

	const stream = providerConfig.streamSimple(model, createUserContext(), {
		apiKey,
		reasoning: "minimal",
	});

	let sawErrorEvent = false;
	for await (const event of stream) {
		if (event.type === "error") {
			sawErrorEvent = true;
		}
	}

	assert.equal(requestUrl, "https://integrate.api.nvidia.com/v1/chat/completions");
	assert.equal(authorizationHeader, "Bearer ABC123");
	assert.equal(sawErrorEvent, true);
});

test("uses auth.json env-derived identifier-shaped NVIDIA NIM credentials for provider requests", async (t) => {
	const tempDir = mkdtempSync(join(tmpdir(), "pi-nvidia-nim-"));
	t.after(() => rmSync(tempDir, { recursive: true, force: true }));
	usePiAgentDir(t, tempDir);

	const authPath = join(tempDir, "auth.json");
	writeFileSync(
		authPath,
		JSON.stringify(
			{
				"nvidia-nim": { type: "api_key", key: "MY_NIM_KEY" },
			},
			null,
			2,
		),
	);

	const originalCustomEnv = process.env.MY_NIM_KEY;
	process.env.MY_NIM_KEY = "ABC123";

	const modelRegistry = await createModelRegistry(authPath, tempDir);

	let providerConfig;
	extension({
		registerProvider(name, config) {
			providerConfig = config;
			modelRegistry.registerProvider(name, config);
		},
		on() {},
	});

	const model = modelRegistry.find("nvidia-nim", "deepseek-ai/deepseek-v3.2");
	assert.ok(model, "expected deepseek-ai/deepseek-v3.2 to be registered");

	const apiKey = await getModelRequestApiKey(modelRegistry, model);
	assert.equal(apiKey, "MY_NIM_KEY");

	const originalFetch = globalThis.fetch;
	let requestUrl;
	let authorizationHeader;

	globalThis.fetch = async (url, init) => {
		requestUrl = String(url);
		authorizationHeader = new Headers(init?.headers).get("authorization");
		return new Response("unauthorized", {
			status: 401,
			headers: { "content-type": "text/plain" },
		});
	};

	t.after(() => {
		globalThis.fetch = originalFetch;
		if (originalCustomEnv === undefined) {
			delete process.env.MY_NIM_KEY;
		} else {
			process.env.MY_NIM_KEY = originalCustomEnv;
		}
	});

	const stream = providerConfig.streamSimple(model, createUserContext(), {
		apiKey,
		reasoning: "minimal",
	});

	let sawErrorEvent = false;
	for await (const event of stream) {
		if (event.type === "error") {
			sawErrorEvent = true;
		}
	}

	assert.equal(requestUrl, "https://integrate.api.nvidia.com/v1/chat/completions");
	assert.equal(authorizationHeader, "Bearer ABC123");
	assert.equal(sawErrorEvent, true);
});

test("uses the active runtime authPath for identifier-shaped discovery and completion credentials", async (t) => {
	const tempDir = mkdtempSync(join(tmpdir(), "pi-nvidia-nim-runtime-auth-"));
	t.after(() => rmSync(tempDir, { recursive: true, force: true }));
	usePiAgentDir(t, join(tempDir, "default-agent-dir"));

	const authPath = join(tempDir, "runtime-auth.json");
	writeFileSync(
		authPath,
		JSON.stringify(
			{
				"nvidia-nim": { type: "api_key", key: "MY_NIM_KEY" },
			},
			null,
			2,
		),
	);

	const originalCustomEnv = process.env.MY_NIM_KEY;
	process.env.MY_NIM_KEY = "runtime-secret";

	const runtime = await createModelRuntime(authPath, tempDir);
	const modelRegistry = new ModelRegistry(runtime);
	let sessionStartHandler;

	extension({
		registerProvider(name, config) {
			modelRegistry.registerProvider(name, config);
		},
		on(eventName, handler) {
			if (eventName === "session_start") {
				sessionStartHandler = handler;
			}
		},
	});

	assert.ok(sessionStartHandler, "extension should register a session_start handler");

	const model = modelRegistry.find("nvidia-nim", "deepseek-ai/deepseek-v3.2");
	assert.ok(model, "expected deepseek-ai/deepseek-v3.2 to be registered");

	const originalFetch = globalThis.fetch;
	const requests = [];

	globalThis.fetch = async (url, init) => {
		const requestUrl = String(url);
		requests.push({
			url: requestUrl,
			authorization: new Headers(init?.headers).get("authorization"),
		});

		if (requestUrl.endsWith("/models")) {
			return Response.json({ data: [] });
		}

		return new Response("unauthorized", {
			status: 401,
			headers: { "content-type": "text/plain" },
		});
	};

	t.after(() => {
		globalThis.fetch = originalFetch;
		if (originalCustomEnv === undefined) {
			delete process.env.MY_NIM_KEY;
		} else {
			process.env.MY_NIM_KEY = originalCustomEnv;
		}
	});

	await sessionStartHandler(
		{ reason: "startup" },
		{
			ui: { notify() {} },
			modelRegistry,
		},
	);

	const stream = runtime.streamSimple(model, createUserContext(), { reasoning: "minimal" });
	for await (const _event of stream) {
		// Consume the request stream so the mocked completion request finishes.
	}

	assert.deepEqual(requests, [
		{
			url: "https://integrate.api.nvidia.com/v1/models",
			authorization: "Bearer runtime-secret",
		},
		{
			url: "https://integrate.api.nvidia.com/v1/chat/completions",
			authorization: "Bearer runtime-secret",
		},
	]);
});

test("uses NVIDIA_NIM_API_KEY env fallback when the resolved request key is still the env placeholder", async (t) => {
	const tempDir = mkdtempSync(join(tmpdir(), "pi-nvidia-nim-"));
	t.after(() => rmSync(tempDir, { recursive: true, force: true }));
	usePiAgentDir(t, tempDir);

	const modelRegistry = await createModelRegistry(join(tempDir, "auth.json"), tempDir);

	let providerConfig;
	extension({
		registerProvider(name, config) {
			providerConfig = config;
			modelRegistry.registerProvider(name, config);
		},
		on() {},
	});

	const model = modelRegistry.find("nvidia-nim", "deepseek-ai/deepseek-v3.2");
	assert.ok(model, "expected deepseek-ai/deepseek-v3.2 to be registered");

	const apiKey = EXPLICIT_NVIDIA_NIM_API_KEY_REF;

	const originalFetch = globalThis.fetch;
	const originalEnv = process.env.NVIDIA_NIM_API_KEY;
	process.env.NVIDIA_NIM_API_KEY = "nvapi-test-key";

	let requestUrl;
	let authorizationHeader;

	globalThis.fetch = async (url, init) => {
		requestUrl = String(url);
		authorizationHeader = new Headers(init?.headers).get("authorization");
		return new Response("unauthorized", {
			status: 401,
			headers: { "content-type": "text/plain" },
		});
	};

	t.after(() => {
		globalThis.fetch = originalFetch;
		if (originalEnv === undefined) {
			delete process.env.NVIDIA_NIM_API_KEY;
		} else {
			process.env.NVIDIA_NIM_API_KEY = originalEnv;
		}
	});

	const stream = providerConfig.streamSimple(model, createUserContext(), {
		apiKey,
		reasoning: "minimal",
	});

	let sawErrorEvent = false;
	for await (const event of stream) {
		if (event.type === "error") {
			sawErrorEvent = true;
		}
	}

	assert.equal(requestUrl, "https://integrate.api.nvidia.com/v1/chat/completions");
	assert.equal(authorizationHeader, "Bearer nvapi-test-key");
	assert.equal(sawErrorEvent, true);
});

test("rewrites stale Authorization headers after resolving env placeholder request keys", async (t) => {
	const tempDir = mkdtempSync(join(tmpdir(), "pi-nvidia-nim-"));
	t.after(() => rmSync(tempDir, { recursive: true, force: true }));
	usePiAgentDir(t, tempDir);

	const modelRegistry = await createModelRegistry(join(tempDir, "auth.json"), tempDir);

	let providerConfig;
	extension({
		registerProvider(name, config) {
			providerConfig = config;
			modelRegistry.registerProvider(name, config);
		},
		on() {},
	});

	const model = modelRegistry.find("nvidia-nim", "deepseek-ai/deepseek-v3.2");
	assert.ok(model, "expected deepseek-ai/deepseek-v3.2 to be registered");

	const originalFetch = globalThis.fetch;
	const originalEnv = process.env.NVIDIA_NIM_API_KEY;
	process.env.NVIDIA_NIM_API_KEY = "nvapi-test-key";

	let requestUrl;
	let authorizationHeader;
	let testHeader;
	let removedHeader;

	globalThis.fetch = async (url, init) => {
		requestUrl = String(url);
		const headers = new Headers(init?.headers);
		authorizationHeader = headers.get("authorization");
		testHeader = headers.get("x-test-header");
		removedHeader = headers.get("x-removed-header");
		return new Response("unauthorized", {
			status: 401,
			headers: { "content-type": "text/plain" },
		});
	};

	t.after(() => {
		globalThis.fetch = originalFetch;
		if (originalEnv === undefined) {
			delete process.env.NVIDIA_NIM_API_KEY;
		} else {
			process.env.NVIDIA_NIM_API_KEY = originalEnv;
		}
	});

	const stream = providerConfig.streamSimple(model, createUserContext(), {
		apiKey: EXPLICIT_NVIDIA_NIM_API_KEY_REF,
		headers: {
			Authorization: `Bearer ${EXPLICIT_NVIDIA_NIM_API_KEY_REF}`,
			authorization: "Bearer stale-lowercase-key",
			"x-test-header": "kept",
			"x-removed-header": null,
		},
		reasoning: "minimal",
	});

	let sawErrorEvent = false;
	for await (const event of stream) {
		if (event.type === "error") {
			sawErrorEvent = true;
		}
	}

	assert.equal(requestUrl, "https://integrate.api.nvidia.com/v1/chat/completions");
	assert.equal(authorizationHeader, "Bearer nvapi-test-key");
	assert.equal(testHeader, "kept");
	assert.equal(removedHeader, null);
	assert.equal(sawErrorEvent, true);
});


test("fails locally when no configured key is available and the resolved request key is still the env placeholder", async (t) => {
	const tempDir = mkdtempSync(join(tmpdir(), "pi-nvidia-nim-"));
	t.after(() => rmSync(tempDir, { recursive: true, force: true }));
	usePiAgentDir(t, tempDir);

	const modelRegistry = await createModelRegistry(join(tempDir, "auth.json"), tempDir);

	let providerConfig;
	extension({
		registerProvider(name, config) {
			providerConfig = config;
			modelRegistry.registerProvider(name, config);
		},
		on() {},
	});

	const model = modelRegistry.find("nvidia-nim", "deepseek-ai/deepseek-v3.2");
	assert.ok(model, "expected deepseek-ai/deepseek-v3.2 to be registered");

	const apiKey = EXPLICIT_NVIDIA_NIM_API_KEY_REF;

	const originalFetch = globalThis.fetch;
	const originalEnv = process.env.NVIDIA_NIM_API_KEY;
	delete process.env.NVIDIA_NIM_API_KEY;

	let fetchCalled = false;
	globalThis.fetch = async () => {
		fetchCalled = true;
		throw new Error("fetch should not be called when auth is unresolved");
	};

	t.after(() => {
		globalThis.fetch = originalFetch;
		if (originalEnv === undefined) {
			delete process.env.NVIDIA_NIM_API_KEY;
		} else {
			process.env.NVIDIA_NIM_API_KEY = originalEnv;
		}
	});

	assert.throws(
		() => providerConfig.streamSimple(model, createUserContext(), { apiKey }),
		/no API key configured/,
	);
	assert.equal(fetchCalled, false);
});


test("fails locally when an auth.json shell-command key resolves to an empty value for provider requests", async (t) => {
	const tempDir = mkdtempSync(join(tmpdir(), "pi-nvidia-nim-"));
	t.after(() => rmSync(tempDir, { recursive: true, force: true }));
	usePiAgentDir(t, tempDir);

	const authPath = join(tempDir, "auth.json");
	writeFileSync(
		authPath,
		JSON.stringify(
			{
				"nvidia-nim": { type: "api_key", key: "!printf ''" },
			},
			null,
			2,
		),
	);

	const runtime = await createModelRuntime(authPath, tempDir);
	const modelRegistry = new ModelRegistry(runtime);

	extension({
		registerProvider(name, config) {
			modelRegistry.registerProvider(name, config);
		},
		on() {},
	});

	const model = modelRegistry.find("nvidia-nim", "deepseek-ai/deepseek-v3.2");
	assert.ok(model, "expected deepseek-ai/deepseek-v3.2 to be registered");

	const originalFetch = globalThis.fetch;
	const originalEnv = process.env.NVIDIA_NIM_API_KEY;
	process.env.NVIDIA_NIM_API_KEY = "nvapi-test-key";

	let fetchCalled = false;
	globalThis.fetch = async () => {
		fetchCalled = true;
		throw new Error("fetch should not be called when auth resolution is empty");
	};

	t.after(() => {
		globalThis.fetch = originalFetch;
		if (originalEnv === undefined) {
			delete process.env.NVIDIA_NIM_API_KEY;
		} else {
			process.env.NVIDIA_NIM_API_KEY = originalEnv;
		}
	});

	const stream = runtime.streamSimple(model, createUserContext(), { reasoning: "minimal" });
	let sawErrorEvent = false;
	for await (const event of stream) {
		if (event.type === "error") {
			sawErrorEvent = true;
		}
	}

	assert.equal(sawErrorEvent, true);
	assert.equal(fetchCalled, false);
});


test("uses identifier-shaped shell-command output credentials for provider requests", async (t) => {
	const tempDir = mkdtempSync(join(tmpdir(), "pi-nvidia-nim-"));
	t.after(() => rmSync(tempDir, { recursive: true, force: true }));
	usePiAgentDir(t, tempDir);

	const authPath = join(tempDir, "auth.json");
	writeFileSync(
		authPath,
		JSON.stringify(
			{
				"nvidia-nim": { type: "api_key", key: "!printf ABC123" },
			},
			null,
			2,
		),
	);

	const modelRegistry = await createModelRegistry(authPath, tempDir);

	let providerConfig;
	extension({
		registerProvider(name, config) {
			providerConfig = config;
			modelRegistry.registerProvider(name, config);
		},
		on() {},
	});

	const model = modelRegistry.find("nvidia-nim", "deepseek-ai/deepseek-v3.2");
	assert.ok(model, "expected deepseek-ai/deepseek-v3.2 to be registered");

	const apiKey = await getModelRequestApiKey(modelRegistry, model);
	assert.equal(apiKey, "ABC123");

	const originalFetch = globalThis.fetch;
	const originalEnv = process.env.NVIDIA_NIM_API_KEY;
	delete process.env.NVIDIA_NIM_API_KEY;

	let requestUrl;
	let authorizationHeader;

	globalThis.fetch = async (url, init) => {
		requestUrl = String(url);
		authorizationHeader = new Headers(init?.headers).get("authorization");
		return new Response("unauthorized", {
			status: 401,
			headers: { "content-type": "text/plain" },
		});
	};

	t.after(() => {
		globalThis.fetch = originalFetch;
		if (originalEnv === undefined) {
			delete process.env.NVIDIA_NIM_API_KEY;
		} else {
			process.env.NVIDIA_NIM_API_KEY = originalEnv;
		}
	});

	const stream = providerConfig.streamSimple(model, createUserContext(), {
		apiKey,
		reasoning: "minimal",
	});

	let sawErrorEvent = false;
	for await (const event of stream) {
		if (event.type === "error") {
			sawErrorEvent = true;
		}
	}

	assert.equal(requestUrl, "https://integrate.api.nvidia.com/v1/chat/completions");
	assert.equal(authorizationHeader, "Bearer ABC123");
	assert.equal(sawErrorEvent, true);
});

test("discovers additional models with auth.json literal identifier-shaped provider credentials", async (t) => {
	const tempDir = mkdtempSync(join(tmpdir(), "pi-nvidia-nim-"));
	t.after(() => rmSync(tempDir, { recursive: true, force: true }));
	usePiAgentDir(t, tempDir);

	const authPath = join(tempDir, "auth.json");
	writeFileSync(
		authPath,
		JSON.stringify(
			{
				"nvidia-nim": { type: "api_key", key: "ABC123" },
			},
			null,
			2,
		),
	);

	const modelRegistry = await createModelRegistry(authPath, tempDir);

	let sessionStartHandler;
	extension({
		registerProvider(name, config) {
			modelRegistry.registerProvider(name, config);
		},
		on(eventName, handler) {
			if (eventName === "session_start") {
				sessionStartHandler = handler;
			}
		},
	});

	assert.ok(sessionStartHandler, "extension should register a session_start handler");

	const apiKey = await modelRegistry.getApiKeyForProvider("nvidia-nim");
	assert.equal(apiKey, "ABC123");

	const originalFetch = globalThis.fetch;
	const originalEnv = process.env.NVIDIA_NIM_API_KEY;
	delete process.env.NVIDIA_NIM_API_KEY;

	let requestUrl;
	let authorizationHeader;

	globalThis.fetch = async (url, init) => {
		requestUrl = String(url);
		authorizationHeader = new Headers(init?.headers).get("authorization");
		return new Response(
			JSON.stringify({
				data: [{ id: "acme/literal-chat-model", object: "model", owned_by: "acme" }],
			}),
			{
				status: 200,
				headers: { "content-type": "application/json" },
			},
		);
	};

	t.after(() => {
		globalThis.fetch = originalFetch;
		if (originalEnv === undefined) {
			delete process.env.NVIDIA_NIM_API_KEY;
		} else {
			process.env.NVIDIA_NIM_API_KEY = originalEnv;
		}
	});

	await sessionStartHandler(
		{ reason: "startup" },
		{
			modelRegistry: {
				getApiKeyForProvider: async (provider) => {
					assert.equal(provider, "nvidia-nim");
					return modelRegistry.getApiKeyForProvider(provider);
				},
			},
		},
	);
	const registeredProviderConfig = modelRegistry.getRegisteredProviderConfig("nvidia-nim");

	assert.equal(requestUrl, "https://integrate.api.nvidia.com/v1/models");
	assert.equal(authorizationHeader, "Bearer ABC123");
	assert.ok(
		registeredProviderConfig?.models?.some((model) => model.id === "acme/literal-chat-model"),
		"expected discovered models to be re-registered",
	);
});

test("discovers additional models with NVIDIA_NIM_API_KEY env fallback", async (t) => {
	const tempDir = mkdtempSync(join(tmpdir(), "pi-nvidia-nim-"));
	t.after(() => rmSync(tempDir, { recursive: true, force: true }));
	usePiAgentDir(t, tempDir);

	const modelRegistry = await createModelRegistry(join(tempDir, "auth.json"), tempDir);

	let sessionStartHandler;
	extension({
		registerProvider(name, config) {
			modelRegistry.registerProvider(name, config);
		},
		on(eventName, handler) {
			if (eventName === "session_start") {
				sessionStartHandler = handler;
			}
		},
	});

	assert.ok(sessionStartHandler, "extension should register a session_start handler");

	const unresolvedApiKey = await modelRegistry.getApiKeyForProvider("nvidia-nim");
	assert.equal(unresolvedApiKey, undefined);

	const originalFetch = globalThis.fetch;
	const originalEnv = process.env.NVIDIA_NIM_API_KEY;
	process.env.NVIDIA_NIM_API_KEY = "nvapi-test-key";

	let requestUrl;
	let authorizationHeader;

	globalThis.fetch = async (url, init) => {
		requestUrl = String(url);
		authorizationHeader = new Headers(init?.headers).get("authorization");
		return new Response(
			JSON.stringify({
				data: [{ id: "acme/env-chat-model", object: "model", owned_by: "acme" }],
			}),
			{
				status: 200,
				headers: { "content-type": "application/json" },
			},
		);
	};

	t.after(() => {
		globalThis.fetch = originalFetch;
		if (originalEnv === undefined) {
			delete process.env.NVIDIA_NIM_API_KEY;
		} else {
			process.env.NVIDIA_NIM_API_KEY = originalEnv;
		}
	});

	await sessionStartHandler(
		{ reason: "startup" },
		{
			modelRegistry: {
				getApiKeyForProvider: async (provider) => {
					assert.equal(provider, "nvidia-nim");
					return modelRegistry.getApiKeyForProvider(provider);
				},
			},
		},
	);
	const registeredProviderConfig = modelRegistry.getRegisteredProviderConfig("nvidia-nim");

	assert.equal(requestUrl, "https://integrate.api.nvidia.com/v1/models");
	assert.equal(authorizationHeader, "Bearer nvapi-test-key");
	assert.equal(registeredProviderConfig?.apiKey, EXPLICIT_NVIDIA_NIM_API_KEY_REF);
	assert.ok(
		registeredProviderConfig?.models?.some((model) => model.id === "acme/env-chat-model"),
		"expected discovered models to be re-registered after env fallback",
	);
});

test("discovers additional models when auth.json env references resolve to identifier-shaped credentials", async (t) => {
	const tempDir = mkdtempSync(join(tmpdir(), "pi-nvidia-nim-"));
	t.after(() => rmSync(tempDir, { recursive: true, force: true }));
	usePiAgentDir(t, tempDir);

	const authPath = join(tempDir, "auth.json");
	writeFileSync(
		authPath,
		JSON.stringify(
			{
				"nvidia-nim": { type: "api_key", key: "MY_NIM_KEY" },
			},
			null,
			2,
		),
	);

	const originalCustomEnv = process.env.MY_NIM_KEY;
	process.env.MY_NIM_KEY = "ABC123";

	const modelRegistry = await createModelRegistry(authPath, tempDir);

	let sessionStartHandler;
	extension({
		registerProvider(name, config) {
			modelRegistry.registerProvider(name, config);
		},
		on(eventName, handler) {
			if (eventName === "session_start") {
				sessionStartHandler = handler;
			}
		},
	});

	assert.ok(sessionStartHandler, "extension should register a session_start handler");

	const apiKey = await modelRegistry.getApiKeyForProvider("nvidia-nim");
	assert.equal(apiKey, "MY_NIM_KEY");

	const originalFetch = globalThis.fetch;
	const originalWarn = console.warn;
	let requestUrl;
	let authorizationHeader;
	const notifications = [];
	const warnings = [];

	globalThis.fetch = async (url, init) => {
		requestUrl = String(url);
		authorizationHeader = new Headers(init?.headers).get("authorization");
		return new Response(
			JSON.stringify({
				data: [{ id: "acme/custom-env-chat-model", object: "model", owned_by: "acme" }],
			}),
			{
				status: 200,
				headers: { "content-type": "application/json" },
			},
		);
	};
	console.warn = (...args) => {
		warnings.push(args.map((arg) => String(arg)).join(" "));
	};

	t.after(() => {
		globalThis.fetch = originalFetch;
		console.warn = originalWarn;
		if (originalCustomEnv === undefined) {
			delete process.env.MY_NIM_KEY;
		} else {
			process.env.MY_NIM_KEY = originalCustomEnv;
		}
	});

	await sessionStartHandler(
		{ reason: "startup" },
		{
			hasUI: true,
			ui: {
				notify(message, level) {
					notifications.push({ message, level });
				},
			},
			modelRegistry: {
				getApiKeyForProvider: async (provider) => {
					assert.equal(provider, "nvidia-nim");
					return modelRegistry.getApiKeyForProvider(provider);
				},
			},
		},
	);
	const registeredProviderConfig = modelRegistry.getRegisteredProviderConfig("nvidia-nim");

	assert.equal(requestUrl, "https://integrate.api.nvidia.com/v1/models");
	assert.equal(authorizationHeader, "Bearer ABC123");
	assert.ok(
		registeredProviderConfig?.models?.some((model) => model.id === "acme/custom-env-chat-model"),
		"expected discovered models to be re-registered",
	);
	assert.deepEqual(notifications, []);
	assert.deepEqual(warnings, []);
});


test("skips model discovery quietly when no NVIDIA NIM credentials are configured", async (t) => {
	const tempDir = mkdtempSync(join(tmpdir(), "pi-nvidia-nim-"));
	t.after(() => rmSync(tempDir, { recursive: true, force: true }));
	usePiAgentDir(t, tempDir);

	const modelRegistry = await createModelRegistry(join(tempDir, "auth.json"), tempDir);

	let sessionStartHandler;
	extension({
		registerProvider(name, config) {
			modelRegistry.registerProvider(name, config);
		},
		on(eventName, handler) {
			if (eventName === "session_start") {
				sessionStartHandler = handler;
			}
		},
	});

	assert.ok(sessionStartHandler, "extension should register a session_start handler");

	const unresolvedApiKey = await modelRegistry.getApiKeyForProvider("nvidia-nim");
	assert.equal(unresolvedApiKey, undefined);

	const originalFetch = globalThis.fetch;
	const originalEnv = process.env.NVIDIA_NIM_API_KEY;
	const originalWarn = console.warn;
	delete process.env.NVIDIA_NIM_API_KEY;

	let fetchCalled = false;
	const notifications = [];
	const warnings = [];

	globalThis.fetch = async () => {
		fetchCalled = true;
		throw new Error("fetch should not be called when auth is unresolved");
	};
	console.warn = (...args) => {
		warnings.push(args.map((arg) => String(arg)).join(" "));
	};

	t.after(() => {
		globalThis.fetch = originalFetch;
		console.warn = originalWarn;
		if (originalEnv === undefined) {
			delete process.env.NVIDIA_NIM_API_KEY;
		} else {
			process.env.NVIDIA_NIM_API_KEY = originalEnv;
		}
	});

	await sessionStartHandler(
		{ reason: "startup" },
		{
			hasUI: true,
			ui: {
				notify(message, level) {
					notifications.push({ message, level });
				},
			},
			modelRegistry: {
				getApiKeyForProvider: async (provider) => {
					assert.equal(provider, "nvidia-nim");
					return modelRegistry.getApiKeyForProvider(provider);
				},
				getProviderAuthStatus(provider) {
					assert.equal(provider, "nvidia-nim");
					return modelRegistry.getProviderAuthStatus(provider);
				},
			},
		},
	);

	assert.equal(fetchCalled, false);
	assert.deepEqual(notifications, []);
	assert.deepEqual(warnings, []);
});

test("skips model discovery quietly when provider auth is absent and getApiKeyForProvider returns undefined", async (t) => {
	const tempDir = mkdtempSync(join(tmpdir(), "pi-nvidia-nim-"));
	t.after(() => rmSync(tempDir, { recursive: true, force: true }));
	usePiAgentDir(t, tempDir);

	let sessionStartHandler;

	extension({
		registerProvider() {},
		on(eventName, handler) {
			if (eventName === "session_start") {
				sessionStartHandler = handler;
			}
		},
	});

	assert.ok(sessionStartHandler, "extension should register a session_start handler");

	const originalFetch = globalThis.fetch;
	const originalWarn = console.warn;

	let fetchCalled = false;
	const notifications = [];
	const warnings = [];

	globalThis.fetch = async () => {
		fetchCalled = true;
		throw new Error("fetch should not be called when auth is absent");
	};
	console.warn = (...args) => {
		warnings.push(args.map((arg) => String(arg)).join(" "));
	};

	t.after(() => {
		globalThis.fetch = originalFetch;
		console.warn = originalWarn;
	});

	await sessionStartHandler(
		{ reason: "startup" },
		{
			hasUI: true,
			ui: {
				notify(message, level) {
					notifications.push({ message, level });
				},
			},
			modelRegistry: {
				getApiKeyForProvider: async (provider) => {
					assert.equal(provider, "nvidia-nim");
					return undefined;
				},
				getProviderAuthStatus(provider) {
					assert.equal(provider, "nvidia-nim");
					return { configured: false };
				},
			},
		},
	);

	assert.equal(fetchCalled, false);
	assert.deepEqual(notifications, []);
	assert.deepEqual(warnings, []);
});


test("skips model discovery when an auth.json shell-command key resolves to an empty value", async (t) => {
	const tempDir = mkdtempSync(join(tmpdir(), "pi-nvidia-nim-"));
	t.after(() => rmSync(tempDir, { recursive: true, force: true }));
	usePiAgentDir(t, tempDir);

	const authPath = join(tempDir, "auth.json");
	writeFileSync(
		authPath,
		JSON.stringify(
			{
				"nvidia-nim": { type: "api_key", key: "!printf '' # discovery-empty-key" },
			},
			null,
			2,
		),
	);

	const modelRegistry = await createModelRegistry(authPath, tempDir);

	let sessionStartHandler;
	extension({
		registerProvider(name, config) {
			modelRegistry.registerProvider(name, config);
		},
		on(eventName, handler) {
			if (eventName === "session_start") {
				sessionStartHandler = handler;
			}
		},
	});

	assert.ok(sessionStartHandler, "extension should register a session_start handler");

	const apiKey = await modelRegistry.getApiKeyForProvider("nvidia-nim");
	assert.equal(apiKey, undefined);

	const originalFetch = globalThis.fetch;
	const originalEnv = process.env.NVIDIA_NIM_API_KEY;
	const originalWarn = console.warn;
	process.env.NVIDIA_NIM_API_KEY = "nvapi-test-key";

	let fetchCalled = false;
	const notifications = [];
	const warnings = [];

	globalThis.fetch = async () => {
		fetchCalled = true;
		throw new Error("fetch should not be called when auth resolution is empty");
	};
	console.warn = (...args) => {
		warnings.push(args.map((arg) => String(arg)).join(" "));
	};

	t.after(() => {
		globalThis.fetch = originalFetch;
		console.warn = originalWarn;
		if (originalEnv === undefined) {
			delete process.env.NVIDIA_NIM_API_KEY;
		} else {
			process.env.NVIDIA_NIM_API_KEY = originalEnv;
		}
	});

	await sessionStartHandler(
		{ reason: "startup" },
		{
			hasUI: true,
			ui: {
				notify(message, level) {
					notifications.push({ message, level });
				},
			},
			modelRegistry: {
				getApiKeyForProvider: async (provider) => {
					assert.equal(provider, "nvidia-nim");
					return modelRegistry.getApiKeyForProvider(provider);
				},
				getProviderAuthStatus(provider) {
					assert.equal(provider, "nvidia-nim");
					return modelRegistry.getProviderAuthStatus(provider);
				},
			},
		},
	);

	assert.equal(fetchCalled, false);
	assert.deepEqual(notifications, [
		{
			message: "NVIDIA NIM model discovery skipped: check your nvidia-nim credentials.",
			level: "warning",
		},
	]);
	assert.equal(warnings.length, 1);
	assert.equal(warnings.some((message) => message.includes("nvapi-test-key")), false);
});


test("skips model discovery and shows a sanitized warning for auth failures", async (t) => {
	let sessionStartHandler;

	extension({
		registerProvider() {},
		on(eventName, handler) {
			if (eventName === "session_start") {
				sessionStartHandler = handler;
			}
		},
	});

	assert.ok(sessionStartHandler, "extension should register a session_start handler");

	const originalFetch = globalThis.fetch;
	const originalEnv = process.env.NVIDIA_NIM_API_KEY;
	delete process.env.NVIDIA_NIM_API_KEY;

	let requestUrl;
	let authorizationHeader;
	let registerCalled = false;
	const notifications = [];

	globalThis.fetch = async (url, init) => {
		requestUrl = String(url);
		authorizationHeader = new Headers(init?.headers).get("authorization");
		return new Response("invalid token details leaked", {
			status: 401,
			headers: { "content-type": "text/plain" },
		});
	};

	t.after(() => {
		globalThis.fetch = originalFetch;
		if (originalEnv === undefined) {
			delete process.env.NVIDIA_NIM_API_KEY;
		} else {
			process.env.NVIDIA_NIM_API_KEY = originalEnv;
		}
	});

	await sessionStartHandler(
		{ reason: "startup" },
		{
			hasUI: true,
			ui: {
				notify(message, level) {
					notifications.push({ message, level });
				},
			},
			modelRegistry: {
				getApiKeyForProvider: async (provider) => {
					assert.equal(provider, "nvidia-nim");
					return "nvapi-real-key";
				},
				registerProvider() {
					registerCalled = true;
				},
			},
		},
	);

	assert.equal(requestUrl, "https://integrate.api.nvidia.com/v1/models");
	assert.equal(authorizationHeader, "Bearer nvapi-real-key");
	assert.equal(registerCalled, false);
	assert.deepEqual(notifications, [
		{
			message: "NVIDIA NIM model discovery skipped: check your nvidia-nim credentials.",
			level: "warning",
		},
	]);
	assert.equal(notifications[0].message.includes("invalid token details leaked"), false);
});


test("skips model discovery without surfacing transient rate limit errors", async (t) => {
	let sessionStartHandler;

	extension({
		registerProvider() {},
		on(eventName, handler) {
			if (eventName === "session_start") {
				sessionStartHandler = handler;
			}
		},
	});

	assert.ok(sessionStartHandler, "extension should register a session_start handler");

	const originalFetch = globalThis.fetch;
	const originalEnv = process.env.NVIDIA_NIM_API_KEY;
	delete process.env.NVIDIA_NIM_API_KEY;

	let registerCalled = false;
	const notifications = [];

	globalThis.fetch = async () =>
		new Response("too many requests", {
			status: 429,
			headers: { "content-type": "text/plain" },
		});

	t.after(() => {
		globalThis.fetch = originalFetch;
		if (originalEnv === undefined) {
			delete process.env.NVIDIA_NIM_API_KEY;
		} else {
			process.env.NVIDIA_NIM_API_KEY = originalEnv;
		}
	});

	await sessionStartHandler(
		{ reason: "startup" },
		{
			hasUI: true,
			ui: {
				notify(message, level) {
					notifications.push({ message, level });
				},
			},
			modelRegistry: {
				getApiKeyForProvider: async () => "nvapi-real-key",
				registerProvider() {
					registerCalled = true;
				},
			},
		},
	);

	assert.equal(registerCalled, false);
	assert.deepEqual(notifications, []);
});


test("skips model discovery when the API returns an invalid payload", async (t) => {
	let sessionStartHandler;

	extension({
		registerProvider() {},
		on(eventName, handler) {
			if (eventName === "session_start") {
				sessionStartHandler = handler;
			}
		},
	});

	assert.ok(sessionStartHandler, "extension should register a session_start handler");

	const originalFetch = globalThis.fetch;
	const originalEnv = process.env.NVIDIA_NIM_API_KEY;
	delete process.env.NVIDIA_NIM_API_KEY;

	let registerCalled = false;
	const notifications = [];

	globalThis.fetch = async () =>
		new Response(JSON.stringify({ models: [] }), {
			status: 200,
			headers: { "content-type": "application/json" },
		});

	t.after(() => {
		globalThis.fetch = originalFetch;
		if (originalEnv === undefined) {
			delete process.env.NVIDIA_NIM_API_KEY;
		} else {
			process.env.NVIDIA_NIM_API_KEY = originalEnv;
		}
	});

	await sessionStartHandler(
		{ reason: "startup" },
		{
			hasUI: true,
			ui: {
				notify(message, level) {
					notifications.push({ message, level });
				},
			},
			modelRegistry: {
				getApiKeyForProvider: async () => "nvapi-real-key",
				registerProvider() {
					registerCalled = true;
				},
			},
		},
	);

	assert.equal(registerCalled, false);
	assert.deepEqual(notifications, []);
});
