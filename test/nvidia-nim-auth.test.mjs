import assert from "node:assert/strict";
import test from "node:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { AuthStorage, ModelRegistry } from "@mariozechner/pi-coding-agent";
import extension from "../index.ts";

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

function createAuthStorage(authPath) {
	return AuthStorage.create(authPath);
}

function createModelRegistry(authStorage, tempDir) {
	return ModelRegistry.create(authStorage, join(tempDir, "models.json"));
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

	const authStorage = createAuthStorage(authPath);
	const modelRegistry = createModelRegistry(authStorage, tempDir);

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

	const authStorage = createAuthStorage(authPath);
	const modelRegistry = createModelRegistry(authStorage, tempDir);

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


test("uses NVIDIA_NIM_API_KEY env fallback when the resolved request key is still the env placeholder", async (t) => {
	const tempDir = mkdtempSync(join(tmpdir(), "pi-nvidia-nim-"));
	t.after(() => rmSync(tempDir, { recursive: true, force: true }));
	usePiAgentDir(t, tempDir);

	const authStorage = createAuthStorage(join(tempDir, "auth.json"));
	const modelRegistry = createModelRegistry(authStorage, tempDir);

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
	assert.equal(apiKey, "NVIDIA_NIM_API_KEY");

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


test("fails locally when no configured key is available and the resolved request key is still the env placeholder", async (t) => {
	const tempDir = mkdtempSync(join(tmpdir(), "pi-nvidia-nim-"));
	t.after(() => rmSync(tempDir, { recursive: true, force: true }));
	usePiAgentDir(t, tempDir);

	const authStorage = createAuthStorage(join(tempDir, "auth.json"));
	const modelRegistry = createModelRegistry(authStorage, tempDir);

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
	assert.equal(apiKey, "NVIDIA_NIM_API_KEY");

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

	const authStorage = createAuthStorage(authPath);
	const modelRegistry = createModelRegistry(authStorage, tempDir);

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
	assert.equal(apiKey, "NVIDIA_NIM_API_KEY");

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

	assert.throws(
		() => providerConfig.streamSimple(model, createUserContext(), { apiKey }),
		/resolved to an empty value/,
	);
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

	const authStorage = createAuthStorage(authPath);
	const modelRegistry = createModelRegistry(authStorage, tempDir);

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

	const authStorage = createAuthStorage(authPath);
	const modelRegistry = createModelRegistry(authStorage, tempDir);

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
	let registeredProviderName;
	let registeredProviderConfig;

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
				authStorage: modelRegistry.authStorage,
				getApiKeyForProvider: async (provider) => {
					assert.equal(provider, "nvidia-nim");
					return modelRegistry.getApiKeyForProvider(provider);
				},
				registerProvider: (name, config) => {
					registeredProviderName = name;
					registeredProviderConfig = config;
				},
			},
		},
	);

	assert.equal(requestUrl, "https://integrate.api.nvidia.com/v1/models");
	assert.equal(authorizationHeader, "Bearer ABC123");
	assert.equal(registeredProviderName, "nvidia-nim");
	assert.ok(
		registeredProviderConfig?.models?.some((model) => model.id === "acme/literal-chat-model"),
		"expected discovered models to be re-registered",
	);
});

test("discovers additional models with NVIDIA_NIM_API_KEY env fallback", async (t) => {
	const tempDir = mkdtempSync(join(tmpdir(), "pi-nvidia-nim-"));
	t.after(() => rmSync(tempDir, { recursive: true, force: true }));
	usePiAgentDir(t, tempDir);

	const authStorage = createAuthStorage(join(tempDir, "auth.json"));
	const modelRegistry = createModelRegistry(authStorage, tempDir);

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
	assert.equal(unresolvedApiKey, "NVIDIA_NIM_API_KEY");

	const originalFetch = globalThis.fetch;
	const originalEnv = process.env.NVIDIA_NIM_API_KEY;
	process.env.NVIDIA_NIM_API_KEY = "nvapi-test-key";

	let requestUrl;
	let authorizationHeader;
	let registeredProviderName;
	let registeredProviderConfig;

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
				registerProvider: (name, config) => {
					registeredProviderName = name;
					registeredProviderConfig = config;
				},
			},
		},
	);

	assert.equal(requestUrl, "https://integrate.api.nvidia.com/v1/models");
	assert.equal(authorizationHeader, "Bearer nvapi-test-key");
	assert.equal(registeredProviderName, "nvidia-nim");
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

	const authStorage = createAuthStorage(authPath);
	const modelRegistry = createModelRegistry(authStorage, tempDir);

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
	const originalWarn = console.warn;
	let requestUrl;
	let authorizationHeader;
	let registeredProviderName;
	let registeredProviderConfig;
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
				authStorage: modelRegistry.authStorage,
				getApiKeyForProvider: async (provider) => {
					assert.equal(provider, "nvidia-nim");
					return modelRegistry.getApiKeyForProvider(provider);
				},
				registerProvider: (name, config) => {
					registeredProviderName = name;
					registeredProviderConfig = config;
				},
			},
		},
	);

	assert.equal(requestUrl, "https://integrate.api.nvidia.com/v1/models");
	assert.equal(authorizationHeader, "Bearer ABC123");
	assert.equal(registeredProviderName, "nvidia-nim");
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

	const authStorage = createAuthStorage(join(tempDir, "auth.json"));
	const modelRegistry = createModelRegistry(authStorage, tempDir);

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
	assert.equal(unresolvedApiKey, "NVIDIA_NIM_API_KEY");

	const originalFetch = globalThis.fetch;
	const originalEnv = process.env.NVIDIA_NIM_API_KEY;
	const originalWarn = console.warn;
	delete process.env.NVIDIA_NIM_API_KEY;

	let fetchCalled = false;
	let registerCalled = false;
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
				authStorage: modelRegistry.authStorage,
				getApiKeyForProvider: async (provider) => {
					assert.equal(provider, "nvidia-nim");
					return modelRegistry.getApiKeyForProvider(provider);
				},
				registerProvider() {
					registerCalled = true;
				},
			},
		},
	);

	assert.equal(fetchCalled, false);
	assert.equal(registerCalled, false);
	assert.deepEqual(notifications, []);
	assert.deepEqual(warnings, []);
});

test("skips model discovery quietly when provider auth is absent and getApiKeyForProvider returns undefined", async (t) => {
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
	let registerCalled = false;
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
				authStorage: {
					has(provider) {
						assert.equal(provider, "nvidia-nim");
						return false;
					},
				},
				getApiKeyForProvider: async (provider) => {
					assert.equal(provider, "nvidia-nim");
					return undefined;
				},
				registerProvider() {
					registerCalled = true;
				},
			},
		},
	);

	assert.equal(fetchCalled, false);
	assert.equal(registerCalled, false);
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

	const authStorage = createAuthStorage(authPath);
	const modelRegistry = createModelRegistry(authStorage, tempDir);

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
	assert.equal(apiKey, "NVIDIA_NIM_API_KEY");

	const originalFetch = globalThis.fetch;
	const originalEnv = process.env.NVIDIA_NIM_API_KEY;
	const originalWarn = console.warn;
	process.env.NVIDIA_NIM_API_KEY = "nvapi-test-key";

	let fetchCalled = false;
	let registerCalled = false;
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
				authStorage: modelRegistry.authStorage,
				getApiKeyForProvider: async (provider) => {
					assert.equal(provider, "nvidia-nim");
					return modelRegistry.getApiKeyForProvider(provider);
				},
				registerProvider() {
					registerCalled = true;
				},
			},
		},
	);

	assert.equal(fetchCalled, false);
	assert.equal(registerCalled, false);
	assert.deepEqual(notifications, [
		{
			message: "NVIDIA NIM model discovery skipped: check your nvidia-nim credentials.",
			level: "warning",
		},
	]);
	assert.ok(
		warnings.some((message) => message.includes("resolved to an empty value")),
	);
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
