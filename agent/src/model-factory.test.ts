import { execFileSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { generateObject, generateText, stepCountIs, tool } from "ai";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { resolveModel } from "./model-factory.js";
import { loadModelCatalog } from "./output-limits.js";

describe("native model resolution", () => {
  it("imports for Bedrock-only runs without reading Vertex credentials", () => {
    const factoryUrl = new URL("./model-factory.ts", import.meta.url).href;
    const script = `const env = process.env; process.env = new Proxy(env, { get(target, key) { if (["GOOGLE_VERTEX_API_KEY", "XAI_API_KEY", "XAI_BASE_URL"].includes(key)) throw new Error("external provider configuration read at import"); return Reflect.get(target, key); } }); await import(${JSON.stringify(factoryUrl)});`;
    expect(() =>
      execFileSync(process.execPath, ["--import", "tsx", "--input-type=module", "-e", script], {
        cwd: resolve(dirname(fileURLToPath(import.meta.url)), ".."),
        stdio: "pipe",
      }),
    ).not.toThrow();
  });
  it("routes the selected Bedrock families to their configured source regions and APIs", async () => {
    const catalog = loadModelCatalog();
    const selected = catalog.slice(0, 16);
    const requests: Array<{ url: string; body: Record<string, unknown> }> = [];
    const fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const body = JSON.parse(String(init?.body));
      requests.push({ url, body });
      const json = '{"selected_skill_ids":["skill_1"]}';
      if (url.endsWith("/responses"))
        return Response.json({
          id: "resp_1",
          object: "response",
          created_at: 1,
          model: body.model,
          status: "completed",
          output: [
            {
              id: "msg_1",
              type: "message",
              status: "completed",
              role: "assistant",
              content: [{ type: "output_text", text: json, annotations: [] }],
            },
          ],
          usage: { input_tokens: 4, output_tokens: 3, total_tokens: 7 },
        });
      if (url.endsWith("/chat/completions"))
        return Response.json({
          id: "chatcmpl_1",
          object: "chat.completion",
          created: 1,
          model: body.model,
          choices: [
            { index: 0, finish_reason: "stop", message: { role: "assistant", content: json } },
          ],
          usage: { prompt_tokens: 4, completion_tokens: 3, total_tokens: 7 },
        });
      return Response.json({
        output: { message: { role: "assistant", content: [{ text: json }] } },
        stopReason: "end_turn",
        usage: { inputTokens: 4, outputTokens: 3, totalTokens: 7 },
        metrics: { latencyMs: 1 },
      });
    };
    for (const entry of selected) {
      const { model } = resolveModel(entry.id, {
        catalog,
        env: { AWS_BEARER_TOKEN_BEDROCK: "fake" },
        fetch,
      });
      const result = await generateObject({
        model,
        prompt: "select",
        maxRetries: 0,
        schema: z.object({ selected_skill_ids: z.array(z.string()) }),
      });
      expect(result.object).toEqual({ selected_skill_ids: ["skill_1"] });
      const { url, body } = requests.at(-1) ?? { url: "", body: {} };
      expect(url).toContain(`.${entry.bedrockRegion}.`);
      expect(url).toContain(
        entry.bedrockEndpoint === "bedrock-mantle"
          ? "/openai/v1/chat/completions"
          : entry.bedrockApi === "responses"
            ? "/openai/v1/responses"
            : "/converse",
      );
      if (body.model) expect(body.model).toBe(entry.bedrockProfile);
      else expect(decodeURIComponent(url)).toContain(String(entry.bedrockProfile));
    }
  });

  it.each([
    "bedrock/openai.gpt-oss-120b-1:0",
    "bedrock/google.gemma-4-31b",
  ])("preserves BFCL tool state for selected %s", async (id) => {
    const bodies: Record<string, unknown>[] = [];
    const { model } = resolveModel(id, {
      env: { AWS_BEARER_TOKEN_BEDROCK: "fake" },
      fetch: async (_input, init) => {
        bodies.push(JSON.parse(String(init?.body)));
        const first = bodies.length === 1;
        if (id.includes("gemma"))
          return Response.json({
            id: `chatcmpl_${bodies.length}`,
            object: "chat.completion",
            created: 1,
            model: "google.gemma-4-31b",
            choices: [
              {
                index: 0,
                finish_reason: first ? "tool_calls" : "stop",
                message: first
                  ? {
                      role: "assistant",
                      content: null,
                      tool_calls: [
                        {
                          id: "call_1",
                          type: "function",
                          function: { name: "lookup", arguments: '{"city":"Rome"}' },
                        },
                      ],
                    }
                  : { role: "assistant", content: "Rome" },
              },
            ],
            usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
          });
        return Response.json({
          output: {
            message: {
              role: "assistant",
              content: first
                ? [{ toolUse: { toolUseId: "call_1", name: "lookup", input: { city: "Rome" } } }]
                : [{ text: "Rome" }],
            },
          },
          stopReason: first ? "tool_use" : "end_turn",
          usage: { inputTokens: 5, outputTokens: 3, totalTokens: 8 },
          metrics: { latencyMs: 1 },
        });
      },
    });
    const result = await generateText({
      model,
      prompt: "Find Rome",
      maxRetries: 0,
      stopWhen: stepCountIs(2),
      tools: {
        lookup: tool({
          inputSchema: z.object({ city: z.string() }),
          execute: async ({ city }) => ({ city }),
        }),
      },
    });
    expect(result.text).toBe("Rome");
    expect(result.steps[0].toolCalls).toMatchObject([{ toolCallId: "call_1", toolName: "lookup" }]);
    expect(JSON.stringify(bodies[1])).toContain("call_1");
  });
  it("routes Bedrock through its own profile and source region without global state", () => {
    const before = process.env.RATEL_LLM_BACKEND;
    const catalog = [
      {
        id: "bedrock/anthropic.claude-sonnet-5",
        bedrockProfile: "global.anthropic.claude-sonnet-5",
        bedrockRegion: "us-west-2",
        bedrockApi: "converse" as const,
      },
    ];
    const bedrock = resolveModel("bedrock/anthropic.claude-sonnet-5", { catalog });
    expect(bedrock.id).toBe("bedrock/anthropic.claude-sonnet-5");
    expect(bedrock.model).toMatchObject({
      modelId: "global.anthropic.claude-sonnet-5",
      provider: "amazon-bedrock",
    });
    expect(process.env.RATEL_LLM_BACKEND).toBe(before);
  });

  it("uses native direct adapters and rejects an unavailable route without fallback", () => {
    const env = { ANTHROPIC_API_KEY: "fake-anthropic", OPENAI_API_KEY: "fake-openai" };
    const anthropic = resolveModel("anthropic/claude-sonnet-5", { env });
    const openai = resolveModel("openai/gpt-6-sol", { env });
    expect(anthropic.model).toMatchObject({
      provider: "anthropic.messages",
      modelId: "claude-sonnet-5",
    });
    expect(openai.model).toMatchObject({ provider: "openai.responses", modelId: "gpt-6-sol" });
    expect(() => resolveModel("gcp/gemini-2.5-pro", { env })).toThrow(/GOOGLE_VERTEX_PROJECT/);
    expect(() =>
      resolveModel("gcp/gemini-3-pro-preview", { env: { GOOGLE_VERTEX_PROJECT: "test" } }),
    ).toThrow(/GOOGLE_VERTEX_LOCATION/);
    expect(() => resolveModel("anthropic/claude-sonnet-5", { env: {} })).toThrow(
      /ANTHROPIC_API_KEY/,
    );
  });

  it("uses the Bedrock runtime Responses endpoint for GPT reasoning", async () => {
    const requests: Array<{ url: string; headers: Headers; body: Record<string, unknown> }> = [];
    const model = resolveModel("bedrock/openai.gpt-6-sol", {
      catalog: [
        {
          id: "bedrock/openai.gpt-6-sol",
          bedrockProfile: "global.openai.gpt-6-sol",
          bedrockRegion: "us-east-1",
          bedrockEndpoint: "bedrock-runtime",
          bedrockApi: "responses",
        },
      ],
      awsCredentials: async () => ({ accessKeyId: "AKIDEXAMPLE", secretAccessKey: "secret" }),
      fetch: async (input, init) => {
        requests.push({
          url: String(input),
          headers: new Headers(init?.headers),
          body: JSON.parse(String(init?.body)),
        });
        return Response.json({
          id: "resp_1",
          object: "response",
          created_at: 1,
          model: "global.openai.gpt-6-sol",
          status: "completed",
          output: [
            {
              id: "msg_1",
              type: "message",
              status: "completed",
              role: "assistant",
              content: [{ type: "output_text", text: "done", annotations: [] }],
            },
          ],
          usage: { input_tokens: 3, output_tokens: 2, total_tokens: 5 },
        });
      },
    });
    const result = await generateText({ model: model.model, prompt: "ping", maxRetries: 0 });
    expect(result.text).toBe("done");
    expect(requests[0].url).toBe(
      "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/responses",
    );
    expect(requests[0].body.model).toBe("global.openai.gpt-6-sol");
    expect(requests[0].headers.get("authorization")).toContain("/us-east-1/bedrock/aws4_request");
  });

  it.each([
    "sol",
    "luna",
  ])("preserves Bedrock GPT-6 %s reasoning and tool IDs across BFCL turns", async (variant) => {
    const bodies: Record<string, unknown>[] = [];
    const model = resolveModel(`bedrock/openai.gpt-6-${variant}`, {
      catalog: [
        {
          id: `bedrock/openai.gpt-6-${variant}`,
          bedrockProfile: `global.openai.gpt-6-${variant}`,
          bedrockApi: "responses",
          bedrockRegion: "us-east-1",
        },
      ],
      awsCredentials: async () => ({ accessKeyId: "AKIDEXAMPLE", secretAccessKey: "secret" }),
      fetch: async (_input, init) => {
        bodies.push(JSON.parse(String(init?.body)));
        return Response.json({
          id: `resp_${bodies.length}`,
          object: "response",
          created_at: 1,
          model: `global.openai.gpt-6-${variant}`,
          status: "completed",
          output:
            bodies.length === 1
              ? [
                  { type: "reasoning", id: "rs_1", summary: [], encrypted_content: "opaque-state" },
                  {
                    type: "function_call",
                    id: "fc_1",
                    call_id: "call_1",
                    name: "lookup",
                    arguments: '{"city":"Rome"}',
                    status: "completed",
                  },
                ]
              : [
                  {
                    id: "msg_2",
                    type: "message",
                    status: "completed",
                    role: "assistant",
                    content: [{ type: "output_text", text: "Rome", annotations: [] }],
                  },
                ],
          usage: { input_tokens: 5, output_tokens: 3, total_tokens: 8 },
        });
      },
    });
    const result = await generateText({
      model: model.model,
      prompt: "Find Rome",
      maxRetries: 0,
      stopWhen: stepCountIs(2),
      tools: {
        lookup: tool({
          inputSchema: z.object({ city: z.string() }),
          execute: async ({ city }) => ({ city }),
        }),
      },
      providerOptions: {
        openai: { reasoningEffort: "medium" },
      },
    });
    expect(result.text).toBe("Rome");
    expect(result.steps[0].toolCalls).toMatchObject([{ toolCallId: "call_1", toolName: "lookup" }]);
    expect(bodies).toHaveLength(2);
    expect(bodies[0].reasoning).toEqual({ effort: "medium" });
    expect(bodies[0].store).toBe(false);
    expect(bodies[0].include).toContain("reasoning.encrypted_content");
    expect(JSON.stringify(bodies[1].input)).toContain("call_1");
    expect(JSON.stringify(bodies[1].input)).toContain("opaque-state");
  });

  it("sends equivalent text requests through Bedrock, Anthropic, and OpenAI", async () => {
    const requests: Array<{ url: string; body: Record<string, unknown> }> = [];
    const fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const body = JSON.parse(String(init?.body));
      requests.push({ url, body });
      if (url.includes("bedrock-runtime"))
        return Response.json({
          output: { message: { role: "assistant", content: [{ text: "ok" }] } },
          stopReason: "end_turn",
          usage: { inputTokens: 2, outputTokens: 1, totalTokens: 3 },
          metrics: { latencyMs: 1 },
        });
      if (url.includes("anthropic.com"))
        return Response.json({
          id: "msg_1",
          type: "message",
          role: "assistant",
          model: "claude-sonnet-5",
          content: [{ type: "text", text: "ok" }],
          stop_reason: "end_turn",
          usage: { input_tokens: 2, output_tokens: 1 },
        });
      return Response.json({
        id: "resp_1",
        object: "response",
        created_at: 1,
        model: "gpt-5.4-mini",
        status: "completed",
        output: [
          {
            id: "msg_1",
            type: "message",
            status: "completed",
            role: "assistant",
            content: [{ type: "output_text", text: "ok", annotations: [] }],
          },
        ],
        usage: { input_tokens: 2, output_tokens: 1, total_tokens: 3 },
      });
    };
    const cases = [
      resolveModel("bedrock/anthropic.claude-sonnet-5", {
        catalog: [
          {
            id: "bedrock/anthropic.claude-sonnet-5",
            bedrockProfile: "global.anthropic.claude-sonnet-5",
            bedrockRegion: "us-east-1",
          },
        ],
        awsCredentials: async () => ({ accessKeyId: "AKIDEXAMPLE", secretAccessKey: "secret" }),
        fetch,
      }),
      resolveModel("anthropic/claude-sonnet-5", { env: { ANTHROPIC_API_KEY: "fake" }, fetch }),
      resolveModel("openai/gpt-5.4-mini", { env: { OPENAI_API_KEY: "fake" }, fetch }),
    ];
    for (const { model } of cases) {
      expect((await generateText({ model, prompt: "ping", maxRetries: 0 })).text).toBe("ok");
    }
    expect(requests.map((request) => request.url)).toEqual([
      expect.stringContaining("/model/global.anthropic.claude-sonnet-5/converse"),
      "https://api.anthropic.com/v1/messages",
      "https://api.openai.com/v1/responses",
    ]);
  });

  it("returns SR structured selections through each native adapter", async () => {
    const json = '{"selected_skill_ids":["skill_1"]}';
    const seen: unknown[] = [];
    const fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      seen.push(JSON.parse(String(init?.body)));
      if (url.includes("bedrock-runtime"))
        return Response.json({
          output: { message: { role: "assistant", content: [{ text: json }] } },
          stopReason: "end_turn",
          usage: { inputTokens: 10, outputTokens: 6, totalTokens: 16 },
          metrics: { latencyMs: 1 },
        });
      if (url.includes("anthropic.com"))
        return Response.json({
          id: "msg_1",
          type: "message",
          role: "assistant",
          model: "claude-sonnet-5",
          content: [{ type: "text", text: json }],
          stop_reason: "end_turn",
          usage: { input_tokens: 10, output_tokens: 6 },
        });
      return Response.json({
        id: "resp_1",
        object: "response",
        created_at: 1,
        model: "gpt-5.4-mini",
        status: "completed",
        output: [
          {
            id: "msg_1",
            type: "message",
            status: "completed",
            role: "assistant",
            content: [{ type: "output_text", text: json, annotations: [] }],
          },
        ],
        usage: { input_tokens: 10, output_tokens: 6, total_tokens: 16 },
      });
    };
    const models = [
      resolveModel("bedrock/anthropic.claude-sonnet-5", {
        catalog: [
          {
            id: "bedrock/anthropic.claude-sonnet-5",
            bedrockProfile: "global.anthropic.claude-sonnet-5",
          },
        ],
        awsCredentials: async () => ({ accessKeyId: "AKIDEXAMPLE", secretAccessKey: "secret" }),
        fetch,
      }),
      resolveModel("anthropic/claude-sonnet-5", { env: { ANTHROPIC_API_KEY: "fake" }, fetch }),
      resolveModel("openai/gpt-5.4-mini", { env: { OPENAI_API_KEY: "fake" }, fetch }),
    ];
    for (const { id, model } of models) {
      const result = await generateObject({
        model,
        prompt: "select",
        schema: z.object({ selected_skill_ids: z.array(z.string()) }),
        maxRetries: 0,
      }).catch((error: unknown) => {
        throw new Error(`${id}: ${JSON.stringify(seen.at(-1))}`, { cause: error });
      });
      expect(result.object).toEqual({ selected_skill_ids: ["skill_1"] });
    }
  });

  it("retains BFCL tool call IDs and inputs through all three adapters", async () => {
    const fetch = async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("bedrock-runtime"))
        return Response.json({
          output: {
            message: {
              role: "assistant",
              content: [
                { toolUse: { toolUseId: "call_1", name: "lookup", input: { city: "Rome" } } },
              ],
            },
          },
          stopReason: "tool_use",
          usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
          metrics: { latencyMs: 1 },
        });
      if (url.includes("anthropic.com"))
        return Response.json({
          id: "msg_1",
          type: "message",
          role: "assistant",
          model: "claude-sonnet-5",
          content: [{ type: "tool_use", id: "call_1", name: "lookup", input: { city: "Rome" } }],
          stop_reason: "tool_use",
          usage: { input_tokens: 10, output_tokens: 5 },
        });
      return Response.json({
        id: "resp_1",
        object: "response",
        created_at: 1,
        model: "gpt-5.4-mini",
        status: "completed",
        output: [
          {
            type: "function_call",
            id: "fc_1",
            call_id: "call_1",
            name: "lookup",
            arguments: '{"city":"Rome"}',
            status: "completed",
          },
        ],
        usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
      });
    };
    const models = [
      resolveModel("bedrock/anthropic.claude-sonnet-5", {
        catalog: [
          {
            id: "bedrock/anthropic.claude-sonnet-5",
            bedrockProfile: "global.anthropic.claude-sonnet-5",
          },
        ],
        awsCredentials: async () => ({ accessKeyId: "AKIDEXAMPLE", secretAccessKey: "secret" }),
        fetch,
      }),
      resolveModel("anthropic/claude-sonnet-5", { env: { ANTHROPIC_API_KEY: "fake" }, fetch }),
      resolveModel("openai/gpt-5.4-mini", { env: { OPENAI_API_KEY: "fake" }, fetch }),
    ];
    for (const { model } of models) {
      const result = await generateText({
        model,
        prompt: "Find Rome",
        maxRetries: 0,
        tools: { lookup: tool({ inputSchema: z.object({ city: z.string() }) }) },
      });
      expect(result.toolCalls).toMatchObject([
        { toolCallId: "call_1", toolName: "lookup", input: { city: "Rome" } },
      ]);
    }
  });

  it("rejects a GPT-6 route that would silently use Converse", () => {
    expect(() =>
      resolveModel("bedrock/openai.gpt-6-sol", {
        catalog: [{ id: "bedrock/openai.gpt-6-sol", bedrockProfile: "global.openai.gpt-6-sol" }],
      }),
    ).toThrow(/Responses API/);
  });

  it("rejects a Gemma 4 route that would silently use runtime", () => {
    expect(() =>
      resolveModel("bedrock/google.gemma-4-31b", {
        catalog: [
          {
            id: "bedrock/google.gemma-4-31b",
            bedrockProfile: "google.gemma-4-31b",
            bedrockRegion: "eu-central-1",
            bedrockEndpoint: "bedrock-runtime",
            bedrockApi: "converse",
          },
        ],
      }),
    ).toThrow(/Mantle/);
  });

  it("uses the configured Mantle Chat endpoint for a Bedrock-only model", async () => {
    const urls: string[] = [];
    const { model } = resolveModel("bedrock/google.gemma-4-31b", {
      catalog: [
        {
          id: "bedrock/google.gemma-4-31b",
          bedrockProfile: "google.gemma-4-31b",
          bedrockRegion: "us-west-2",
          bedrockEndpoint: "bedrock-mantle",
          bedrockApi: "chat",
        },
      ],
      awsCredentials: async () => ({ accessKeyId: "AKIDEXAMPLE", secretAccessKey: "secret" }),
      fetch: async (input) => {
        urls.push(String(input));
        return Response.json({
          id: "chatcmpl_1",
          object: "chat.completion",
          created: 1,
          model: "google.gemma-4-31b",
          choices: [
            { index: 0, finish_reason: "stop", message: { role: "assistant", content: "ok" } },
          ],
          usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 },
        });
      },
    });
    expect((await generateText({ model, prompt: "ping", maxRetries: 0 })).text).toBe("ok");
    expect(urls).toEqual(["https://bedrock-mantle.us-west-2.api.aws/openai/v1/chat/completions"]);
  });

  it("uses Responses reasoning for a direct GPT-6 override", async () => {
    let request: Record<string, unknown> = {};
    const { model } = resolveModel("openai/gpt-6-sol", {
      env: { OPENAI_API_KEY: "fake" },
      fetch: async (_input, init) => {
        request = JSON.parse(String(init?.body));
        return Response.json({
          id: "resp_1",
          object: "response",
          created_at: 1,
          model: "gpt-6-sol",
          status: "completed",
          output: [
            {
              id: "msg_1",
              type: "message",
              status: "completed",
              role: "assistant",
              content: [{ type: "output_text", text: "ok", annotations: [] }],
            },
          ],
          usage: { input_tokens: 2, output_tokens: 1, total_tokens: 3 },
        });
      },
    });
    expect(
      (
        await generateText({
          model,
          prompt: "ping",
          maxRetries: 0,
          providerOptions: { openai: { reasoningEffort: "medium" } },
        })
      ).text,
    ).toBe("ok");
    expect(request.reasoning).toEqual({ effort: "medium" });
    expect(request.store).toBe(false);
  });

  it("accepts a structured SR result from Bedrock GPT-6 Responses", async () => {
    let request: Record<string, unknown> = {};
    const { model } = resolveModel("bedrock/openai.gpt-6-sol", {
      catalog: [
        {
          id: "bedrock/openai.gpt-6-sol",
          bedrockProfile: "global.openai.gpt-6-sol",
          bedrockApi: "responses",
          bedrockRegion: "us-east-1",
        },
      ],
      awsCredentials: async () => ({ accessKeyId: "AKIDEXAMPLE", secretAccessKey: "secret" }),
      fetch: async (_input, init) => {
        request = JSON.parse(String(init?.body));
        return Response.json({
          id: "resp_1",
          object: "response",
          created_at: 1,
          model: "global.openai.gpt-6-sol",
          status: "completed",
          output: [
            {
              id: "msg_1",
              type: "message",
              status: "completed",
              role: "assistant",
              content: [
                {
                  type: "output_text",
                  text: '{"selected_skill_ids":["skill_1"]}',
                  annotations: [],
                },
              ],
            },
          ],
          usage: { input_tokens: 10, output_tokens: 6, total_tokens: 16 },
        });
      },
    });
    const result = await generateObject({
      model,
      prompt: "select",
      schema: z.object({ selected_skill_ids: z.array(z.string()) }),
      maxRetries: 0,
      providerOptions: { openai: { reasoningEffort: "medium" } },
    });
    expect(result.object).toEqual({ selected_skill_ids: ["skill_1"] });
    expect(request.reasoning).toEqual({ effort: "medium" });
    expect(request.text).toMatchObject({ format: { type: "json_schema" } });
  });

  it("rejects unsupported catalog APIs and endpoints before inference", () => {
    const entry = {
      id: "bedrock/anthropic.claude-sonnet-5",
      bedrockProfile: "global.anthropic.claude-sonnet-5",
    };
    expect(() =>
      resolveModel(entry.id, { catalog: [{ ...entry, bedrockApi: "messages" as never }] }),
    ).toThrow(/unsupported Bedrock API/);
    expect(() =>
      resolveModel(entry.id, { catalog: [{ ...entry, bedrockEndpoint: "proxy" as never }] }),
    ).toThrow(/unsupported Bedrock endpoint/);
  });

  it("keeps Ollama and hosted URLs on their existing Chat routes", () => {
    const ollama = resolveModel("ollama:qwen3.5", { ollamaBaseURL: "http://localhost:11434/v1" });
    const hosted = resolveModel("https://models.example.com/v1#meta/llama", {
      modelApiKey: "fake",
    });
    expect(ollama).toMatchObject({
      id: "ollama:qwen3.5",
      model: { provider: "openai.chat", modelId: "qwen3.5" },
    });
    expect(hosted).toMatchObject({
      id: "https://models.example.com/v1#meta/llama",
      model: { provider: "openai.chat", modelId: "meta/llama" },
    });
  });

  it("uses Bedrock bearer credentials without reading direct provider keys", async () => {
    let authorization: string | null = null;
    const { model } = resolveModel("bedrock/anthropic.claude-sonnet-5", {
      env: { AWS_BEARER_TOKEN_BEDROCK: "fake-bedrock", AWS_REGION: "us-east-1" },
      catalog: [
        {
          id: "bedrock/anthropic.claude-sonnet-5",
          bedrockProfile: "global.anthropic.claude-sonnet-5",
        },
      ],
      fetch: async (_input, init) => {
        authorization = new Headers(init?.headers).get("authorization");
        return Response.json({
          output: { message: { role: "assistant", content: [{ text: "ok" }] } },
          stopReason: "end_turn",
          usage: { inputTokens: 2, outputTokens: 1, totalTokens: 3 },
          metrics: { latencyMs: 1 },
        });
      },
    });
    expect((await generateText({ model, prompt: "ping", maxRetries: 0 })).text).toBe("ok");
    expect(authorization).toBe("Bearer fake-bedrock");
  });

  it("keeps reasoning enabled on Bedrock Mantle GPT-6 Responses", async () => {
    let request: Record<string, unknown> = {};
    const { model } = resolveModel("bedrock/openai.gpt-6-sol", {
      catalog: [
        {
          id: "bedrock/openai.gpt-6-sol",
          bedrockProfile: "openai.gpt-6-sol",
          bedrockEndpoint: "bedrock-mantle",
          bedrockApi: "responses",
          bedrockRegion: "us-west-2",
        },
      ],
      awsCredentials: async () => ({ accessKeyId: "AKIDEXAMPLE", secretAccessKey: "secret" }),
      fetch: async (_input, init) => {
        request = JSON.parse(String(init?.body));
        return Response.json({
          id: "resp_1",
          object: "response",
          created_at: 1,
          model: "openai.gpt-6-sol",
          status: "completed",
          output: [
            {
              id: "msg_1",
              type: "message",
              status: "completed",
              role: "assistant",
              content: [{ type: "output_text", text: "ok", annotations: [] }],
            },
          ],
          usage: { input_tokens: 2, output_tokens: 1, total_tokens: 3 },
        });
      },
    });
    expect(
      (
        await generateText({
          model,
          prompt: "ping",
          maxRetries: 0,
          providerOptions: { openai: { reasoningEffort: "medium" } },
        })
      ).text,
    ).toBe("ok");
    expect(request.reasoning).toEqual({ effort: "medium" });
    expect(request.store).toBe(false);
  });
});
