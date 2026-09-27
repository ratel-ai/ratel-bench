import type { ExecutableTool } from "@ratel-ai/sdk";
import { APICallError } from "ai";
import { MockLanguageModelV3 } from "ai/test";
import { describe, expect, it } from "vitest";
import { CellTimeoutError } from "../cell-errors.js";
import type { AgentRunInput, ToolSpec } from "../types.js";
import {
  buildToolBundle,
  emptyToolBundle,
  normalizeInputSchema,
  registerGateway,
  runMeteredLoop,
  sanitizeToolName,
  withTimeout,
} from "./_shared.js";

describe("sanitizeToolName", () => {
  it("leaves ids that already match the provider pattern unchanged", () => {
    expect(sanitizeToolName("read_file")).toBe("read_file");
    expect(sanitizeToolName("search-tools")).toBe("search-tools");
  });

  it("replaces invalid characters with underscores", () => {
    expect(sanitizeToolName("fs.read_file")).toBe("fs_read_file");
    expect(sanitizeToolName("api/v2/get")).toBe("api_v2_get");
    expect(sanitizeToolName("foo:bar baz")).toBe("foo_bar_baz");
  });

  it("trims leading/trailing underscores left over from sanitization", () => {
    expect(sanitizeToolName(".dotted.")).toBe("dotted");
  });

  it("throws when sanitization yields an empty string", () => {
    expect(() => sanitizeToolName("...")).toThrow(/empty/);
  });
});

describe("normalizeInputSchema", () => {
  it("defaults type to object when missing", () => {
    expect(normalizeInputSchema({})).toEqual({ type: "object" });
  });

  it("leaves a valid object schema untouched", () => {
    const schema = { type: "object", properties: { x: { type: "string" } } };
    expect(normalizeInputSchema(schema)).toEqual(schema);
  });

  it("treats null/undefined/non-object as empty", () => {
    expect(normalizeInputSchema(null)).toEqual({ type: "object" });
    expect(normalizeInputSchema(undefined)).toEqual({ type: "object" });
    expect(normalizeInputSchema("nope")).toEqual({ type: "object" });
  });
});

describe("buildToolBundle", () => {
  const specs: ToolSpec[] = [
    {
      id: "fs.read_file",
      name: "read_file",
      description: "Read a file from disk.",
      input_schema: { type: "object" },
    },
    {
      id: "mail.send",
      name: "send_email",
      description: "Send an email.",
      input_schema: {},
    },
  ];

  it("registers each spec under its sanitized name and tracks canonical ids", () => {
    const bundle = buildToolBundle(specs);
    expect(Object.keys(bundle.tools).sort()).toEqual(["fs_read_file", "mail_send"]);
    expect(bundle.activeToolIds).toEqual(["fs.read_file", "mail.send"]);
    expect(bundle.nameToId.get("fs_read_file")).toBe("fs.read_file");
    expect(bundle.nameToId.get("mail_send")).toBe("mail.send");
  });

  it("disambiguates distinct ids that sanitize to the same provider name", () => {
    // BFCL contains exactly this shape (e.g. solve.quadratic_equation vs
    // solve_quadratic_equation); both must register with the gold id recoverable.
    const collision: ToolSpec[] = [
      { ...specs[0], id: "solve.quadratic_equation" },
      { ...specs[0], id: "solve_quadratic_equation" },
    ];
    const bundle = buildToolBundle(collision);
    // Both canonical ids are present (nothing dropped).
    expect(bundle.activeToolIds.sort()).toEqual([
      "solve.quadratic_equation",
      "solve_quadratic_equation",
    ]);
    // Two distinct provider-safe names: the base + a suffixed variant.
    expect(Object.keys(bundle.tools).sort()).toEqual([
      "solve_quadratic_equation",
      "solve_quadratic_equation_2",
    ]);
    // Each name maps back to the right canonical id (so judging stays correct).
    expect(new Set(bundle.nameToId.values())).toEqual(
      new Set(["solve.quadratic_equation", "solve_quadratic_equation"]),
    );
  });

  it("treats a repeated identical id as a no-op", () => {
    const dup: ToolSpec[] = [
      { ...specs[0], id: "fs.read_file" },
      { ...specs[0], id: "fs.read_file" },
    ];
    const bundle = buildToolBundle(dup);
    expect(bundle.activeToolIds).toEqual(["fs.read_file"]);
    expect(Object.keys(bundle.tools)).toEqual(["fs_read_file"]);
  });

  it("normalizes empty schemas at the AI SDK boundary", () => {
    const bundle = buildToolBundle([specs[1]]);
    const t = bundle.tools.mail_send as unknown as {
      inputSchema: { jsonSchema: { type?: string } };
    };
    expect(t.inputSchema.jsonSchema.type).toBe("object");
  });
});

describe("registerGateway", () => {
  const stub: ExecutableTool = {
    id: "search_tools",
    name: "search_tools",
    description: "stub",
    inputSchema: { type: "object" },
    outputSchema: { type: "object" },
    execute: async () => ({}),
  };

  it("registers the gateway tool and counts it toward activeToolIds", () => {
    // The catalog column in the report reads activeToolIds.length, and we
    // want gateway tools (search_tools / invoke_tool) to count there — the
    // agent really did see them, even though they aren't direct tools.
    const bundle = emptyToolBundle();
    registerGateway(stub, bundle);
    expect(Object.keys(bundle.tools)).toEqual(["search_tools"]);
    expect(bundle.activeToolIds).toEqual(["search_tools"]);
    expect(bundle.nameToId.size).toBe(0);
  });

  it("throws on duplicate registration", () => {
    const bundle = emptyToolBundle();
    registerGateway(stub, bundle);
    expect(() => registerGateway(stub, bundle)).toThrow(/already registered/);
  });
});

describe("withTimeout", () => {
  it("rejects with CellTimeoutError, message unchanged", async () => {
    const never = new Promise<never>(() => {});
    const err = await withTimeout(never, 5).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(CellTimeoutError);
    expect((err as Error).message).toBe("run timed out after 5ms");
  });
});

describe("runMeteredLoop", () => {
  const spec: ToolSpec = {
    id: "fs.read_file",
    name: "read_file",
    description: "Read a file from disk.",
    input_schema: { type: "object", properties: { path: { type: "string" } } },
  };

  const usage = (input: number, output: number) => ({
    inputTokens: { total: input, noCache: input, cacheRead: 0, cacheWrite: 0 },
    outputTokens: { total: output, text: output, reasoning: 0 },
  });

  function input(model: MockLanguageModelV3): AgentRunInput {
    return {
      scenario: {
        id: "s-1",
        prompt: "read /etc/hosts",
        candidate_pool: [spec],
        gold_tools: [spec.id],
      },
      pool: [spec],
      poolSize: 1,
      model: { id: "priced-model", model },
      runIndex: 0,
      topK: 5,
      retriever: "bm25",
      maxSteps: 5,
      perRunTimeoutMs: 5_000,
      seed: 1,
      pricing: {
        "priced-model": {
          inputPer1M: 1,
          outputPer1M: 5,
          cachedInputPer1M: 0,
          cacheCreationPer1M: 0,
        },
      },
    };
  }

  it("keeps usage of completed steps when a later step throws", async () => {
    let call = 0;
    const model = new MockLanguageModelV3({
      doGenerate: async () => {
        call++;
        if (call === 1) {
          return {
            content: [
              {
                type: "tool-call",
                toolCallId: "c1",
                toolName: "fs_read_file",
                input: JSON.stringify({ path: "/etc/hosts" }),
              },
            ],
            finishReason: { unified: "tool-calls", raw: "tool_use" },
            usage: usage(120, 30),
            warnings: [],
          };
        }
        // Non-retryable, so the SDK's real backoff never runs.
        throw new APICallError({
          message: "tools: too many tools",
          url: "https://api.test/v1",
          requestBodyValues: {},
          statusCode: 400,
          isRetryable: false,
        });
      },
    });

    const cell = await runMeteredLoop("control-baseline", input(model), buildToolBundle([spec]));

    expect(call).toBe(2);
    expect(cell.error).toMatch(/too many tools/);
    expect(cell.error_class).toBe("request");
    expect(cell.input_tokens).toBe(120);
    expect(cell.output_tokens).toBe(30);
    expect(cell.max_step_output_tokens).toBe(30);
    expect(cell.dollar_cost).toBeGreaterThan(0);
    // The partial trace is not scored: an errored cell stays a fail.
    expect(cell.effective_tool_ids).toEqual([]);
    expect(cell.turns).toBe(0);
  });

  it("stamps provider = model.provider and records per-step truncation", async () => {
    const model = new MockLanguageModelV3({
      provider: "mock-provider",
      doGenerate: async () => ({
        content: [{ type: "text", text: "partial" }],
        finishReason: { unified: "length", raw: "max_tokens" },
        usage: usage(10, 64),
        warnings: [],
      }),
    });

    const cell = await runMeteredLoop("control-baseline", input(model), buildToolBundle([spec]));

    expect(cell.error).toBeNull();
    expect(cell.provider).toBe("mock-provider");
    expect(cell.finish_reason).toBe("length");
    expect(cell.truncated_steps).toBe(1);
    expect(cell.max_step_output_tokens).toBe(64);
  });
});
