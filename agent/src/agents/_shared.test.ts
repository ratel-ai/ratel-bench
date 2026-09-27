import type { ExecutableTool } from "@ratel-ai/sdk";
import { APICallError } from "ai";
import { MockLanguageModelV3 } from "ai/test";
import { describe, expect, it, vi } from "vitest";
import { type RetrySettings, sleep as realSleep } from "../llm-retry.js";
import type { AgentRunInput, ToolSpec } from "../types.js";
import {
  buildToolBundle,
  emptyToolBundle,
  normalizeInputSchema,
  registerGateway,
  runMeteredLoop,
  sanitizeToolName,
} from "./_shared.js";

// Passthrough `ai`, except ToolLoopAgent records the settings it is built with.
const { agentSettings } = vi.hoisted(() => ({
  agentSettings: [] as Array<{ maxRetries?: number }>,
}));
vi.mock("ai", async () => {
  const actual = await vi.importActual<typeof import("ai")>("ai");
  class ToolLoopAgent<
    // biome-ignore lint/suspicious/noExplicitAny: ToolLoopAgent's own generic defaults
    C = any,
    // biome-ignore lint/suspicious/noExplicitAny: ToolLoopAgent's own generic defaults
    T extends Record<string, any> = Record<string, any>,
  > extends actual.ToolLoopAgent<C, T> {
    constructor(settings: ConstructorParameters<typeof actual.ToolLoopAgent<C, T>>[0]) {
      agentSettings.push(settings);
      super(settings);
    }
  }
  return { ...actual, ToolLoopAgent };
});

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

  function input(
    model: MockLanguageModelV3,
    maxOutputTokens: number | null = null,
    over: Partial<AgentRunInput> = {},
  ): AgentRunInput {
    return {
      scenario: {
        id: "s-1",
        prompt: "read /etc/hosts",
        candidate_pool: [spec],
        gold_tools: [spec.id],
      },
      pool: [spec],
      poolSize: 1,
      model: { id: "priced-model", model, maxOutputTokens },
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
      ...over,
    };
  }

  /** Retry settings with an injected sleep: no real backoff ever runs. */
  function retry(sleep: RetrySettings["sleep"], graceMs = 30_000): RetrySettings {
    return {
      policy: { maxAttempts: 4, baseMs: 1000, maxDelayMs: 8000, maxTotalWaitMs: 60_000 },
      graceMs,
      sleep,
      random: () => 0,
    };
  }

  const throttled = () =>
    new APICallError({
      message: "Too Many Requests",
      url: "https://api.test/v1",
      requestBodyValues: {},
      statusCode: 429,
      isRetryable: true,
    });

  const done = {
    content: [{ type: "text" as const, text: "done" }],
    finishReason: { unified: "stop" as const, raw: "end_turn" },
    usage: usage(10, 5),
    warnings: [],
  };

  /** A doGenerate that settles only when its abort signal fires (with the abort reason). */
  function hangUntilAborted(signals: AbortSignal[]) {
    return ({ abortSignal }: { abortSignal?: AbortSignal }) => {
      if (abortSignal) signals.push(abortSignal);
      return new Promise<never>((_resolve, reject) => {
        abortSignal?.addEventListener("abort", () => reject(abortSignal.reason), { once: true });
      });
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

  it("every doGenerate call receives the model's maxOutputTokens; the cell stamps it", async () => {
    let call = 0;
    const model = new MockLanguageModelV3({
      doGenerate: async () => {
        call++;
        if (call === 1) {
          return {
            content: [
              { type: "tool-call", toolCallId: "c1", toolName: "fs_read_file", input: "{}" },
            ],
            finishReason: { unified: "tool-calls", raw: "tool_use" },
            usage: usage(10, 5),
            warnings: [],
          };
        }
        return {
          content: [{ type: "text", text: "done" }],
          finishReason: { unified: "stop", raw: "end_turn" },
          usage: usage(10, 5),
          warnings: [],
        };
      },
    });

    const cell = await runMeteredLoop(
      "control-baseline",
      input(model, 1234),
      buildToolBundle([spec]),
    );

    expect(model.doGenerateCalls).toHaveLength(2);
    expect(model.doGenerateCalls.map((c) => c.maxOutputTokens)).toEqual([1234, 1234]);
    expect(cell.max_output_tokens).toBe(1234);
  });

  it("a null cap sends no maxOutputTokens and stamps null", async () => {
    const model = new MockLanguageModelV3({
      doGenerate: async () => ({
        content: [{ type: "text", text: "done" }],
        finishReason: { unified: "stop", raw: "end_turn" },
        usage: usage(10, 5),
        warnings: [],
      }),
    });

    const cell = await runMeteredLoop("control-baseline", input(model), buildToolBundle([spec]));

    expect(model.doGenerateCalls[0].maxOutputTokens).toBeUndefined();
    expect(cell.max_output_tokens).toBeNull();
  });

  it("stamps the cap on an errored cell too", async () => {
    const model = new MockLanguageModelV3({
      doGenerate: async () => {
        throw new APICallError({
          message: "bad request",
          url: "https://api.test/v1",
          requestBodyValues: {},
          statusCode: 400,
          isRetryable: false,
        });
      },
    });

    const cell = await runMeteredLoop(
      "control-baseline",
      input(model, 64),
      buildToolBundle([spec]),
    );

    expect(cell.error_class).toBe("request");
    expect(cell.max_output_tokens).toBe(64);
  });

  it("records retries; retry waits are not charged to the timeout", async () => {
    vi.useFakeTimers();
    try {
      let call = 0;
      const model = new MockLanguageModelV3({
        doGenerate: async () => {
          if (call++ === 0) throw throttled();
          return done;
        },
      });
      // An abort-aware backoff of 5s (fake time), past the 1s active-time deadline:
      // were the deadline not paused, it would fire mid-sleep and abort the cell.
      const sleep: RetrySettings["sleep"] = (_ms, signal) => realSleep(5000, signal);

      const pending = runMeteredLoop(
        "control-baseline",
        input(model, null, { perRunTimeoutMs: 1000, retry: retry(sleep) }),
        buildToolBundle([spec]),
      );
      await vi.advanceTimersByTimeAsync(6000);
      const cell = await pending;

      expect(cell.error).toBeNull();
      expect(model.doGenerateCalls).toHaveLength(2);
      expect(model.doGenerateCalls.every((c) => c.abortSignal !== undefined)).toBe(true);
      expect(cell.retries).toBe(1);
      expect(cell.throttled_retries).toBe(1);
      expect(cell.retry_wait_ms).toBe(500); // rng=0 → d/2 with base 1000
      expect(cell.retry_policy).toBe("a4/b1000/c8000/w60000;timeout=active:1000+g30000");
      expect(cell.wall_ms).toBeGreaterThanOrEqual(5000);
    } finally {
      vi.useRealTimers();
    }
  });

  it("aborts a model that honours the signal: doGenerate's abortSignal fires at the timeout", async () => {
    const signals: AbortSignal[] = [];
    const model = new MockLanguageModelV3({ doGenerate: hangUntilAborted(signals) });

    const cell = await runMeteredLoop(
      "control-baseline",
      input(model, null, { perRunTimeoutMs: 30, retry: retry(async () => {}) }),
      buildToolBundle([spec]),
    );

    expect(signals).toHaveLength(1);
    expect(signals[0].aborted).toBe(true);
    expect(cell.error).toBe("run timed out after 30ms");
    expect(cell.error_class).toBe("timeout");
    expect(cell.retries).toBe(0);
  });

  it("a model that ignores the abort still yields a timeout cell at deadline + grace", async () => {
    const model = new MockLanguageModelV3({ doGenerate: () => new Promise<never>(() => {}) });

    const startedAt = Date.now();
    const cell = await runMeteredLoop(
      "control-baseline",
      input(model, null, { perRunTimeoutMs: 20, retry: retry(async () => {}, 30) }),
      buildToolBundle([spec]),
    );

    expect(cell.error).toBe("run timed out after 20ms");
    expect(cell.error_class).toBe("timeout");
    expect(Date.now() - startedAt).toBeGreaterThanOrEqual(45);
    expect(cell.retry_policy).toBe("a4/b1000/c8000/w60000;timeout=active:20+g30");
  });

  it("a timeout after retries>0 is error_class 'transient'", async () => {
    const signals: AbortSignal[] = [];
    const hang = hangUntilAborted(signals);
    let call = 0;
    const model = new MockLanguageModelV3({
      doGenerate: async (options) => {
        if (call++ === 0) throw throttled();
        return hang(options);
      },
    });

    const cell = await runMeteredLoop(
      "control-baseline",
      input(model, null, { perRunTimeoutMs: 30, retry: retry(async () => {}) }),
      buildToolBundle([spec]),
    );

    expect(cell.error).toBe("run timed out after 30ms");
    expect(cell.retries).toBe(1);
    expect(cell.error_class).toBe("transient");
  });

  it("builds the agent loop with the SDK's retries off (maxRetries: 0)", async () => {
    const model = new MockLanguageModelV3({ doGenerate: async () => done });

    await runMeteredLoop("control-baseline", input(model), buildToolBundle([spec]));

    expect(agentSettings.at(-1)?.maxRetries).toBe(0);
  });

  it("an error withRetry passes through is not retried by the SDK either", async () => {
    // Retryable to the SDK but an `outcome` to withRetry, which rethrows it as-is.
    // (Were the SDK's retries on, its real 2s+4s backoff would time this test out.)
    const model = new MockLanguageModelV3({
      doGenerate: async () => {
        throw new APICallError({
          message: "prompt is too long",
          url: "https://api.test/v1",
          requestBodyValues: {},
          statusCode: 500,
          isRetryable: true,
        });
      },
    });

    const cell = await runMeteredLoop(
      "control-baseline",
      input(model, null, { retry: retry(async () => {}) }),
      buildToolBundle([spec]),
    );

    expect(model.doGenerateCalls).toHaveLength(1);
    expect(cell.error_class).toBe("outcome");
    expect(cell.retries).toBe(0);
  });

  it("an exhausted retry budget is a transient cell; a fatal error an access cell", async () => {
    const exhausted = new MockLanguageModelV3({
      doGenerate: async () => {
        throw throttled();
      },
    });
    const cell = await runMeteredLoop(
      "control-baseline",
      input(exhausted, null, { retry: retry(async () => {}) }),
      buildToolBundle([spec]),
    );
    expect(exhausted.doGenerateCalls).toHaveLength(4);
    expect(cell.error).toBe("Failed after 4 attempts. Last error: Too Many Requests");
    expect(cell.error_class).toBe("transient");
    expect(cell.retries).toBe(3);

    const gated = new MockLanguageModelV3({
      doGenerate: async () => {
        throw new APICallError({
          message: "Forbidden",
          url: "https://api.test/v1",
          requestBodyValues: {},
          statusCode: 403,
        });
      },
    });
    const fatal = await runMeteredLoop(
      "control-baseline",
      input(gated, null, { retry: retry(async () => {}) }),
      buildToolBundle([spec]),
    );
    expect(gated.doGenerateCalls).toHaveLength(1);
    expect(fatal.error).toBe("Forbidden");
    expect(fatal.error_class).toBe("access");
  });
});
