import { NoObjectGeneratedError, RetryError } from "ai";
import { describe, expect, it, vi } from "vitest";
import { judgeLLM } from "./llm.js";

vi.mock("ai", async () => {
  const actual = await vi.importActual<typeof import("ai")>("ai");
  return {
    ...actual,
    generateObject: vi.fn(),
  };
});

describe("judgeLLM", () => {
  it("returns n/a when the model call throws (criteria path)", async () => {
    const result = await judgeLLM({
      prompt: "what is 2+2?",
      judgeCriteria: "must mention 4",
      finalText: "the answer is 4",
      // biome-ignore lint/suspicious/noExplicitAny: invalid model surfaces a thrown error from generateObject
      model: { specificationVersion: "v0" } as any,
    });
    expect(result.verdict).toBe("n/a");
    expect(result.explanation).toMatch(/judge failed/);
  });

  it("falls back to prompt-only judging when criteria is empty (the MetaTool case)", async () => {
    const ai = await import("ai");
    const mock = vi.mocked(ai.generateObject);
    mock.mockResolvedValueOnce({
      object: { verdict: "pass", explanation: "addresses the request" },
      // biome-ignore lint/suspicious/noExplicitAny: only `object` matters for this test path
    } as any);

    const result = await judgeLLM({
      prompt: "what's the weather in Paris?",
      finalText: "Paris is currently 18°C and partly cloudy.",
      // biome-ignore lint/suspicious/noExplicitAny: model call is mocked
      model: {} as any,
    });

    expect(result.verdict).toBe("pass");
    // Verify the judge was invoked with the prompt-only system, not the criteria one.
    const callArgs = mock.mock.calls.at(-1)?.[0];
    expect(callArgs?.system).toContain("USER_REQUEST");
    expect(callArgs?.system).not.toContain("SUCCESS_CRITERIA");
    expect(callArgs?.prompt).toContain("USER_REQUEST:");
    expect(callArgs?.prompt).toContain("what's the weather in Paris?");
  });

  it("uses the criteria-based system when judgeCriteria is non-empty", async () => {
    const ai = await import("ai");
    const mock = vi.mocked(ai.generateObject);
    mock.mockResolvedValueOnce({
      object: { verdict: "fail", explanation: "missing localhost" },
      // biome-ignore lint/suspicious/noExplicitAny: only `object` matters for this test path
    } as any);

    const result = await judgeLLM({
      prompt: "show /etc/hosts",
      judgeCriteria: "must mention localhost",
      finalText: "the file contains some IP addresses",
      // biome-ignore lint/suspicious/noExplicitAny: model call is mocked
      model: {} as any,
    });

    expect(result.verdict).toBe("fail");
    const callArgs = mock.mock.calls.at(-1)?.[0];
    expect(callArgs?.system).toContain("SUCCESS_CRITERIA");
    expect(callArgs?.system).not.toContain("USER_REQUEST");
    expect(callArgs?.prompt).toContain("SUCCESS_CRITERIA:");
    expect(callArgs?.prompt).toContain("must mention localhost");
  });

  it("treats whitespace-only criteria as missing (falls through to prompt-only)", async () => {
    const ai = await import("ai");
    const mock = vi.mocked(ai.generateObject);
    mock.mockResolvedValueOnce({
      object: { verdict: "pass", explanation: "ok" },
      // biome-ignore lint/suspicious/noExplicitAny: only `object` matters for this test path
    } as any);

    await judgeLLM({
      prompt: "say hi",
      judgeCriteria: "   \n  ",
      finalText: "hi",
      // biome-ignore lint/suspicious/noExplicitAny: model call is mocked
      model: {} as any,
    });

    const callArgs = mock.mock.calls.at(-1)?.[0];
    expect(callArgs?.system).toContain("USER_REQUEST");
  });

  it("substitutes (empty) for blank assistant output", async () => {
    const ai = await import("ai");
    const mock = vi.mocked(ai.generateObject);
    mock.mockResolvedValueOnce({
      object: { verdict: "fail", explanation: "no output" },
      // biome-ignore lint/suspicious/noExplicitAny: only `object` matters for this test path
    } as any);

    await judgeLLM({
      prompt: "do something",
      finalText: "",
      // biome-ignore lint/suspicious/noExplicitAny: model call is mocked
      model: {} as any,
    });

    const callArgs = mock.mock.calls.at(-1)?.[0];
    expect(callArgs?.prompt).toContain("(empty)");
  });

  it("uses the strict system prompt when promptVariant: 'strict' is passed (no criteria)", async () => {
    const ai = await import("ai");
    const mock = vi.mocked(ai.generateObject);
    mock.mockResolvedValueOnce({
      object: { verdict: "fail", explanation: "polite refusal, no substantive answer" },
      // biome-ignore lint/suspicious/noExplicitAny: only `object` matters for this test path
    } as any);

    await judgeLLM({
      prompt: "List the top 5 trending repos",
      finalText: "I don't have a tool for that, try GitHub Trending",
      // biome-ignore lint/suspicious/noExplicitAny: model call is mocked
      model: {} as any,
      promptVariant: "strict",
    });

    const callArgs = mock.mock.calls.at(-1)?.[0];
    // Still the no-criteria branch — uses USER_REQUEST framing.
    expect(callArgs?.system).toContain("USER_REQUEST");
    // Strict variant must forbid rewarding refusals / redirects.
    expect(callArgs?.system).toMatch(/refus|redirect|substantive|attempt/i);
    // And must NOT carry the lenient coherence hedge.
    expect(callArgs?.system).not.toContain(
      "Don't penalize the assistant for not naming a particular tool",
    );
  });

  it("defaults to the strict prompt when promptVariant is omitted", async () => {
    const ai = await import("ai");
    const mock = vi.mocked(ai.generateObject);
    mock.mockResolvedValueOnce({
      object: { verdict: "fail", explanation: "polite refusal" },
      // biome-ignore lint/suspicious/noExplicitAny: only `object` matters for this test path
    } as any);

    await judgeLLM({
      prompt: "do something",
      finalText: "I can't but here are alternatives",
      // biome-ignore lint/suspicious/noExplicitAny: model call is mocked
      model: {} as any,
    });

    const callArgs = mock.mock.calls.at(-1)?.[0];
    expect(callArgs?.system).toMatch(/refus|redirect|substantive|attempt/i);
    expect(callArgs?.system).not.toContain(
      "Don't penalize the assistant for not naming a particular tool",
    );
  });

  it("uses the lenient coherence prompt when promptVariant: 'coherence' is explicitly passed", async () => {
    const ai = await import("ai");
    const mock = vi.mocked(ai.generateObject);
    mock.mockResolvedValueOnce({
      object: { verdict: "pass", explanation: "ok" },
      // biome-ignore lint/suspicious/noExplicitAny: only `object` matters for this test path
    } as any);

    await judgeLLM({
      prompt: "do something",
      finalText: "I can't but here are alternatives",
      // biome-ignore lint/suspicious/noExplicitAny: model call is mocked
      model: {} as any,
      promptVariant: "coherence",
    });

    const callArgs = mock.mock.calls.at(-1)?.[0];
    expect(callArgs?.system).toContain(
      "Don't penalize the assistant for not naming a particular tool",
    );
  });

  it("promptVariant has no effect on the criteria path", async () => {
    const ai = await import("ai");
    const mock = vi.mocked(ai.generateObject);
    mock.mockResolvedValueOnce({
      object: { verdict: "pass", explanation: "ok" },
      // biome-ignore lint/suspicious/noExplicitAny: only `object` matters for this test path
    } as any);

    await judgeLLM({
      prompt: "p",
      judgeCriteria: "must mention X",
      finalText: "X",
      // biome-ignore lint/suspicious/noExplicitAny: model call is mocked
      model: {} as any,
      promptVariant: "strict",
    });

    const callArgs = mock.mock.calls.at(-1)?.[0];
    expect(callArgs?.system).toContain("SUCCESS_CRITERIA");
    expect(callArgs?.system).not.toContain("USER_REQUEST");
  });

  it("forwards maxOutputTokens to the judge call (unset → not sent)", async () => {
    const ai = await import("ai");
    const mock = vi.mocked(ai.generateObject);
    const ok = { object: { verdict: "pass", explanation: "ok" } };
    // biome-ignore lint/suspicious/noExplicitAny: only `object` matters for this test path
    mock.mockResolvedValueOnce(ok as any).mockResolvedValueOnce(ok as any);

    // biome-ignore lint/suspicious/noExplicitAny: model call is mocked
    await judgeLLM({ prompt: "p", finalText: "x", model: {} as any, maxOutputTokens: 512 });
    expect(mock.mock.calls.at(-1)?.[0].maxOutputTokens).toBe(512);

    // biome-ignore lint/suspicious/noExplicitAny: model call is mocked
    await judgeLLM({ prompt: "p", finalText: "x", model: {} as any });
    expect(mock.mock.calls.at(-1)?.[0].maxOutputTokens).toBeUndefined();
  });

  const noObject = (
    finishReason: "length" | "stop" = "length",
    message = "No object generated: could not parse the response.",
    text = '{"verdict": "pa',
  ) =>
    new NoObjectGeneratedError({
      message,
      text,
      response: { id: "r1", timestamp: new Date(0), modelId: "judge" },
      usage: {
        inputTokens: 10,
        outputTokens: 512,
        totalTokens: 522,
        inputTokenDetails: { noCacheTokens: 10, cacheReadTokens: 0, cacheWriteTokens: 0 },
        outputTokenDetails: { textTokens: 512, reasoningTokens: 0 },
      },
      finishReason,
    });

  it("a truncated judge answer (finish 'length') → n/a 'judge truncated at N'", async () => {
    const ai = await import("ai");
    vi.mocked(ai.generateObject).mockRejectedValueOnce(noObject());

    const result = await judgeLLM({
      prompt: "p",
      finalText: "x",
      // biome-ignore lint/suspicious/noExplicitAny: model call is mocked
      model: {} as any,
      maxOutputTokens: 512,
    });

    expect(result.verdict).toBe("n/a");
    expect(result.explanation).toMatch(/^judge truncated at 512 output tokens/);
  });

  it("[guard] a schema mismatch (finish 'stop') is 'judge failed', not truncation", async () => {
    const ai = await import("ai");
    vi.mocked(ai.generateObject).mockRejectedValueOnce(
      noObject(
        "stop",
        "No object generated: response did not match schema.",
        '{"verdict":"PASS","explanation":"ok"}',
      ),
    );

    const result = await judgeLLM({
      prompt: "p",
      finalText: "x",
      // biome-ignore lint/suspicious/noExplicitAny: model call is mocked
      model: {} as any,
      maxOutputTokens: 512,
    });

    expect(result.verdict).toBe("n/a");
    expect(result.explanation).toMatch(/^judge failed/);
    expect(result.explanation).not.toMatch(/truncated/);
  });

  it("detects truncation through a RetryError too", async () => {
    const ai = await import("ai");
    vi.mocked(ai.generateObject).mockRejectedValueOnce(
      new RetryError({
        message: "Failed after 2 attempts.",
        reason: "errorNotRetryable",
        errors: [new Error("Overloaded"), noObject()],
      }),
    );

    // biome-ignore lint/suspicious/noExplicitAny: model call is mocked
    const result = await judgeLLM({ prompt: "p", finalText: "x", model: {} as any });

    expect(result.verdict).toBe("n/a");
    expect(result.explanation).toMatch(/^judge truncated at the provider default/);
  });
});
