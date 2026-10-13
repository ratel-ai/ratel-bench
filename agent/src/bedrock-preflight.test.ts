import { describe, expect, it } from "vitest";
import { preflightBedrockModels } from "./bedrock-preflight.js";

const credentials = async () => ({ accessKeyId: "AKIDEXAMPLE", secretAccessKey: "secret" });

describe("Bedrock campaign preflight", () => {
  it("accepts a global profile's regionless ARN and checks its regional model", async () => {
    const urls: string[] = [];
    const fetch = async (input: RequestInfo | URL) => {
      const url = String(input);
      urls.push(url);
      if (url.includes("/inference-profiles/")) {
        return Response.json({
          inferenceProfileId: "global.anthropic.claude-haiku-4-5-20251001-v1:0",
          status: "ACTIVE",
          models: [
            {
              modelArn:
                "arn:aws:bedrock:::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0",
            },
            {
              modelArn:
                "arn:aws:bedrock:eu-central-1::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0",
            },
          ],
        });
      }
      return Response.json({
        agreementAvailability: { status: "AVAILABLE" },
        authorizationStatus: "AUTHORIZED",
        entitlementAvailability: "AVAILABLE",
        regionAvailability: "AVAILABLE",
      });
    };

    await preflightBedrockModels(["bedrock/anthropic.claude-haiku-4-5"], {
      credentials,
      fetch,
    });

    expect(urls).toEqual([
      "https://bedrock.eu-central-1.amazonaws.com/inference-profiles/global.anthropic.claude-haiku-4-5-20251001-v1%3A0",
      "https://bedrock.eu-central-1.amazonaws.com/foundation-model-availability/anthropic.claude-haiku-4-5-20251001-v1%3A0",
    ]);
  });

  it("rejects a global profile with no regional model to check", async () => {
    const urls: string[] = [];
    await expect(
      preflightBedrockModels(["bedrock/anthropic.claude-haiku-4-5"], {
        credentials,
        fetch: async (input) => {
          urls.push(String(input));
          return Response.json({
            inferenceProfileId: "global.anthropic.claude-haiku-4-5-20251001-v1:0",
            status: "ACTIVE",
            models: [
              {
                modelArn:
                  "arn:aws:bedrock:::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0",
              },
            ],
          });
        },
      }),
    ).rejects.toThrow(/no regional model/i);
    expect(urls).toHaveLength(1);
  });

  it("rejects a regionless ARN in a regional profile", async () => {
    const urls: string[] = [];
    await expect(
      preflightBedrockModels(["bedrock/meta.llama4-scout-17b-instruct-v1:0"], {
        credentials,
        fetch: async (input) => {
          urls.push(String(input));
          return Response.json({
            inferenceProfileId: "us.meta.llama4-scout-17b-instruct-v1:0",
            status: "ACTIVE",
            models: [
              {
                modelArn: "arn:aws:bedrock:::foundation-model/meta.llama4-scout-17b-instruct-v1:0",
              },
            ],
          });
        },
      }),
    ).rejects.toThrow(/unexpected model/i);
    expect(urls).toHaveLength(1);
  });

  it("checks Mantle model discovery with the Mantle signing service", async () => {
    const urls: string[] = [];
    await preflightBedrockModels(["bedrock/google.gemma-4-31b"], {
      credentials,
      fetch: async (input, init) => {
        urls.push(String(input));
        expect(init?.method).toBe("GET");
        expect(new Headers(init?.headers).get("authorization")).toContain(
          "/bedrock-mantle/aws4_request",
        );
        return Response.json({
          data: [{ id: "google.gemma-4-31b", object: "model", status: "available" }],
        });
      },
    });
    expect(urls).toEqual(["https://bedrock-mantle.eu-central-1.api.aws/v1/models"]);
  });

  it("checks source profiles and destination entitlements without inference", async () => {
    const urls: string[] = [];
    const fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      urls.push(url);
      expect(init?.method).toBe("GET");
      expect(new Headers(init?.headers).get("authorization")).toContain("/bedrock/aws4_request");
      if (url.includes("/inference-profiles/")) {
        return Response.json({
          inferenceProfileId: "global.openai.gpt-6-sol",
          status: "ACTIVE",
          models: [{ modelArn: "arn:aws:bedrock:us-east-1::foundation-model/openai.gpt-6-sol" }],
        });
      }
      return Response.json({
        agreementAvailability: { status: "AVAILABLE" },
        authorizationStatus: "AUTHORIZED",
        entitlementAvailability: "AVAILABLE",
        regionAvailability: "AVAILABLE",
      });
    };
    await preflightBedrockModels(["bedrock/openai.gpt-6-sol", "bedrock/qwen.qwen3-coder-next"], {
      credentials,
      fetch,
      requirePricing: true,
      pricing: Object.fromEntries(
        ["bedrock/openai.gpt-6-sol", "bedrock/qwen.qwen3-coder-next"].map((id) => [
          id,
          { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 0, cacheCreationPer1M: 0 },
        ]),
      ),
    });
    expect(urls).toEqual([
      "https://bedrock.eu-central-1.amazonaws.com/inference-profiles/global.openai.gpt-6-sol",
      "https://bedrock.us-east-1.amazonaws.com/foundation-model-availability/openai.gpt-6-sol",
      "https://bedrock.us-east-1.amazonaws.com/foundation-model-availability/qwen.qwen3-coder-next",
    ]);
  });

  it("fails closed on missing entitlement and never tries another provider", async () => {
    const urls: string[] = [];
    await expect(
      preflightBedrockModels(["bedrock/qwen.qwen3-coder-next"], {
        credentials,
        fetch: async (input) => {
          urls.push(String(input));
          return Response.json({
            agreementAvailability: { status: "AVAILABLE" },
            authorizationStatus: "AUTHORIZED",
            entitlementAvailability: "NOT_AVAILABLE",
            regionAvailability: "AVAILABLE",
          });
        },
      }),
    ).rejects.toThrow(/qwen.qwen3-coder-next.*entitlement/i);
    expect(urls).toHaveLength(1);
    expect(urls[0]).toContain("bedrock.us-east-1.amazonaws.com");
  });

  it("rejects incomplete routing before making control-plane requests", async () => {
    let calls = 0;
    await expect(
      preflightBedrockModels(["bedrock/unknown"], {
        credentials,
        fetch: async () => {
          calls++;
          return Response.json({});
        },
      }),
    ).rejects.toThrow(/configured Bedrock/);
    expect(calls).toBe(0);
  });

  it("rejects a pending model agreement before inference", async () => {
    await expect(
      preflightBedrockModels(["bedrock/qwen.qwen3-coder-next"], {
        credentials,
        fetch: async () =>
          Response.json({
            agreementAvailability: { status: "PENDING" },
            authorizationStatus: "AUTHORIZED",
            entitlementAvailability: "AVAILABLE",
            regionAvailability: "AVAILABLE",
          }),
      }),
    ).rejects.toThrow(/agreement=PENDING/);
  });

  it("rejects an unpriced funded route before any request", async () => {
    let calls = 0;
    await expect(
      preflightBedrockModels(["bedrock/qwen.qwen3-coder-next"], {
        credentials,
        requirePricing: true,
        fetch: async () => {
          calls++;
          return Response.json({});
        },
      }),
    ).rejects.toThrow(/Bedrock price/i);
    expect(calls).toBe(0);
  });
});
