import { createAmazonBedrock } from "@ai-sdk/amazon-bedrock";
import { createBedrockMantle } from "@ai-sdk/amazon-bedrock/mantle";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createOpenAI } from "@ai-sdk/openai";
import { fromNodeProviderChain } from "@aws-sdk/credential-providers";
import type { LanguageModel } from "ai";
import { AwsV4Signer } from "aws4fetch";
import { parseCustomEndpoint } from "./model-endpoint.js";
import { parseModelIdentity } from "./model-identity.js";
import {
  findModelCatalogEntry,
  loadModelCatalog,
  type ModelCatalogEntry,
} from "./output-limits.js";
import type { ResolvedModel } from "./types.js";

export interface ModelFactoryCatalogEntry extends ModelCatalogEntry {
  bedrockRegion?: string;
  bedrockApi?: "converse" | "responses" | "chat";
  bedrockEndpoint?: "bedrock-runtime" | "bedrock-mantle";
}

export interface ResolveModelOptions {
  catalog?: readonly ModelFactoryCatalogEntry[];
  env?: Record<string, string | undefined>;
  fetch?: typeof globalThis.fetch;
  ollamaBaseURL?: string;
  modelApiKey?: string;
  awsCredentials?: () => Promise<{
    accessKeyId: string;
    secretAccessKey: string;
    sessionToken?: string;
  }>;
}

type ProviderResolver = (
  id: string,
  model: string,
  opts: ResolveModelOptions,
  env: Record<string, string | undefined>,
) => ResolvedModel;

const providerResolvers: Partial<Record<string, ProviderResolver>> = {
  bedrock: resolveBedrock,
  anthropic: resolveAnthropic,
  openai: resolveOpenAI,
};

/** Resolve one serving identity to one native AI SDK model. */
export function resolveModel(modelId: string, opts: ResolveModelOptions = {}): ResolvedModel {
  const identity = parseModelIdentity(modelId);
  const endpoint = parseCustomEndpoint(modelId);
  if (endpoint) {
    return {
      id: identity.canonicalId,
      model: createOpenAI({
        baseURL: endpoint.baseURL,
        apiKey: opts.modelApiKey ?? "none",
        fetch: opts.fetch,
      }).chat(endpoint.modelName),
    };
  }
  if (identity.kind === "ollama") {
    return {
      id: identity.canonicalId,
      model: createOpenAI({
        baseURL: opts.ollamaBaseURL ?? "http://localhost:11434/v1",
        apiKey: "ollama",
        fetch: opts.fetch,
      }).chat(identity.model),
    };
  }
  if (identity.kind !== "provider")
    throw new Error(`unsupported model route: ${identity.canonicalId}`);
  const env = opts.env ?? process.env;
  const resolver = providerResolvers[identity.provider];
  if (!resolver)
    throw new Error(`model ${identity.canonicalId} needs a ${identity.provider} adapter`);
  return resolver(identity.canonicalId, identity.model, opts, env);
}

function resolveBedrock(
  id: string,
  _model: string,
  opts: ResolveModelOptions,
  env: Record<string, string | undefined>,
): ResolvedModel {
  const entry = findModelCatalogEntry(id, opts.catalog ?? loadModelCatalog()) as
    | ModelFactoryCatalogEntry
    | undefined;
  if (!entry?.bedrockProfile) {
    throw new Error(`model ${id} needs a configured Bedrock profile`);
  }
  const region = entry.bedrockRegion ?? env.AWS_REGION ?? "eu-central-1";
  const api = entry.bedrockApi ?? "converse";
  const endpoint = entry.bedrockEndpoint ?? "bedrock-runtime";
  if (!["converse", "responses", "chat"].includes(api)) {
    throw new Error(`model ${id}: unsupported Bedrock API ${api}`);
  }
  if (!["bedrock-runtime", "bedrock-mantle"].includes(endpoint)) {
    throw new Error(`model ${id}: unsupported Bedrock endpoint ${endpoint}`);
  }
  if (/(?:^|\.)openai\.gpt-6/.test(entry.bedrockProfile) && api !== "responses") {
    throw new Error(`model ${id} requires the Bedrock Responses API for reasoning and tool calls`);
  }
  const apiKey = env.AWS_BEARER_TOKEN_BEDROCK;
  const credentials = opts.awsCredentials ?? fromNodeProviderChain();
  if (endpoint === "bedrock-mantle") {
    if (api === "converse")
      throw new Error(`model ${id}: Bedrock Mantle does not support Converse`);
    const mantle = createBedrockMantle({
      region,
      baseURL: `https://bedrock-mantle.${region}.api.aws/${/^(?:google\.gemma-4|openai\.gpt-(?!oss-)|xai\.)/.test(entry.bedrockProfile) ? "openai/v1" : "v1"}`,
      ...(apiKey ? { apiKey } : { credentialProvider: credentials }),
      fetch: opts.fetch,
    });
    const model =
      api === "responses"
        ? mantle.responses(entry.bedrockProfile)
        : mantle.chat(entry.bedrockProfile);
    return {
      id,
      model:
        api === "responses" && /(?:^|\.)openai\.gpt-6/.test(entry.bedrockProfile)
          ? withModelIdentityAndReasoning(
              model,
              api,
              entry.bedrockProfile,
              true,
              "bedrock-mantle.responses",
            )
          : model,
    };
  }
  if (api === "responses" || api === "chat") {
    const baseURL = `https://bedrock-runtime.${region}.amazonaws.com/openai/v1`;
    const sdkModelId =
      api === "responses" && /(?:^|\.)openai\.gpt-6/.test(entry.bedrockProfile)
        ? entry.bedrockProfile.slice(entry.bedrockProfile.indexOf("openai.") + "openai.".length)
        : entry.bedrockProfile;
    const openai = createOpenAI({
      baseURL,
      apiKey: apiKey ?? "sigv4",
      fetch: apiKey
        ? profileFetch(entry.bedrockProfile, opts.fetch)
        : sigv4Fetch(region, credentials, entry.bedrockProfile, opts.fetch),
    });
    const model = api === "responses" ? openai.responses(sdkModelId) : openai.chat(sdkModelId);
    return {
      id,
      model: withModelIdentityAndReasoning(
        model,
        api,
        entry.bedrockProfile,
        sdkModelId.startsWith("gpt-6"),
      ),
    };
  }
  const provider = createAmazonBedrock({
    region,
    ...(apiKey ? { apiKey } : { credentialProvider: credentials }),
    fetch: opts.fetch,
  });
  return { id, model: provider(entry.bedrockProfile) };
}

function resolveAnthropic(
  id: string,
  model: string,
  opts: ResolveModelOptions,
  env: Record<string, string | undefined>,
): ResolvedModel {
  const apiKey = env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error(`model ${id} requires ANTHROPIC_API_KEY`);
  return { id, model: createAnthropic({ apiKey, fetch: opts.fetch })(model) };
}

function resolveOpenAI(
  id: string,
  model: string,
  opts: ResolveModelOptions,
  env: Record<string, string | undefined>,
): ResolvedModel {
  const apiKey = env.OPENAI_API_KEY;
  if (!apiKey) throw new Error(`model ${id} requires OPENAI_API_KEY`);
  const direct = createOpenAI({ apiKey, fetch: opts.fetch }).responses(model);
  return {
    id,
    model: model.startsWith("gpt-6")
      ? withModelIdentityAndReasoning(direct, "responses", model, true, "openai.responses")
      : direct,
  };
}

function withModelIdentityAndReasoning(
  model: LanguageModel,
  api: "responses" | "chat",
  modelId: string,
  forceReasoning = false,
  providerId = `amazon-bedrock.${api}`,
): LanguageModel {
  if (typeof model === "string") throw new Error("expected a resolved Bedrock model");
  return new Proxy(model, {
    get(target, property, receiver) {
      if (property === "provider") return providerId;
      if (property === "modelId") return modelId;
      if (forceReasoning && (property === "doGenerate" || property === "doStream")) {
        return (args: { providerOptions?: Record<string, Record<string, unknown>> }) => {
          const openai = args.providerOptions?.openai ?? {};
          const options = {
            ...args,
            providerOptions: {
              ...args.providerOptions,
              openai: { ...openai, forceReasoning: true, store: false },
            },
          };
          return Reflect.apply(Reflect.get(target, property), target, [options]);
        };
      }
      return Reflect.get(target, property, receiver);
    },
  });
}

function sigv4Fetch(
  region: string,
  credentials: () => Promise<{
    accessKeyId: string;
    secretAccessKey: string;
    sessionToken?: string;
  }>,
  profile: string,
  transport: typeof globalThis.fetch = globalThis.fetch,
): typeof globalThis.fetch {
  return async (input, init) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const headers = new Headers(init?.headers);
    headers.delete("authorization");
    const body = profileBody(init?.body, profile);
    const signer = new AwsV4Signer({
      url,
      method: init?.method ?? "POST",
      headers: Object.fromEntries(headers),
      body,
      region,
      service: "bedrock",
      ...(await credentials()),
    });
    const signed = await signer.sign();
    return transport(input, { ...init, headers: signed.headers, body });
  };
}

function profileFetch(
  profile: string,
  transport: typeof globalThis.fetch = globalThis.fetch,
): typeof globalThis.fetch {
  return (input, init) => transport(input, { ...init, body: profileBody(init?.body, profile) });
}

function profileBody(body: BodyInit | null | undefined, profile: string): string | undefined {
  if (typeof body !== "string") return undefined;
  return JSON.stringify({ ...JSON.parse(body), model: profile });
}
