import { fromNodeProviderChain } from "@aws-sdk/credential-providers";
import { AwsV4Signer } from "aws4fetch";
import type { PricingTable } from "./metering.js";
import type { ModelFactoryCatalogEntry } from "./model-factory.js";
import { parseModelIdentity } from "./model-identity.js";
import { findModelCatalogEntry, loadModelCatalog } from "./output-limits.js";
import { loadModelPricing } from "./pricing.js";

type Credentials = () => Promise<{
  accessKeyId: string;
  secretAccessKey: string;
  sessionToken?: string;
}>;

interface PreflightOptions {
  catalog?: readonly ModelFactoryCatalogEntry[];
  credentials?: Credentials;
  fetch?: typeof globalThis.fetch;
  requirePricing?: boolean;
  pricing?: PricingTable;
}

interface BedrockRoute {
  id: string;
  profile: string;
  region: string;
  endpoint: "bedrock-runtime" | "bedrock-mantle";
}

/** Read-only Runtime access and Mantle discovery gate, before any inference. */
export async function preflightBedrockModels(
  modelIds: readonly string[],
  options: PreflightOptions = {},
): Promise<void> {
  const catalog = options.catalog ?? loadModelCatalog();
  const routes = modelIds.map((id) => configuredRoute(id, catalog));
  if (options.requirePricing) {
    const prices = options.pricing ?? loadModelPricing();
    for (const route of routes) {
      const price = prices[route.id];
      if (
        !price ||
        !Number.isFinite(price.inputPer1M) ||
        price.inputPer1M <= 0 ||
        !Number.isFinite(price.outputPer1M) ||
        price.outputPer1M <= 0 ||
        !Number.isFinite(price.cachedInputPer1M) ||
        price.cachedInputPer1M < 0 ||
        !Number.isFinite(price.cacheCreationPer1M) ||
        price.cacheCreationPer1M < 0
      ) {
        throw new Error(
          `model ${route.id} needs a configured Bedrock price before funded inference`,
        );
      }
    }
  }
  const credentials = options.credentials ?? fromNodeProviderChain();
  const fetch = options.fetch ?? globalThis.fetch;
  for (const route of routes) {
    if (route.endpoint === "bedrock-mantle") {
      await checkMantleModel(route, credentials, fetch);
      continue;
    }
    if (isProfile(route.profile)) {
      const profile = await getJson(
        route,
        route.region,
        `/inference-profiles/${encodeURIComponent(route.profile)}`,
        credentials,
        fetch,
      );
      if (profile.inferenceProfileId !== route.profile || profile.status !== "ACTIVE") {
        throw new Error(`Bedrock preflight ${route.id}: profile ${route.profile} is not active`);
      }
      const models = profile.models;
      if (!Array.isArray(models) || models.length === 0) {
        throw new Error(`Bedrock preflight ${route.id}: profile has no destination models`);
      }
      const expectedBase = route.profile.slice(route.profile.indexOf(".") + 1);
      // Availability is checked in the source region only: other destinations
      // may be opt-in Regions the account never enabled (403), and AWS routes
      // there without requiring it.
      const regions = new Set<string>();
      for (const model of models) {
        const parsed = /^arn:[^:]+:bedrock:([a-z0-9-]*):[^:]*:foundation-model\/(.+)$/.exec(
          String((model as { modelArn?: unknown }).modelArn),
        );
        if (
          !parsed ||
          parsed[2] !== expectedBase ||
          (!parsed[1] && !route.profile.startsWith("global."))
        ) {
          throw new Error(`Bedrock preflight ${route.id}: profile targets an unexpected model`);
        }
        if (parsed[1]) regions.add(parsed[1]);
      }
      if (regions.size === 0) {
        throw new Error(`Bedrock preflight ${route.id}: profile has no regional model to check`);
      }
      if (!regions.has(route.region)) {
        throw new Error(
          `Bedrock preflight ${route.id}: profile does not serve source region ${route.region}`,
        );
      }
      await checkAvailability(route, route.region, expectedBase, credentials, fetch);
    } else {
      await checkAvailability(route, route.region, route.profile, credentials, fetch);
    }
  }
}

function configuredRoute(
  modelId: string,
  catalog: readonly ModelFactoryCatalogEntry[],
): BedrockRoute {
  const identity = parseModelIdentity(modelId);
  if (identity.kind !== "provider" || identity.provider !== "bedrock") {
    throw new Error(`Bedrock preflight requires a bedrock/ route: ${modelId}`);
  }
  const entry = findModelCatalogEntry(identity.canonicalId, catalog) as
    | ModelFactoryCatalogEntry
    | undefined;
  if (
    !entry?.bedrockProfile ||
    !entry.bedrockRegion ||
    !entry.bedrockEndpoint ||
    !entry.bedrockApi
  ) {
    throw new Error(
      `model ${identity.canonicalId} needs a configured Bedrock profile, region, endpoint and API`,
    );
  }
  if (entry.bedrockEndpoint === "bedrock-mantle" && entry.bedrockApi === "converse") {
    throw new Error(`model ${identity.canonicalId}: Bedrock Mantle does not support Converse`);
  }
  return {
    id: identity.canonicalId,
    profile: entry.bedrockProfile,
    region: entry.bedrockRegion,
    endpoint: entry.bedrockEndpoint,
  };
}

function isProfile(modelId: string): boolean {
  return /^(?:global|us|eu|au|jp|us-gov)\./.test(modelId);
}

async function checkMantleModel(
  route: BedrockRoute,
  credentials: Credentials,
  fetch: typeof globalThis.fetch,
): Promise<void> {
  const listing = await getJson(
    route,
    route.region,
    "/v1/models",
    credentials,
    fetch,
    "bedrock-mantle",
  );
  const models = listing.data;
  if (
    !Array.isArray(models) ||
    !models.some(
      (model) =>
        model &&
        typeof model === "object" &&
        model.id === route.profile &&
        model.object === "model" &&
        model.status === "available",
    )
  ) {
    throw new Error(`Bedrock preflight ${route.id}: model is not available in Mantle listing`);
  }
}

async function checkAvailability(
  route: BedrockRoute,
  region: string,
  modelId: string,
  credentials: Credentials,
  fetch: typeof globalThis.fetch,
): Promise<void> {
  const availability = await getJson(
    route,
    region,
    `/foundation-model-availability/${encodeURIComponent(modelId)}`,
    credentials,
    fetch,
  );
  const agreement = (availability.agreementAvailability as { status?: unknown } | undefined)
    ?.status;
  if (
    agreement !== "AVAILABLE" ||
    availability.authorizationStatus !== "AUTHORIZED" ||
    availability.entitlementAvailability !== "AVAILABLE" ||
    availability.regionAvailability !== "AVAILABLE"
  ) {
    throw new Error(
      `Bedrock preflight ${route.id} in ${region}: agreement=${agreement ?? "unknown"}, authorization=${availability.authorizationStatus ?? "unknown"}, entitlement=${availability.entitlementAvailability ?? "unknown"}, region=${availability.regionAvailability ?? "unknown"}`,
    );
  }
}

async function getJson(
  route: BedrockRoute,
  region: string,
  path: string,
  credentials: Credentials,
  fetch: typeof globalThis.fetch,
  service: "bedrock" | "bedrock-mantle" = "bedrock",
): Promise<Record<string, unknown>> {
  const url =
    service === "bedrock-mantle"
      ? `https://bedrock-mantle.${region}.api.aws${path}`
      : `https://bedrock.${region}.amazonaws.com${path}`;
  const signer = new AwsV4Signer({
    url,
    method: "GET",
    region,
    service,
    ...(await credentials()),
  });
  const signed = await signer.sign();
  const response = await fetch(url, { method: "GET", headers: signed.headers });
  if (!response.ok) {
    throw new Error(
      `Bedrock preflight ${route.id} in ${region}: control-plane HTTP ${response.status}`,
    );
  }
  return (await response.json()) as Record<string, unknown>;
}
