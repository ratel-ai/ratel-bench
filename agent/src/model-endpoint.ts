// Shared helpers for user-hosted, OpenAI-compatible model endpoints addressed as
// `<baseURL>#<model-name>` (e.g. a vLLM/TGI server, or an AWS API-Gateway-fronted
// model). Kept CLI- and runner-agnostic so both the BFCL campaign (`cli.ts`) and
// the SR-Agents selection campaign (`sragents-select.ts`) can share the endpoint
// parse + warm-up without importing each other.

/** Parsed custom endpoint: the OpenAI-compatible base URL and the served model name. */
export interface CustomEndpoint {
  baseURL: string;
  modelName: string;
}

/**
 * Parse a `<baseURL>#<model-name>` model string. Returns `null` for non-URL model
 * ids (so callers fall through to their provider prefixes). Throws a clear error
 * when the `#` separator or the model name is missing — self-hosted servers can't
 * be queried without an explicit model id.
 */
export function parseCustomEndpoint(modelId: string): CustomEndpoint | null {
  if (!modelId.startsWith("http://") && !modelId.startsWith("https://")) return null;
  const hashIdx = modelId.indexOf("#");
  if (hashIdx === -1) {
    throw new Error(
      `custom model URL must be of the form <baseURL>#<model-name>, ` +
        `e.g. https://my-host:8000/v1#llama-3.1-70b (got "${modelId}")`,
    );
  }
  const baseURL = modelId.slice(0, hashIdx);
  const modelName = modelId.slice(hashIdx + 1);
  if (!modelName) {
    throw new Error(`custom model URL "${modelId}" is missing a model name after "#"`);
  }
  return { baseURL, modelName };
}

/** Resolve the `POST /warm` URL from a `/v1`-style base URL (trailing slash tolerant). */
function warmUrl(baseURL: string): string {
  return `${baseURL.replace(/\/+$/, "")}/warm`;
}

/** Overall budget (ms) to wait for a cold model to become ready before giving up. */
const WARM_DEADLINE_MS = 180_000;
/** Fallback poll interval (s) when the gateway doesn't send `retry_after_seconds`. */
const WARM_DEFAULT_RETRY_S = 15;
/** Cap a server-advertised `retry_after_seconds` so one huge value can't stall the run. */
const WARM_MAX_RETRY_S = 30;

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

/**
 * Warm up each user-hosted endpoint in `rawModelIds` before the campaign. Some
 * gateways (scale-to-zero Lambda/API Gateway) cold-start the model asynchronously:
 * `POST <baseURL>/warm {model}` returns `{state:"warming", retry_after_seconds}`
 * and the chat endpoint 503s until it flips to `{state:"ready"}`. We poll `/warm`
 * until ready so early cells don't all fail on the cold start.
 *
 * Lenient by design: a non-2xx response or network error is logged as a warning
 * and skipped, so generic OpenAI-compatible servers with no `/warm` route (vLLM,
 * TGI, LM Studio) still run normally. Endpoints are deduped by `baseURL|model`.
 */
export async function warmUpModels(rawModelIds: string[], apiKey?: string): Promise<void> {
  const seen = new Set<string>();
  for (const id of rawModelIds) {
    const ep = parseCustomEndpoint(id);
    if (!ep) continue;
    const key = `${ep.baseURL}|${ep.modelName}`;
    if (seen.has(key)) continue;
    seen.add(key);
    await warmOne(ep, apiKey);
  }
}

/** POST `/warm` once. Returns the parsed body, or `null` on non-2xx / non-JSON / error. */
async function pokeWarm(
  url: string,
  ep: CustomEndpoint,
  apiKey?: string,
): Promise<Record<string, unknown> | null> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60_000);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
      },
      body: JSON.stringify({ model: ep.modelName }),
      signal: controller.signal,
    });
    if (!res.ok) {
      console.warn(
        `warn: warm-up POST ${url} returned ${res.status} — continuing ` +
          `(the endpoint may not support /warm; first calls could be slow).`,
      );
      return null;
    }
    return (await res.json().catch(() => ({}))) as Record<string, unknown>;
  } catch (err) {
    console.warn(
      `warn: warm-up POST ${url} failed (${(err as Error).message}) — continuing; ` +
        `first calls to ${ep.modelName} could time out on cold start.`,
    );
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

async function warmOne(ep: CustomEndpoint, apiKey?: string): Promise<void> {
  const url = warmUrl(ep.baseURL);
  const deadline = Date.now() + WARM_DEADLINE_MS;
  let announced = false;
  for (;;) {
    const body = await pokeWarm(url, ep, apiKey);
    // Non-2xx / error / no state field → nothing more we can do; pokeWarm already
    // warned. A 200 with no `state` (generic server) counts as warmed.
    if (body === null) return;
    const state = typeof body.state === "string" ? body.state : undefined;
    if (state !== "warming") {
      console.log(`warmed ${ep.modelName} at ${ep.baseURL}`);
      return;
    }
    if (Date.now() >= deadline) {
      console.warn(
        `warn: ${ep.modelName} still warming after ${WARM_DEADLINE_MS / 1000}s — ` +
          `continuing; early cells may fail until it loads.`,
      );
      return;
    }
    if (!announced) {
      console.log(`warming ${ep.modelName} at ${ep.baseURL} (cold start, polling)…`);
      announced = true;
    }
    const retryS =
      typeof body.retry_after_seconds === "number" ? body.retry_after_seconds : undefined;
    const waitS = Math.min(retryS ?? WARM_DEFAULT_RETRY_S, WARM_MAX_RETRY_S);
    await sleep(waitS * 1000);
  }
}
