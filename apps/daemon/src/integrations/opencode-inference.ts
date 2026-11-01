import { createHash, randomUUID } from 'node:crypto';
import type { ModelCost } from '@open-design/contracts';
import type { ProviderModelOption } from '@open-design/contracts/api/providerModels';
import { modelCostFromInputPrice } from './model-pricing.js';

/**
 * OpenCode Console inference gateway quirks.
 *
 * `https://opencode.ai/inference/**` is not a plain OpenAI-compatible host:
 *
 *   1. Every request that reaches a model must carry a stable
 *      `x-opencode-session` header. Without it the gateway refuses the request
 *      with `400 MissingSessionID` ("Request is missing x-opencode-session and
 *      cannot be routed efficiently"), even when the API key is valid. See
 *      <https://opencode.ai/v2/docs/console/go/#where-can-i-use-it>. The
 *      session is per conversation, not per request: reusing it lets the
 *      gateway route and prompt-cache the turn.
 *   2. The client must identify itself with its own User-Agent instead of a
 *      generic HTTP-library name. Node's fetch sends `node`, which is exactly
 *      the generic shape the gateway asks clients to avoid, so daemon-side
 *      calls set an explicit OpenDesign UA. OpenCode CLI subprocesses already
 *      send their own UA and only need the session header.
 *   3. The model catalogue does NOT live under the chat base URL. Both
 *      `/inference/go/openai/v1/models` and `/inference/go/anthropic/v1/models`
 *      return 404; the catalogue is `/inference/go/v1/models` (Go plan) or
 *      `/inference/v1/models` (everything else).
 *
 * This module is the single source of truth for those three facts so the
 * model-discovery, connection-test, legacy proxy, memory-extraction and
 * finalize call sites cannot drift apart.
 */

const OPENCODE_INFERENCE_HOSTNAME = 'opencode.ai';
const OPENCODE_INFERENCE_PATH_PREFIX = '/inference/';
const OPENCODE_GO_PATH_SEGMENT = '/inference/go/';
const OPENCODE_CATALOG_PATH = '/inference/v1/models';
const OPENCODE_GO_CATALOG_PATH = '/inference/go/v1/models';

/**
 * Neutral, stable client identity. The gateway only asks that the value not be
 * a generic SDK/HTTP-library name; it does not require a semantic version.
 */
export const OPENCODE_INFERENCE_USER_AGENT = 'OpenDesign/1.0';

export interface OpenCodeInferenceBaseUrl {
  origin: string;
  /** True when the URL points at the Go-plan inference surface. */
  go: boolean;
}

export function parseOpenCodeInferenceBaseUrl(
  baseUrl: string | null | undefined,
): OpenCodeInferenceBaseUrl | null {
  const trimmed = typeof baseUrl === 'string' ? baseUrl.trim() : '';
  if (!trimmed) return null;
  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    return null;
  }
  if (parsed.hostname.toLowerCase() !== OPENCODE_INFERENCE_HOSTNAME) return null;
  if (!parsed.pathname.startsWith(OPENCODE_INFERENCE_PATH_PREFIX)) return null;
  return {
    origin: parsed.origin,
    go: parsed.pathname.includes(OPENCODE_GO_PATH_SEGMENT),
  };
}

export function isOpenCodeInferenceBaseUrl(
  baseUrl: string | null | undefined,
): boolean {
  return parseOpenCodeInferenceBaseUrl(baseUrl) !== null;
}

/**
 * The catalogue endpoint for a gateway base URL, or null when the URL is not
 * an OpenCode inference endpoint (callers then keep their protocol default).
 */
export function openCodeInferenceCatalogUrl(
  baseUrl: string | null | undefined,
): string | null {
  const parsed = parseOpenCodeInferenceBaseUrl(baseUrl);
  if (!parsed) return null;
  return `${parsed.origin}${parsed.go ? OPENCODE_GO_CATALOG_PATH : OPENCODE_CATALOG_PATH}`;
}

/**
 * Build the `x-opencode-session` value. A seed (conversation id, project id,
 * run id) makes it stable across turns, which is what the gateway needs for
 * routing and prompt caching; without a seed we still mint a well-formed id.
 */
export function createOpenCodeInferenceSessionId(
  seed?: string | null,
): string {
  const trimmed = typeof seed === 'string' ? seed.trim() : '';
  const source = trimmed || randomUUID();
  return `ses_${createHash('sha256').update(source).digest('hex').slice(0, 24)}`;
}

export interface OpenCodeInferenceHeaderOptions {
  /** Stable conversation/run seed. Generated per call when omitted. */
  seed?: string | null;
  /** Exact session id, when the caller already owns it. */
  sessionId?: string | null;
  /** Override the default OpenDesign UA (CLI subprocesses keep their own). */
  userAgent?: string | null;
}

export function openCodeInferenceHeaders(
  options: OpenCodeInferenceHeaderOptions = {},
): Record<string, string> {
  const sessionId =
    options.sessionId?.trim() ||
    createOpenCodeInferenceSessionId(options.seed ?? null);
  return {
    'x-opencode-session': sessionId,
    'user-agent': options.userAgent?.trim() || OPENCODE_INFERENCE_USER_AGENT,
  };
}

/**
 * Headers to merge into a request aimed at `baseUrl`. Returns an empty object
 * for every other host so call sites can spread it unconditionally.
 */
export function openCodeInferenceRequestHeaders(
  baseUrl: string | null | undefined,
  options: OpenCodeInferenceHeaderOptions = {},
): Record<string, string> {
  if (!isOpenCodeInferenceBaseUrl(baseUrl)) return {};
  return openCodeInferenceHeaders(options);
}

interface OpenCodeCatalogReasoning {
  supported?: unknown;
  toggle?: unknown;
  effort?: unknown;
}

interface OpenCodeCatalogCost {
  text?: {
    input?: { price?: unknown };
    output?: { price?: unknown };
  };
}

interface OpenCodeCatalogModel {
  id?: unknown;
  type?: unknown;
  name?: unknown;
  capabilities?: { reasoning?: OpenCodeCatalogReasoning };
  limit?: { context?: unknown };
  cost?: OpenCodeCatalogCost;
}

/**
 * Parse the OpenCode inference catalogue (`GET /inference/v1/models` and
 * `/inference/go/v1/models`).
 *
 * The payload is OpenAI-envelope shaped (`{ data: [...] }`) but each row
 * carries models.dev-v2 metadata: reasoning effort levels, context window and
 * per-million-token prices. Turning that into `reasoningOptions` /
 * `metadata` is what lets the BYOK pickers offer real effort choices instead
 * of a hardcoded list.
 */
export function parseOpenCodeInferenceCatalog(
  data: unknown,
): ProviderModelOption[] {
  const items = (data as { data?: unknown } | null | undefined)?.data;
  if (!Array.isArray(items)) return [];

  const models: ProviderModelOption[] = [];
  const seen = new Set<string>();
  for (const item of items) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) continue;
    const record = item as OpenCodeCatalogModel;
    const id = typeof record.id === 'string' ? record.id.trim() : '';
    if (!id || seen.has(id)) continue;
    // The catalogue also lists media/decision entries; this picker feeds chat
    // runs only. Rows without a type are accepted for forward compatibility.
    if (typeof record.type === 'string' && record.type !== 'chat') continue;
    seen.add(id);

    const label =
      typeof record.name === 'string' && record.name.trim()
        ? record.name.trim()
        : id;
    const reasoningOptions = reasoningOptionsFromCatalog(
      record.capabilities?.reasoning,
    );
    const contextWindowTokens = positiveInteger(record.limit?.context);
    const inputPriceUsdPerMillion = finitePrice(
      record.cost?.text?.input?.price,
    );
    const outputPriceUsdPerMillion = finitePrice(
      record.cost?.text?.output?.price,
    );
    const metadata = modelMetadata({
      inputPriceUsdPerMillion,
      contextWindowTokens,
    });

    models.push({
      id,
      label,
      ...(inputPriceUsdPerMillion === undefined
        ? {}
        : { inputPriceUsdPerMillion }),
      ...(outputPriceUsdPerMillion === undefined
        ? {}
        : { outputPriceUsdPerMillion }),
      ...(metadata ? { metadata } : {}),
      ...(reasoningOptions.length > 0 ? { reasoningOptions } : {}),
    });
  }
  return models;
}

function reasoningOptionsFromCatalog(
  reasoning: OpenCodeCatalogReasoning | undefined,
): ProviderModelOption[] {
  if (!reasoning || reasoning.supported !== true) return [];
  const efforts = Array.isArray(reasoning.effort)
    ? reasoning.effort
        .filter((effort): effort is string => typeof effort === 'string')
        .map((effort) => effort.trim())
        .filter(Boolean)
    : [];
  const ids = new Set(efforts);
  // `toggle: true` advertises an off switch even when the model does not list
  // `none` explicitly (e.g. MiniMax M3 / LongCat).
  if (reasoning.toggle === true) ids.add('none');
  if (ids.size === 0) return [];
  const ordered = Array.from(ids).sort(
    (a, b) => effortRank(a) - effortRank(b) || a.localeCompare(b),
  );
  return [
    { id: 'default', label: 'Default' },
    ...ordered.map((id) => ({ id, label: id })),
  ];
}

const EFFORT_ORDER = [
  'none',
  'minimal',
  'low',
  'medium',
  'high',
  'xhigh',
  'max',
] as const;

function effortRank(id: string): number {
  const index = EFFORT_ORDER.indexOf(id as (typeof EFFORT_ORDER)[number]);
  return index === -1 ? EFFORT_ORDER.length : index;
}

function modelMetadata(input: {
  inputPriceUsdPerMillion: number | undefined;
  contextWindowTokens: number | undefined;
}): ProviderModelOption['metadata'] | null {
  const cost: ModelCost | undefined =
    input.inputPriceUsdPerMillion === undefined
      ? undefined
      : modelCostFromInputPrice(input.inputPriceUsdPerMillion);
  const contextWindowTokens = input.contextWindowTokens;
  if (!cost && contextWindowTokens === undefined) return null;
  return {
    ...(cost ? { cost } : {}),
    ...(contextWindowTokens === undefined ? {} : { contextWindowTokens }),
  };
}

function positiveInteger(value: unknown): number | undefined {
  return typeof value === 'number' &&
    Number.isSafeInteger(value) &&
    value > 0 &&
    value <= 1_000_000_000
    ? value
    : undefined;
}

function finitePrice(value: unknown): number | undefined {
  const price =
    typeof value === 'number'
      ? value
      : typeof value === 'string' && value.trim()
        ? Number(value.trim())
        : NaN;
  return Number.isFinite(price) && price >= 0 ? price : undefined;
}
