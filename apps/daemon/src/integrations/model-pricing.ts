import type { ModelCost } from '@open-design/contracts';

/**
 * Bucket a provider-reported input price (USD per 1M tokens) into the coarse
 * cost tiers the model pickers render.
 *
 * Shared by every catalog adapter that receives a numeric price but has no
 * provider-authored tier (AMR/Vela, the OpenCode inference catalog). Keeping
 * one mapping means a model can never look "low" in one picker and "medium" in
 * another.
 */
export function modelCostFromInputPrice(
  inputPriceUsdPerMillion: number,
): ModelCost {
  if (inputPriceUsdPerMillion <= 0.5) return 'low';
  if (inputPriceUsdPerMillion <= 1) return 'medium';
  if (inputPriceUsdPerMillion <= 4) return 'high';
  return 'very_high';
}
