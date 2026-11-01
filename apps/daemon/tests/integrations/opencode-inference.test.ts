import { promises as dnsPromises } from 'node:dns';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { listProviderModels } from '../../src/integrations/provider-models.js';
import {
  createOpenCodeInferenceSessionId,
  openCodeInferenceCatalogUrl,
  openCodeInferenceRequestHeaders,
  parseOpenCodeInferenceBaseUrl,
  parseOpenCodeInferenceCatalog,
} from '../../src/integrations/opencode-inference.js';

describe('OpenCode inference gateway quirks', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it.each([
    ['https://opencode.ai/inference/go/openai/v1', true],
    ['https://opencode.ai/inference/go/anthropic/v1', true],
    ['https://opencode.ai/inference/openai/v1', false],
    ['https://opencode.ai/inference/v1', false],
  ])('detects %s as an inference base URL (go=%s)', (baseUrl, go) => {
    expect(parseOpenCodeInferenceBaseUrl(baseUrl)).toMatchObject({ go });
  });

  it.each([
    'https://api.openai.com/v1',
    'https://opencode.ai/docs',
    'https://example.com/inference/go/openai/v1',
    'not a url',
  ])('ignores %s', (baseUrl) => {
    expect(parseOpenCodeInferenceBaseUrl(baseUrl)).toBeNull();
    expect(openCodeInferenceRequestHeaders(baseUrl)).toEqual({});
  });

  it('points the catalogue at the shared inference path, not the chat prefix', () => {
    // The chat base URL has no /models route; both of these 404 upstream.
    expect(
      openCodeInferenceCatalogUrl('https://opencode.ai/inference/go/openai/v1'),
    ).toBe('https://opencode.ai/inference/go/v1/models');
    expect(
      openCodeInferenceCatalogUrl('https://opencode.ai/inference/openai/v1'),
    ).toBe('https://opencode.ai/inference/v1/models');
  });

  it('reuses one session id per conversation and mints well-formed ones', () => {
    const first = createOpenCodeInferenceSessionId('conversation-42');
    const second = createOpenCodeInferenceSessionId('conversation-42');
    expect(first).toBe(second);
    expect(first).toMatch(/^ses_[0-9a-f]{24}$/);
    expect(createOpenCodeInferenceSessionId('other')).not.toBe(first);
    expect(createOpenCodeInferenceSessionId()).toMatch(/^ses_[0-9a-f]{24}$/);
  });

  it('sends the session header and its own user agent', () => {
    expect(
      openCodeInferenceRequestHeaders(
        'https://opencode.ai/inference/go/openai/v1',
        { seed: 'conversation-42' },
      ),
    ).toEqual({
      'x-opencode-session': createOpenCodeInferenceSessionId('conversation-42'),
      'user-agent': 'OpenDesign/1.0',
    });
  });

  it('parses reasoning effort, limits and prices from the go catalogue', () => {
    const models = parseOpenCodeInferenceCatalog({
      object: 'list',
      data: [
        {
          id: 'deepseek-v4.1-flash',
          type: 'chat',
          name: 'DeepSeek V4.1 Flash',
          capabilities: {
            reasoning: { supported: true, effort: ['low', 'high', 'max'] },
          },
          limit: { context: 1_000_000, output: 384_000 },
          cost: { text: { input: { price: 0.15 }, output: { price: 0.6 } } },
        },
        {
          id: 'minimax-m3',
          type: 'chat',
          name: 'MiniMax-M3',
          capabilities: { reasoning: { supported: true, toggle: true } },
        },
        {
          id: 'gpt-image-2',
          type: 'image',
          name: 'Image',
        },
        {
          id: 'no-reasoning',
          type: 'chat',
          name: 'No reasoning',
        },
      ],
    });

    expect(models.map((model) => model.id)).toEqual([
      'deepseek-v4.1-flash',
      'minimax-m3',
      'no-reasoning',
    ]);
    expect(models[0]).toMatchObject({
      label: 'DeepSeek V4.1 Flash',
      inputPriceUsdPerMillion: 0.15,
      outputPriceUsdPerMillion: 0.6,
      metadata: { cost: 'low', contextWindowTokens: 1_000_000 },
      reasoningOptions: [
        { id: 'default', label: 'Default' },
        { id: 'low', label: 'low' },
        { id: 'high', label: 'high' },
        { id: 'max', label: 'max' },
      ],
    });
    // `toggle: true` advertises the off switch even without an explicit `none`.
    expect(models[1]?.reasoningOptions?.map((option: { id: string }) => option.id)).toEqual([
      'default',
      'none',
    ]);
    expect(models[2]?.reasoningOptions).toBeUndefined();
  });

  it('discovers models from the go catalogue with gateway headers', async () => {
    vi.spyOn(dnsPromises, 'lookup').mockResolvedValue([
      { address: '104.18.0.1', family: 4 },
    ] as never);
    const fetchMock = vi.fn(async () =>
      new Response(
        JSON.stringify({
          data: [
            {
              id: 'deepseek-v4.1-flash',
              type: 'chat',
              name: 'DeepSeek V4.1 Flash',
              capabilities: {
                reasoning: { supported: true, effort: ['low', 'high'] },
              },
            },
          ],
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const result = await listProviderModels({
      protocol: 'openai',
      baseUrl: 'https://opencode.ai/inference/go/openai/v1',
      apiKey: 'oc_sk_test',
    });

    expect(result).toMatchObject({ ok: true, kind: 'success' });
    expect(result.models?.[0]).toMatchObject({
      id: 'deepseek-v4.1-flash',
      reasoningOptions: expect.arrayContaining([
        expect.objectContaining({ id: 'high' }),
      ]),
    });
    expect(fetchMock).toHaveBeenCalledWith(
      'https://opencode.ai/inference/go/v1/models',
      expect.objectContaining({
        headers: expect.objectContaining({
          authorization: 'Bearer oc_sk_test',
          'x-opencode-session': expect.stringMatching(/^ses_/),
          'user-agent': 'OpenDesign/1.0',
        }),
      }),
    );
  });
});
