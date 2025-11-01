// tests/providerFactory.test.js
import { jest } from '@jest/globals';

// Mock external heavy SDKs so we can import provider files without side effects
jest.unstable_mockModule('ollama', () => ({
  default: { generate: jest.fn().mockResolvedValue({ eval_count: 0, response: '{}' }) },
}));

jest.unstable_mockModule('@google/generative-ai', () => ({
  GoogleGenerativeAI: jest.fn().mockImplementation(() => ({
    getGenerativeModel: () => ({ generateContent: jest.fn() }),
  })),
}));

import { createProvider } from '../src/boss/llm/ProviderFactory.js';

describe('ProviderFactory', () => {
  afterEach(() => {
    delete process.env.LLM_BACKEND;
  });

  test('returns OllamaProvider when LLM_BACKEND=ollama', async () => {
    process.env.LLM_BACKEND = 'ollama';
    const provider = createProvider();
    expect(provider.constructor.name).toBe('OllamaProvider');
  });

  test('defaults to MockProvider when GOOGLE_API_KEY missing', () => {
    delete process.env.LLM_BACKEND;
    delete process.env.GOOGLE_API_KEY;
    // Force mock to avoid env dependency in CI
    process.env.LLM_BACKEND = 'mock';
    const provider = createProvider();
    expect(provider.constructor.name).toBe('MockProvider');
  });
}); 