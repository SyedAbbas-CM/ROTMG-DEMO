// src/llm/providers/MockProvider.js
// Simple deterministic mock LLM provider for offline smoke tests.
// Returns canned responses and tracks call count.

import { BaseProvider } from './BaseProvider.js';

export default class MockProvider extends BaseProvider {
  constructor(opts = {}) {
    super(opts);
    this._count = 0;
  }

  async generate(_snapshot) {
    // Deterministic stub – issue no actions so the engine just idles.
    this._count += 1;
    return {
      json: { actions: [] },
      deltaMs: 0,
      tokens: 0,
      _mockId: this._count,
    };
  }
} 