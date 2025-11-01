// File: src/llm/ProviderFactory.js
import GeminiProvider from './providers/GeminiProvider.js';
import OllamaProvider from './providers/OllamaProvider.js';
import MockProvider from './providers/MockProvider.js';

/**
 * Create an LLM provider with custom configuration
 * @param {Object} config - Provider configuration
 * @param {string} config.backend - 'gemini', 'ollama', 'mock'
 * @param {string} config.model - Full model identifier (e.g., 'models/gemini-2.5-flash-lite')
 * @param {string} config.apiKey - Optional API key override
 * @param {number} config.temperature - Sampling temperature (0-1)
 * @param {number} config.maxTokens - Max output tokens
 * @returns {Provider} Configured provider instance
 */
export function createProvider(config = {}) {
  // Support both new config object style and legacy env var style
  const backend = config.backend || process.env.LLM_BACKEND || 'gemini';
  const model = config.model || process.env.LLM_MODEL || 'models/gemini-2.5-flash';
  const apiKey = config.apiKey || process.env.GOOGLE_API_KEY;

  const opts = {
    model,
    temperature: config.temperature ?? +process.env.LLM_TEMP ?? 0.7,
    maxTokens: config.maxTokens ?? +process.env.LLM_MAXTOKENS ?? 8192,
  };

  switch (backend) {
    case 'gemini':
      if (!apiKey) {
        throw new Error('GOOGLE_API_KEY not set (required for Gemini)');
      }
      return new GeminiProvider(apiKey, opts);

    case 'ollama':
      return new OllamaProvider(opts);

    case 'mock':
      return new MockProvider(opts);

    default:
      throw new Error(`Unknown LLM_BACKEND '${backend}'`);
  }
}

/**
 * Model presets for common use cases
 */
export const ModelPresets = {
  // Tactical tier (fast, frequent)
  TACTICAL_FASTEST: 'models/gemini-2.0-flash-lite',      // 30 RPM, 200 RPD
  TACTICAL_CAPACITY: 'models/gemini-2.5-flash-lite',     // 15 RPM, 1,000 RPD
  TACTICAL_BALANCED: 'models/gemini-2.5-flash',          // 10 RPM, 250 RPD
  TACTICAL_MEGA: 'models/gemma-3n',                      // 30 RPM, 14,400 RPD!

  // Strategic tier (smart, infrequent)
  STRATEGIC_BEST: 'models/gemini-2.5-pro',               // 5 RPM, 100 RPD
  STRATEGIC_FAST: 'models/gemini-2.0-flash',             // 15 RPM, 200 RPD

  // Legacy/experimental
  GEMINI_2_0_FLASH: 'models/gemini-2.0-flash',
  GEMINI_2_5_FLASH_PREVIEW: 'models/gemini-2.5-flash-preview',
};

/**
 * Get recommended model configuration for a use case
 */
export function getRecommendedConfig(useCase) {
  const configs = {
    tactical: {
      model: ModelPresets.TACTICAL_CAPACITY,  // gemini-2.5-flash-lite (1,000 RPD)
      temperature: 0.7,
      maxTokens: 1024
    },
    strategic: {
      model: ModelPresets.STRATEGIC_BEST,      // gemini-2.5-pro (100 RPD)
      temperature: 0.9,
      maxTokens: 4096
    },
    highVolume: {
      model: ModelPresets.TACTICAL_MEGA,       // gemma-3n (14,400 RPD!)
      temperature: 0.6,
      maxTokens: 512
    }
  };

  return configs[useCase] || configs.tactical;
}