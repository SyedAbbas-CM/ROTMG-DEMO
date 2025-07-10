// File: src/llm/ProviderFactory.js
import GeminiProvider from './providers/GeminiProvider.js';
import OllamaProvider from './providers/OllamaProvider.js';

export function createProvider() {
  const backend = process.env.LLM_BACKEND || 'gemini';
  const model   = process.env.LLM_MODEL   || 'gemini-1.5-pro-latest';
  const opts = {
    model,
    temperature: +process.env.LLM_TEMP      || 0.7,
    maxTokens  : +process.env.LLM_MAXTOKENS || 256,
  };

  switch (backend) {
    case 'gemini':
      if (!process.env.GOOGLE_API_KEY) {
        throw new Error('GOOGLE_API_KEY not set');
      }
      return new GeminiProvider(process.env.GOOGLE_API_KEY, opts);

    case 'ollama':
      return new OllamaProvider(opts);

    default:
      throw new Error(`Unknown LLM_BACKEND '${backend}'`);
  }
}
