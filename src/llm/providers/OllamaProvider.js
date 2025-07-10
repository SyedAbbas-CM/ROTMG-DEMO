// src/llm/providers/OllamaProvider.js
import ollama from 'ollama';
import { BaseProvider } from './BaseProvider.js';
import { trace } from '@opentelemetry/api';
const tracer = trace.getTracer('llm');

export default class OllamaProvider extends BaseProvider {
  constructor(opts = {}) {
    super(opts);
  }
  async generate(snapshot) {
    return tracer.startActiveSpan('llm.generate.ollama', async span => {
      const prompt =
        'You are the tactical brain of the boss.\nWORLD_STATE:\n' +
        JSON.stringify(snapshot);
      const t0 = Date.now();
      const res = await ollama.generate({
        model: this.opts.model || 'phi3',
        prompt,
        stream: false,
      });
      const delta = Date.now() - t0;
      span.setAttribute('latency_ms', delta);
      span.setAttribute('tokens', res.eval_count);
      span.end();
      return {
        json: JSON.parse(res.response),
        deltaMs: delta,
        tokens: res.eval_count,
      };
    });
  }
}
