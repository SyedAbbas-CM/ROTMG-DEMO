// src/llm/providers/OllamaProvider.js
import ollama from 'ollama';
import { BaseProvider } from './BaseProvider.js';
import { trace } from '@opentelemetry/api';
import { registry } from '../../../registry/index.js';
const tracer = trace.getTracer('llm');

let abilitiesDoc = Object.keys(registry.validators).join(', ');
registry.events.on('update', () => {
  abilitiesDoc = Object.keys(registry.validators).join(', ');
});

export default class OllamaProvider extends BaseProvider {
  constructor(opts = {}) {
    super(opts);
  }
  async generate(snapshot) {
    return tracer.startActiveSpan('llm.generate.ollama', async span => {
      // abilitiesDoc kept up to date via registry events
      const systemMsg = `You can control the boss via JSON {actions:[{ability,args}]} . Allowed abilities: ${abilitiesDoc}.`;
      const prompt = 'You are the tactical brain of the boss.\nWORLD_STATE:\n' + JSON.stringify(snapshot);
      const t0 = Date.now();
      const res = await Promise.race([
        ollama.generate({
          model: this.opts.model || 'phi3',
          prompt: `${systemMsg}\n\n${prompt}`,
          stream: false,
        }),
        new Promise((_ , rej)=> setTimeout(()=>rej(new Error('Ollama timeout')), 15_000))
      ]);
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
