// File: src/llm/providers/GeminiProvider.js
import { GoogleGenerativeAI } from '@google/generative-ai';
import { BaseProvider }       from './BaseProvider.js';
import { issueActionsFn }     from '../planFunction.js';
import { trace }              from '@opentelemetry/api';
import { registry }          from '../../../registry/index.js';

const tracer = trace.getTracer('llm');

function toSnake(id){
  const [, name] = id.split(':');
  return name.split('@')[0]
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .toLowerCase();
}

let abilitiesDoc = Object.keys(registry.validators).map(toSnake).join(', ');
registry.events.on('update', () => {
  abilitiesDoc = Object.keys(registry.validators).map(toSnake).join(', ');
});

export default class GeminiProvider extends BaseProvider {
  constructor(apiKey, opts) {
    super(opts);
    const {
      model: modelId = 'models/gemini-2.5-flash',
      temperature = 0.7,
      maxTokens   = 256,
      ...restConfig
    } = opts;

    const generationConfig = {
      temperature,
      maxOutputTokens: maxTokens,
      ...restConfig,
    };

    this.ai    = new GoogleGenerativeAI(apiKey);
    this.model = this.ai.getGenerativeModel(
      { model: modelId, generationConfig },
      { apiVersion: 'v1beta' }
    );
  }

  async generate(snapshot) {
    return tracer.startActiveSpan('llm.generate.gemini', async span => {
      // abilitiesDoc already kept up-to-date via registry event
      // Gemini v1beta text models no longer allow a standalone 'system' role –
      // instead prepend the system instructions to the user content.
      const systemMsg = `You are the tactical brain of the Hyper-Boss.\n\nReply ONLY with raw JSON (no markdown).  Valid top-level fields:\n  • \"explain\"      – short reasoning sentence\n  • \"self_score\"   – 0-1 confidence of success\n  • \"actions\"      – array of { ability, args }\n  • \"define_component\" – optional { manifest, impl } to add a new power\n\n### EXAMPLE good plan\n{\n  \"explain\":\"Dash west to dodge then scatter players\",\n  \"self_score\":0.83,\n  \"actions\":[\n    {\"ability\":\"dash\",\"args\":{\"dx\":-1,\"dy\":0,\"speed\":12,\"duration\":0.3}},\n    {\"ability\":\"projectile_spread\",\"args\":{\"count\":18,\"arc\":90,\"speed\":10}}\n  ]\n}\n\n### EXAMPLE adding a new power\n{\n  \"explain\":\"Introduce grenade ability then use it\",\n  \"self_score\":0.9,\n  \"define_component\":{\n     \"manifest\":\"{\\\"$id\\\":\\\"Emitter:Grenade@0.1.0\\\",\\\"type\\\":\\\"object\\\",\\\"properties\\\":{\\\"speed\\\":{\\\"type\\\":\\\"number\\\"}}}\",\n     \"impl\":\"export function compile(b){return{ability:'grenade',args:{speed:b.speed||6},_capType:b.type}};export function invoke(){return true};\"\n  },\n  \"actions\":[{\"ability\":\"grenade\",\"args\":{\"speed\":6}}]\n}\n\nIf none of the abilities fit, propose a new capability as shown.\nKeep manifest < 4kB, perfCost ≤ 3.0.\n\nAllowed abilities (snake_case): ${abilitiesDoc}.`;
      const userMsg   = `${systemMsg}\n\nYou are the tactical brain of the boss.\nWORLD_STATE:\n${JSON.stringify(snapshot)}`;

      const t0 = Date.now();

      // --- 15 s timeout guard so server loop never blocks indefinitely ---
      const ctrl  = new AbortController();
      const timer = setTimeout(() => ctrl.abort('LLM timeout'), 15_000);

      let res;
      try {
        res = await this.model.generateContent(
          {
            contents: [{ role: 'user', parts: [{ text: userMsg }] }],
            tools   : [{ function_declarations: [issueActionsFn] }],
            safety_settings: undefined,
          },
          { signal: ctrl.signal }
        );
      } finally {
        clearTimeout(timer);
      }

      // Debug: Log response structure
      if (process.env.DEBUG_GEMINI) {
        console.log('[Gemini Debug] Response:', JSON.stringify(res?.response, null, 2));
      }

      // Try function call first (preferred)
      const call = res
        ?.response
        ?.candidates?.[0]
        ?.content?.parts?.[0]
        ?.functionCall;

      let payload;

      if (call) {
        // Function call mode (structured response)
        if (call.name !== 'issue_actions') {
          throw new Error(`Gemini: unexpected function '${call.name}'`);
        }
        payload = call.args;
        if (typeof payload !== 'object' || payload === null) {
          throw new Error('Gemini: functionCall.args is not an object');
        }
      } else {
        // Fallback: parse text response as JSON
        const textPart = res?.response?.candidates?.[0]?.content?.parts?.[0]?.text;
        if (!textPart) {
          throw new Error('Gemini: no function call or text response');
        }

        // Extract JSON from potential markdown code blocks
        let jsonText = textPart.trim();
        const jsonMatch = jsonText.match(/```(?:json)?\s*\n?([\s\S]*?)\n?```/);
        if (jsonMatch) {
          jsonText = jsonMatch[1].trim();
        }

        try {
          payload = JSON.parse(jsonText);
        } catch (err) {
          console.error('[Gemini] Failed to parse JSON response:', jsonText.substring(0, 200));
          throw new Error(`Gemini: invalid JSON response - ${err.message}`);
        }
      }

      const delta = Date.now() - t0;
      span.setAttribute('latency_ms', delta);
      span.setAttribute('tokens', res.response.usageMetadata?.totalTokens ?? 0);
      span.end();

      return {
        json    : payload,
        deltaMs : delta,
        tokens  : res.response.usageMetadata?.totalTokens ?? 0,
      };
    });
  }

  /**
   * Generate a short taunt / speech line. Keeps prompt minimal and expects raw JSON line.
   */
  async generateSpeech(snapshot) {
    return tracer.startActiveSpan('llm.generate.gemini', async span => {
      const prompt =
        'You are the boss. Speak a short taunt or comment. Reply ONLY with JSON {"line":string,"emote":string?}\n' +
        'WORLD_STATE:\n' + JSON.stringify(snapshot);

      const t0 = Date.now();
      const res = await this.model.generateContent({
        contents: [{ role: 'user', parts: [{ text: prompt }] }]
      });

      const delta = Date.now() - t0;
      span.setAttribute('latency_ms', delta);
      span.setAttribute('tokens', res.response.usageMetadata?.totalTokens ?? 0);
      span.end();

      let payload = null;
      const txt = res?.response?.candidates?.[0]?.content?.parts?.[0]?.text || '';
      try {
        payload = JSON.parse(txt);
      } catch(_){ /* swallow */ }

      return {
        json    : payload,
        deltaMs : delta,
        tokens  : res.response.usageMetadata?.totalTokens ?? 0,
      };
    });
  }
}
