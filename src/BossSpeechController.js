import { createProvider } from './llm/ProviderFactory.js';
import { logLLM } from './llm/llmLogger.js';
import llmConfig from './config/llmConfig.js';

export default class BossSpeechController {
  constructor(bossMgr, networkMgr) {
    this.bossMgr = bossMgr;
    this.networkMgr = networkMgr;
    this.timer = 0;
    this.period = llmConfig.speechPeriodSec;

    this.provider = null;
  }

  async tick(dt, players) {
    this.timer += dt;
    if (this.timer < this.period) return;
    this.timer = 0;
    const snap = this.bossMgr.buildSnapshot(players, 0);
    try {
      if (!this.provider) this.provider = createProvider();
      const { json: speech } = await (typeof this.provider.generateSpeech === 'function'
        ? this.provider.generateSpeech(snap)
        : this.provider.generate(snap));
      if (speech) {
        logLLM({ ts: Date.now(), type:'speech_line', ...speech });
        console.log(`[Boss] says: ${speech.line}`);
        if (this.networkMgr) {
          this.networkMgr.broadcast({ type:'boss-speech', line: speech.line, emote: speech.emote||null });
        }
      }
    } catch(err) {
      console.warn('[BossSpeech] failed', err);
    }
  }
} 