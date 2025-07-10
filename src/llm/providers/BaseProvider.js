// src/llm/providers/BaseProvider.js
export class BaseProvider {
    /** @param {{model:string,temperature?:number,maxTokens?:number}} opts */
    constructor(opts) { this.opts = opts; }
    /** Return { json:any, deltaMs:number, tokens:number } */
    /* eslint-disable-next-line no-unused-vars */
    async generate(_snapshot) { throw new Error('generate() not implemented'); }

  /** Optional helper for short speech lines. Returns {json:{line:string,emote?:string},deltaMs:number,tokens:number} */
  /* eslint-disable-next-line no-unused-vars */
  async generateSpeech(_snapshot) {
    // Fallback: reuse main generate method if subclass didn\'t override.
    return this.generate(_snapshot);
  }
  }
  