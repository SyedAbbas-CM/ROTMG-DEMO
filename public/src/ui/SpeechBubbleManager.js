// public/src/ui/SpeechBubbleManager.js
// Very small, dependency-free manager for text bubbles (à la Realm of the Mad God)
// ────────────────────────────────────────────────────────────────────────────
// Usage:
//   import { speechBubbleManager } from '../ui/SpeechBubbleManager.js';
//   speechBubbleManager.addBubble({ id, idType:'player'|'enemy'|'object'|'tile', text:'Hello!', ttl:4000 });
//
// The manager stores bubbles and provides `update(time)` and `render(ctx)` for
// top-down view plus `renderFirstPerson(camera, overlayDiv)` for FP.  For now
// we implement only the 2-D canvas variant – FP support can be added later by
// re-using the same data.

import { gameState } from '../game/gamestate.js';

class SpeechBubbleManager {
  constructor() {
    this.bubbles = new Map(); // key: `${idType}:${id}` → bubble
  }

  /**
   * Adds (or replaces) a speech bubble.
   * @param {Object} opts {id, idType, text, ttl=4000, color='#fff'}
   */
  addBubble(opts) {
    const { id, idType='player', text='', ttl=4000, color='#ffffff' } = opts;
    if (!id || !text) return;
    const key = `${idType}:${id}`;
    this.bubbles.set(key, {
      id, idType, text, color,
      expiresAt: performance.now() + ttl
    });
  }

  /** call each frame */
  update(now=performance.now()) {
    for (const [key, bubble] of this.bubbles) {
      // lifetime expired
      if (bubble.expiresAt < now) {
        this.bubbles.delete(key);
        continue;
      }

      // entity no longer exists → drop instantly to avoid ghosts
      if (!this._getEntityWorldPos(bubble)) {
        this.bubbles.delete(key);
      }
    }
  }

  /**
   * Draw bubbles in the 2-D top-down canvas.
   * @param {CanvasRenderingContext2D} ctx context of gameCanvas (already scaled)
   */
  render(ctx) {
    if (!ctx) return;
    const cam = gameState.camera;
    if (!cam) return;

    ctx.save();
    ctx.font = 'bold 6px monospace';
    ctx.textBaseline = 'bottom';

    for (const bubble of this.bubbles.values()) {
      const pos = this._getEntityWorldPos(bubble);
      if (!pos) continue;
      const screen = cam.worldToScreen(pos.x, pos.y, ctx.canvas.width, ctx.canvas.height);
      const text = bubble.text;
      const metrics = ctx.measureText(text);
      const pad = 2;
      const w = metrics.width + pad*2;
      const h = 8;
      const x = screen.x - w/2;
      const y = screen.y - 16; // 16 px above entity centre

      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(x, y-h, w, h);
      ctx.strokeStyle = '#000';
      ctx.strokeRect(x, y-h, w, h);
      ctx.fillStyle = bubble.color;
      ctx.fillText(text, x+pad, y-1);
    }
    ctx.restore();
  }

  _getEntityWorldPos(bubble) {
    const { id, idType } = bubble;
    switch(idType) {
      case 'player': {
        const p = (id === gameState.character?.id) ? gameState.character : gameState.playerManager?.players.get(id);
        if (p) return { x: p.x, y: p.y };
        break;
      }
      case 'enemy': {
        const em = gameState.enemyManager;
        if (!em) break;
        const idx = em.findIndexById?.(id);
        if (idx !== -1 && idx !== undefined) return { x: em.x[idx], y: em.y[idx] };
        break;
      }
      default: break;
    }
    return null;
  }
}

export const speechBubbleManager = new SpeechBubbleManager(); 