// src/BossManager.js
// Skeleton implementation for a single LLM-controlled boss.
// Responsible for deterministic physics, baseline attacks and snapshot building.

import { v4 as uuidv4 } from 'uuid';

export default class BossManager {
  /**
   * @param {Object|null} enemyMgr  – Reference to the EnemyManager so we can
   *                                 spawn a visible shell and keep it in sync.
   * @param {number} maxBosses
   */
  constructor(enemyMgr = null, maxBosses = 1) {
    this.enemyMgr = enemyMgr;
    this.maxBosses = maxBosses;
    this.bossCount = 0;

    // Structure of Arrays (SOA)
    this.id        = new Array(maxBosses);
    this.x         = new Float32Array(maxBosses);
    this.y         = new Float32Array(maxBosses);
    this.hp        = new Float32Array(maxBosses);      // 0-1 range (fraction)
    this.phase     = new Uint8Array(maxBosses);        // e.g. 0,1,2

    // Which world/map the boss lives in (string)
    this.worldId   = new Array(maxBosses);

    // Cooldown timers for baseline patterns
    this.cooldownDash  = new Float32Array(maxBosses);
    this.cooldownAOE   = new Float32Array(maxBosses);

    // Mapping bossIndex -> enemyManager index (for mirroring)
    this.enemyIndex    = new Int32Array(maxBosses).fill(-1);

    // Mapping for quick lookup
    this.idToIndex = new Map();

    // Per-boss action queue (array of arrays) – filled by ScriptBehaviourRunner / LLM
    this.actionQueue = Array.from({ length: maxBosses }, () => []);

    // Track previous health to compute delta within buildSnapshot()
    this.prevHp = new Float32Array(maxBosses).fill(1.0);
  }

  /**
   * Spawn a boss definition.  For now take just id and position; extend later.
   */
  spawnBoss(defId, x, y, worldId = 'default') {
    if (this.bossCount >= this.maxBosses) throw new Error('Max bosses reached');
    const i = this.bossCount++;

    this.id[i]   = uuidv4();
    this.x[i]    = x;
    this.y[i]    = y;
    this.hp[i]   = 1.0;      // full health
    this.phase[i] = 0;

    this.worldId[i] = worldId;

    this.cooldownDash[i] = 0;
    this.cooldownAOE[i]  = 0;

    this.idToIndex.set(this.id[i], i);

    // ----- Spawn a corresponding EnemyManager entity so the boss is visible & hittable
    if (this.enemyMgr) {
      try {
        const enemyId = this.enemyMgr.spawnEnemyById(defId, x, y, worldId);
        const eIdx = this.enemyMgr.findIndexById(enemyId);
        if (eIdx !== -1) {
          this.enemyIndex[i] = eIdx;
          console.log(`[BossManager] Linked boss ${defId} to enemy index ${eIdx}`);
        } else {
          console.warn(`[BossManager] Failed to locate enemy index for id ${enemyId}`);
        }
      } catch (err) {
        console.warn('[BossManager] spawnEnemyById failed', err);
      }
    }
    console.log(`[BossManager] Spawned boss ${defId} at (${x},${y}) id:${this.id[i]}`);
    return this.id[i];
  }

  /** Simple Euler integration placeholder. */
  tick(dt, bulletMgr) {
    for (let i = 0; i < this.bossCount; i++) {

      // ---- Mirror boss→enemy position so clients can render it
      const ei = this.enemyIndex[i];
      if (ei !== undefined && ei >= 0 && this.enemyMgr) {
        this.enemyMgr.x[ei] = this.x[i];
        this.enemyMgr.y[ei] = this.y[i];

        // Pull back health (enemy hitpoints → boss hp fraction)
        const h  = this.enemyMgr.health[ei];
        const mh = this.enemyMgr.maxHealth[ei] || 1;
        this.hp[i] = Math.max(0, Math.min(1, h / mh));
      }

      // Baseline automatic fire (cone AOE every 3 s)
      this.cooldownAOE[i] -= dt;
      if (this.cooldownAOE[i] <= 0) {
        this.cooldownAOE[i] = 3.0;   // hard-coded for MVP
        // Baseline radial burst 12 bullets
        if (bulletMgr && typeof bulletMgr.addBullet === 'function') {
          const projectiles = 12;
          const speed = 6;
          const twoPi = Math.PI * 2;
          for (let p = 0; p < projectiles; p++) {
            const theta = (p / projectiles) * twoPi;
            const eid = this.enemyIndex[i] >= 0 ? this.enemyMgr.id[this.enemyIndex[i]] : this.id[i];
            bulletMgr.addBullet({
              x: this.x[i],
              y: this.y[i],
              vx: Math.cos(theta) * speed,
              vy: Math.sin(theta) * speed,
              ownerId: eid,
              damage: 8,
              width: 0.6,
              height: 0.6,
              spriteName: null,
              worldId: this.worldId[i]
            });
          }
        }
      }

      // TODO: other baseline updates (movement, dash cooldown etc.)
    }
  }

  /**
   * Build a lightweight snapshot for the LLM containing additional tactical
   * context (bullets around boss, recent damage, etc.).
   * @param {Array}  players      – trimmed array sent from LLMBossController
   * @param {BulletManager|null} bulletMgr – optional bullet manager reference
   * @param {number} tickNumber   – server tick counter
   */
  buildSnapshot(players, bulletMgr = null, tickNumber = 0) {
    if (!this.bossCount) return null;
    const i = 0;
    // ---------- bullets summary (if manager provided) ----------
    let bulletsNear = 0;
    let uniqueAttackers = new Set();
    if (bulletMgr && bulletMgr.bulletCount) {
      const R2 = 144; // radius^2 (12 tiles) – tweak later
      for (let b = 0; b < bulletMgr.bulletCount; b++) {
        if (bulletMgr.worldId[b] !== this.worldId[i]) continue;
        const dx = bulletMgr.x[b] - this.x[i];
        const dy = bulletMgr.y[b] - this.y[i];
        if (dx*dx + dy*dy <= R2) {
          bulletsNear++;
          if (bulletMgr.ownerId[b]) uniqueAttackers.add(bulletMgr.ownerId[b]);
        }
      }
    }

    // ---------- recent damage (since last snapshot) ----------
    const hpDelta = Number((this.hp[i] - this.prevHp[i]).toFixed(3));
    this.prevHp[i] = this.hp[i];

    const snap = {
      tick : tickNumber,
      boss : {
        hp      : Number(this.hp[i].toFixed(2)),
        pos     : [ Number(this.x[i].toFixed(1)), Number(this.y[i].toFixed(1)) ],
        phase   : this.phase[i],
        cooldowns : {
          aoe  : Number(this.cooldownAOE[i].toFixed(1))
        },
        damageDelta : hpDelta,              // negative = damage taken
        bulletsNear,
        attackers   : Array.from(uniqueAttackers).slice(0,8)
      },
      players : players.slice(0, 8).map(p => ({
        id  : p.id,
        hp  : Number(p.hp.toFixed(2)),
        pos : [ Number(p.x.toFixed(1)), Number(p.y.toFixed(1)) ],
        vel : [ Number((p.vx||0).toFixed(2)), Number((p.vy||0).toFixed(2)) ]
      }))
    };
    return snap;
  }

  /** Lightweight helper for server broadcast. */
  getBossData(worldId) {
    if (!this.bossCount) return [];
    const arr = [];
    for (let i = 0; i < this.bossCount; i++) {
      if (worldId && this.worldId[i] !== worldId) continue;
      arr.push({
        id   : this.id[i],
        x    : this.x[i],
        y    : this.y[i],
        hp   : this.hp[i],
        worldId: this.worldId[i]
      });
    }
    return arr;
  }
} 