/**
 * AbilitySystem - Handles ability cooldowns and execution
 */

export default class AbilitySystem {
  constructor() {
    // playerId -> { abilityId -> remainingCooldown }
    this.cooldowns = new Map();
    // playerId -> active effects { effectType -> endTime }
    this.activeEffects = new Map();
  }

  /**
   * Check if ability is ready
   */
  canUseAbility(playerId, abilityId) {
    const playerCooldowns = this.cooldowns.get(playerId);
    if (!playerCooldowns) return true;
    return (playerCooldowns[abilityId] || 0) <= 0;
  }

  /**
   * Get remaining cooldown for an ability
   */
  getCooldown(playerId, abilityId) {
    const playerCooldowns = this.cooldowns.get(playerId);
    if (!playerCooldowns) return 0;
    return Math.max(0, playerCooldowns[abilityId] || 0);
  }

  /**
   * Start cooldown for an ability
   */
  startCooldown(playerId, abilityId, duration) {
    if (!this.cooldowns.has(playerId)) {
      this.cooldowns.set(playerId, {});
    }
    this.cooldowns.get(playerId)[abilityId] = duration;
  }

  /**
   * Update cooldowns (call each tick)
   */
  update(deltaTime) {
    const now = Date.now();

    // Update cooldowns
    for (const [playerId, abilities] of this.cooldowns.entries()) {
      for (const abilityId in abilities) {
        abilities[abilityId] = Math.max(0, abilities[abilityId] - deltaTime);
      }
    }

    // Check expired effects
    for (const [playerId, effects] of this.activeEffects.entries()) {
      for (const [effectType, endTime] of Object.entries(effects)) {
        if (now >= endTime) {
          delete effects[effectType];
        }
      }
    }
  }

  /**
   * Check if player has active effect
   */
  hasEffect(playerId, effectType) {
    const effects = this.activeEffects.get(playerId);
    if (!effects) return false;
    const endTime = effects[effectType];
    return endTime && Date.now() < endTime;
  }

  /**
   * Add active effect to player
   */
  addEffect(playerId, effectType, durationMs) {
    if (!this.activeEffects.has(playerId)) {
      this.activeEffects.set(playerId, {});
    }
    this.activeEffects.get(playerId)[effectType] = Date.now() + durationMs;
  }

  /**
   * Execute ability effect
   */
  executeAbility(playerId, ability, player, bulletMgr, enemyMgr, worldId) {
    if (!this.canUseAbility(playerId, ability.id)) {
      return { success: false, reason: 'cooldown', remaining: this.getCooldown(playerId, ability.id) };
    }

    // Check mana
    if (player.mana < ability.manaCost) {
      return { success: false, reason: 'mana', required: ability.manaCost, current: player.mana };
    }

    // Deduct mana
    player.mana -= ability.manaCost;

    // Start cooldown
    this.startCooldown(playerId, ability.id, ability.cooldown);

    // Execute effect
    switch (ability.effect) {
      case 'charge':
        return this.executeCharge(player, bulletMgr, worldId, ability);
      case 'multishot':
        return this.executeMultishot(player, bulletMgr, worldId, ability);
      case 'aoe_explosion':
        return this.executeFireball(player, bulletMgr, worldId, ability);
      case 'stealth':
        return this.executeStealth(playerId, player, ability);
      case 'shield':
        return this.executeShield(playerId, player, ability);
      case 'lifedrain':
        return this.executeLifeDrain(player, enemyMgr, worldId, ability);
      default:
        return { success: false, reason: 'unknown_ability' };
    }
  }

  executeCharge(player, bulletMgr, worldId, ability) {
    const chargeDistance = ability.chargeDistance || 5;
    const angle = player.rotation || 0;

    // Store original position
    const startX = player.x;
    const startY = player.y;

    // Move player forward
    player.x += Math.cos(angle) * chargeDistance;
    player.y += Math.sin(angle) * chargeDistance;

    // Create damage trail
    if (bulletMgr) {
      for (let i = 0; i < 5; i++) {
        const trailX = startX + Math.cos(angle) * (i * chargeDistance / 5);
        const trailY = startY + Math.sin(angle) * (i * chargeDistance / 5);
        bulletMgr.addBullet({
          x: trailX,
          y: trailY,
          vx: 0,
          vy: 0,
          damage: ability.chargeDamage || 50,
          lifetime: 0.3,
          ownerId: player.id,
          worldId: worldId,
          width: 1.5,
          height: 1.5
        });
      }
    }

    return { success: true, effect: 'charge', distance: chargeDistance };
  }

  executeMultishot(player, bulletMgr, worldId, ability) {
    const baseAngle = player.rotation || 0;
    const spreadAngle = ability.spreadAngle || Math.PI / 6;
    const bulletCount = ability.arrowCount || 5;

    if (bulletMgr) {
      for (let i = 0; i < bulletCount; i++) {
        const angle = baseAngle + (i - (bulletCount - 1) / 2) * (spreadAngle / (bulletCount - 1));
        bulletMgr.addBullet({
          x: player.x + Math.cos(angle) * 0.5,
          y: player.y + Math.sin(angle) * 0.5,
          vx: Math.cos(angle) * 12,
          vy: Math.sin(angle) * 12,
          damage: player.damage || 15,
          lifetime: 1.5,
          ownerId: player.id,
          worldId: worldId
        });
      }
    }

    return { success: true, effect: 'multishot', count: bulletCount };
  }

  executeFireball(player, bulletMgr, worldId, ability) {
    const angle = player.rotation || 0;

    if (bulletMgr) {
      bulletMgr.addBullet({
        x: player.x + Math.cos(angle) * 0.5,
        y: player.y + Math.sin(angle) * 0.5,
        vx: Math.cos(angle) * 8,
        vy: Math.sin(angle) * 8,
        damage: ability.explosionDamage || 80,
        lifetime: 2.0,
        width: 1.0,
        height: 1.0,
        ownerId: player.id,
        worldId: worldId,
        isExplosive: true,
        explosionRadius: ability.explosionRadius || 3
      });
    }

    return { success: true, effect: 'fireball' };
  }

  executeStealth(playerId, player, ability) {
    const duration = ability.duration || 3;
    player.isStealthed = true;
    this.addEffect(playerId, 'stealth', duration * 1000);

    return { success: true, effect: 'stealth', duration };
  }

  executeShield(playerId, player, ability) {
    const duration = ability.duration || 2;
    player.isShielded = true;
    this.addEffect(playerId, 'shield', duration * 1000);

    return { success: true, effect: 'shield', duration };
  }

  executeLifeDrain(player, enemyMgr, worldId, ability) {
    const drainRadius = ability.drainRadius || 4;
    const drainAmount = ability.drainAmount || 30;
    let totalDrained = 0;

    if (enemyMgr) {
      for (let i = 0; i < enemyMgr.enemyCount; i++) {
        if (enemyMgr.worldId[i] !== worldId) continue;
        if (enemyMgr.health[i] <= 0) continue;

        const dx = enemyMgr.x[i] - player.x;
        const dy = enemyMgr.y[i] - player.y;
        const distSq = dx * dx + dy * dy;

        if (distSq <= drainRadius * drainRadius) {
          const damage = Math.min(drainAmount, enemyMgr.health[i]);
          enemyMgr.health[i] -= damage;
          totalDrained += damage;
        }
      }
    }

    // Heal player
    if (totalDrained > 0) {
      player.health = Math.min(player.maxHealth, player.health + totalDrained);
    }

    return { success: true, effect: 'lifedrain', drained: totalDrained };
  }

  /**
   * Clean up player data
   */
  removePlayer(playerId) {
    this.cooldowns.delete(playerId);
    this.activeEffects.delete(playerId);
  }
}
