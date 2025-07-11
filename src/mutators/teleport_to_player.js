// src/mutators/teleport_to_player.js
// Teleport boss to target player or random player

/**
 * Teleport boss to a player location
 * @param {object} state - per-action scratch object
 * @param {{target:string, offset:number, effect:boolean}} args
 * @param {number} dt - deltaTime (seconds)
 * @param {*} bossMgr - BossManager instance
 * @param {*} bulletMgr - BulletManager instance
 * @param {*} mapMgr - MapManager instance
 * @param {*} enemyMgr - EnemyManager instance
 * @returns {boolean} finished?
 */
export function teleport_to_player(state, args, dt, bossMgr, bulletMgr, mapMgr, enemyMgr) {
  if (state.done) return true;

  const idx = 0;
  const target = args.target ?? 'random'; // 'random', 'nearest', 'farthest', 'highest_dps'
  const offset = args.offset ?? 2; // distance from player
  const effect = args.effect ?? true; // show teleport effect
  
  // Get target player (simplified - would need access to player manager)
  const targetPlayer = getTargetPlayer(target, bossMgr, idx);
  
  if (targetPlayer) {
    // Calculate teleport position with offset
    const angle = Math.random() * Math.PI * 2;
    const newX = targetPlayer.x + Math.cos(angle) * offset;
    const newY = targetPlayer.y + Math.sin(angle) * offset;
    
    // Store old position for effect
    if (effect) {
      state.oldX = bossMgr.x[idx];
      state.oldY = bossMgr.y[idx];
    }
    
    // Teleport boss
    bossMgr.x[idx] = newX;
    bossMgr.y[idx] = newY;
    
    // Add teleport effect if requested
    if (effect && bulletMgr.addEffect) {
      // Teleport out effect
      bulletMgr.addEffect({
        type: 'teleport_out',
        x: state.oldX,
        y: state.oldY,
        duration: 0.5
      });
      
      // Teleport in effect
      bulletMgr.addEffect({
        type: 'teleport_in',
        x: newX,
        y: newY,
        duration: 0.5
      });
    }
  }

  state.done = true;
  return true;
}

function getTargetPlayer(targetType, bossMgr, bossIdx) {
  // Simplified implementation - in real game would access player manager
  // For now, return a mock player position
  const mockPlayers = [
    { x: bossMgr.x[bossIdx] + 10, y: bossMgr.y[bossIdx] + 5 },
    { x: bossMgr.x[bossIdx] - 8, y: bossMgr.y[bossIdx] + 12 },
    { x: bossMgr.x[bossIdx] + 15, y: bossMgr.y[bossIdx] - 7 }
  ];
  
  switch (targetType) {
    case 'random':
      return mockPlayers[Math.floor(Math.random() * mockPlayers.length)];
    case 'nearest':
      return mockPlayers[0]; // Would calculate actual nearest
    case 'farthest':
      return mockPlayers[mockPlayers.length - 1]; // Would calculate actual farthest
    default:
      return mockPlayers[0];
  }
}