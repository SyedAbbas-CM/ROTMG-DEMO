import fs from 'fs';
import path from 'path';
import Ajv from 'ajv';
import { BehaviorState } from './Behaviours/BehaviorState.js';
import * as Behaviors from './Behaviours/Behaviors.js';
import * as Transitions from './Behaviours/Transitions.js';

const ajv = new Ajv({ allErrors: true, strict:false });
const __dirname = path.dirname(new URL(import.meta.url).pathname);
let schemaPath = path.join(__dirname,'schema','enemySchema.json');
if(!fs.existsSync(schemaPath)){
  // fallback to cwd
  schemaPath = path.join(process.cwd(),'src','schema','enemySchema.json');
}
const schema = JSON.parse(fs.readFileSync(schemaPath,'utf8'));
const validate = ajv.compile(schema);

/**
 * Behavior Factory - Creates behavior instances from JSON definitions
 *
 * Available behaviors:
 * - wander: Random movement within area (speed, duration)
 * - chase: Move toward target (speed, minDistance)
 * - follow: Smart follow with acquire/lose range (speed, acquireRange, loseRange, stopDistance)
 * - shoot: Fire projectiles at target (cooldownMultiplier, projectileCount, spread, inaccuracy)
 * - orbit: Circle around target (speed, radius)
 * - runAway: Flee from target (speed, maxDistance)
 * - protect: Guard another entity (protectType, speed, acquireRange, protectionRange)
 * - teleportToTarget: Instant movement near target (range, cooldown)
 * - tossObject: Spawn entities (childType, range, cooldown)
 * - suicide: Self-destruct behavior
 * - swirl: Spiral movement pattern (speed, radius, rotationSpeed, clockwise)
 * - charge: Cavalry-style charge attack (chargeSpeed, acceleration, duration, range, idleSpeed, deceleration)
 * - directionalShoot: Shoot in movement direction (cooldownMultiplier, coneAngle)
 */
const behaviourFactory = {
  // Movement behaviors
  wander: p => new Behaviors.Wander(p.speed || 1, p.duration || 3),
  chase: p => new Behaviors.Chase(p.speed || 1, p.minDistance || 0),
  follow: p => new Behaviors.Follow(p.speed || 1, p.acquireRange || 120, p.loseRange || 160, p.stopDistance || 20),
  orbit: p => new Behaviors.Orbit(p.speed || 1, p.radius || 6),
  runaway: p => new Behaviors.RunAway(p.speed || 1.5, p.maxDistance || 500),
  swirl: p => new Behaviors.Swirl(p.speed || 1, p.radius || 5, p.rotationSpeed || 8, p.clockwise !== false),
  charge: p => new Behaviors.CavalryCharge(
    p.chargeSpeed || 2.5,
    p.acceleration || 1.2,
    p.duration || 2.0,
    p.range || 8.0,
    p.idleSpeed || 0.3,
    p.deceleration || 2.5
  ),

  // Attack behaviors
  shoot: p => new Behaviors.Shoot(p.cooldownMultiplier || 1, p.projectileCount || 1, p.spread || 0, p.inaccuracy || 0),
  directionalshoot: p => new Behaviors.DirectionalShoot(p.cooldownMultiplier || 1, p.coneAngle || Math.PI / 3),

  // Utility behaviors
  protect: p => new Behaviors.Protect(p.protectType || 0, p.speed || 1, p.acquireRange || 10, p.protectionRange || 2, p.reprotectRange || 1),
  // teleporttotarget: Not implemented - use charge instead
  spawn: p => new Behaviors.Spawn(p.childType || 0, p.count || 1, p.cooldown || 5),
  healself: p => new Behaviors.HealSelf(p.amount || 50, p.cooldown || 5)
};

/**
 * Transition Factory - Creates state transitions from JSON definitions
 *
 * Available transitions:
 * - timed: After fixed duration (time in seconds)
 * - randomTimer: After random duration (minTime, maxTime)
 * - playerWithin: When player enters range (range in tiles)
 * - noPlayerWithin: When player leaves range (range in tiles)
 * - healthBelow: When HP drops below threshold (threshold 0.0-1.0)
 * - healthAbove: When HP rises above threshold (threshold 0.0-1.0)
 * - damageTaken: After taking specified damage (threshold)
 * - randomChance: Random probability per second (chance 0.0-1.0)
 * - entityExists: When specific entity exists nearby (entityId, range)
 * - entityNotExists: When specific entity doesn't exist (entityId, range)
 */
function createTransition(tr, targetState) {
  switch (tr.type) {
    case 'timed':
      return new Transitions.TimedTransition(tr.time || 1, targetState);
    case 'randomTimer':
      return new Transitions.RandomTimer(tr.minTime || 1, tr.maxTime || 3, targetState);
    case 'playerWithin':
      return new Transitions.PlayerWithinRange(tr.range || 8, targetState);
    case 'noPlayerWithin':
      return new Transitions.NoPlayerWithinRange(tr.range || 15, targetState);
    case 'healthBelow':
      return new Transitions.HealthBelow(tr.threshold || 0.5, targetState);
    case 'healthAbove':
      return new Transitions.HealthAbove(tr.threshold || 0.7, targetState);
    case 'damageTaken':
      return new Transitions.DamageTaken(tr.threshold || 100, targetState);
    case 'randomChance':
      return new Transitions.RandomChance(tr.chance || 0.1, targetState);
    case 'entityExists':
      return new Transitions.EntityExists(tr.entityId, tr.range || 0, targetState);
    case 'entityNotExists':
      return new Transitions.EntityNotExists(tr.entityId, tr.range || 0, targetState);
    default:
      console.warn(`[EnemyDefLoader] Unknown transition type: ${tr.type}`);
      return null;
  }
}

/**
 * Load *.enemy.json files from a directory, validate, and return array of defs.
 * @param {string} dir
 */
export function loadEnemyDefinitions(dir){
  if(!fs.existsSync(dir)) return [];
  // Filter out macOS resource fork files (._*) and only get .enemy.json files
  const files = fs.readdirSync(dir).filter(f=>f.endsWith('.enemy.json') && !f.startsWith('._'));
  const defs = [];
  files.forEach(f=>{
    try{
      const raw = JSON.parse(fs.readFileSync(path.join(dir,f),'utf8'));
      const ok = validate(raw);
      if(!ok){
        console.warn(`[EnemyDefLoader] Validation errors in ${f}:`, validate.errors);
        return;
      }
      defs.push(raw);
    }catch(err){
      console.error('[EnemyDefLoader] Failed to load',f,err);
    }
  });
  return defs;
}

/**
 * Compile an enemy definition JSON into a runtime template and behavior state machine
 * @param {Object} def - The enemy definition object
 * @returns {{ template: Object, rootState: BehaviorState }} The compiled enemy template and root state
 */
export function compileEnemy(def) {
  const stateObjs = new Map();

  // First pass: Create all behavior states
  Object.entries(def.states).forEach(([name, spec]) => {
    const bs = new BehaviorState(name); // Pass name for debugging
    (spec.behaviours || []).forEach(b => {
      let inst = null;
      if (typeof b === 'string') {
        // Simple string format: "wander"
        inst = behaviourFactory[b.toLowerCase()]?.({});
      } else if (typeof b === 'object') {
        // Object format: { "type": "wander", "speed": 0.5 } OR { "type": "wander", "params": { "speed": 0.5 } }
        const type = b.type.toLowerCase();
        // Support both formats: params on object directly OR nested in params field
        inst = behaviourFactory[type]?.(b.params || b);
      }
      if (inst) {
        bs.addBehavior(inst);
      } else {
        console.warn(`[EnemyDefLoader] Unknown behavior: ${typeof b === 'string' ? b : b.type}`);
      }
    });
    stateObjs.set(name, bs);
  });

  // Second pass: Wire up transitions between states
  Object.entries(def.states).forEach(([name, spec]) => {
    const bs = stateObjs.get(name);
    (spec.transitions || []).forEach(tr => {
      const targetState = stateObjs.get(tr.target);
      if (!targetState) {
        console.warn(`[EnemyDefLoader] Transition target state not found: ${tr.target}`);
        return;
      }
      const transition = createTransition(tr, targetState);
      if (transition) {
        bs.addTransition(transition);
      }
    });
  });

  // Get the initial state from def.initialState, then fallback to 'idle', 'default', or first state
  let rootState = stateObjs.get(def.initialState) || stateObjs.get('idle') || stateObjs.get('default') || stateObjs.values().next().value;

  // Build the enemy template with all attack properties
  const template = {
    id: def.id,
    name: def.name,
    spriteName: (def.sprite || '').replace(/^chars:/, ''),
    maxHealth: def.hp,
    speed: def.speed || 10,
    width: def.width || 1,
    height: def.height || 1,
    renderScale: def.renderScale || 2,
    shootRange: def.attack?.range || 120,
    shootCooldown: (def.attack?.cooldown || 1000) / 1000,
    bulletSpeed: def.attack?.speed || 20,
    bulletLifetime: (def.attack?.lifetime || 2000) / 1000,
    projectileCount: def.attack?.projectileCount || def.attack?.count || 1,
    spread: (def.attack?.spread || 0) * Math.PI / 180,
    inaccuracy: (def.attack?.inaccuracy || 0) * Math.PI / 180,
    damage: def.attack?.damage || 10,
    bulletSpriteName: def.attack?.sprite || null,
    contactDamage: def.contactDamage || 0,
    knockbackForce: def.knockbackForce || 0,
    dropTable: def.loot?.drops || []
  };

  console.log(`[EnemyDefLoader] Compiled enemy: ${def.name} (${def.id}) with ${stateObjs.size} states`);
  return { template, rootState };
} 