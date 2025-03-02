import { MovementBehavior } from "./MovementBehavior.js";
import { AttackBehavior } from "./AttackBehavior.js";
import { DefensiveBehavior } from "./DefensiveBehavior.js";
import { SummonBehavior } from "./SummonBehavior.js";
import { BuffBehavior } from "./BuffBehavior.js";
import {
  LayTrapBehavior,
  TimeDistortBehavior,
  GravityWellBehavior
} from "./AdditionalBehaviors.js";

export function createBlackOverlord() {
    const entity = { x: 0, y: 0, hp: 100, maxHp: 100, sprite: "blackOverlord" };
  
    const keepDistance = new MovementBehavior({
      mode: "maintainDistance",
      distance: 8,
      speed: 2
    });
  
    const homingShoot = new AttackBehavior({
      cooldown: 2000,
      pattern: "homing", 
      numProjectiles: 3,
      spreadAngle: 0.3,
      damage: 15,
      piercing: true
    });
  
    const phase = new DefensiveBehavior({
      mode: "phase",
      cooldown: 8000,
      phaseDuration: 2000
    });
  
    const layTrap = new LayTrapBehavior({
      cooldown: 6000,
      trapData: { damage: 20, radius: 1.5, duration: 4000 }
    });
  
    entity.update = (target, dt) => {
      keepDistance.update(entity, target, dt);
      homingShoot.update(entity, target, dt);
      phase.update(entity, target, dt);
      layTrap.update(entity, target, dt);
    };
  
    return entity;
  }

  
  export function createGoldGuardian() {
    const entity = { x: 0, y: 0, hp: 120, maxHp: 120, sprite: "goldGuardian" };
  
    const slowMove = new MovementBehavior({
      mode: "approach", 
      speed: 0.5,
      distance: 3
    });
  
    const largeProjectileShoot = new AttackBehavior({
      cooldown: 2500,
      pattern: "straight",
      damage: 20,
      projectileSpeed: 3
    });
  
    // Simulate "Block" with reflect
    const blockDefensive = new DefensiveBehavior({
      mode: "reflect",
      cooldown: 8000,
      reflectDuration: 2000
    });
  
    const defenseBuff = new BuffBehavior({
      stat: "defense",
      amount: 5,
      radius: 3,
      duration: 3000,
      cooldown: 8000
    });
  
    entity.update = (target, dt) => {
      slowMove.update(entity, target, dt);
      largeProjectileShoot.update(entity, target, dt);
      blockDefensive.update(entity, target, dt);
      defenseBuff.update(entity, target, dt);
    };
  
    return entity;
  }
  
  export function createRedBerserker() {
    const entity = { x: 0, y: 0, hp: 80, maxHp: 80, sprite: "redBerserker" };
  
    const chargeApproach = new MovementBehavior({
      mode: "approach",
      speed: 3,
      distance: 1
    });
  
    const explosiveShoot = new AttackBehavior({
      cooldown: 1200,
      pattern: "random", 
      numProjectiles: 5,
      spreadAngle: 0.5,
      damage: 10,
      explodeOnLowHealth: true, 
      explosionRadius: 3,
      explosionDamage: 30
    });
  
    entity.update = (target, dt) => {
      chargeApproach.update(entity, target, dt);
      explosiveShoot.update(entity, target, dt);
    };
  
    return entity;
  }
  
  export function createPurpleIllusionist() {
    const entity = { x: 0, y: 0, hp: 60, maxHp: 60, sprite: "purpleIllusionist" };
  
    const randomMove = new MovementBehavior({
      mode: "random",
      speed: 2
    });
  
    const waveShoot = new AttackBehavior({
      cooldown: 2000,
      pattern: "wave",
      numProjectiles: 2,
      spreadAngle: 0.2,
      damage: 12
    });
  
    const disguise = new DefensiveBehavior({
      mode: "disguise",
      cooldown: 10000,
      disguiseDuration: 4000,
      disguisedSprite: "barrel.png"
    });
  
    const teleport = new DefensiveBehavior({
      mode: "teleport",
      cooldown: 8000,
      teleportRange: 5
    });
  
    const layTrap = new LayTrapBehavior({
      cooldown: 7000,
      trapData: { damage: 15, radius: 1, duration: 5000 }
    });
  
    entity.update = (target, dt) => {
      randomMove.update(entity, target, dt);
      waveShoot.update(entity, target, dt);
      disguise.update(entity, target, dt);
      teleport.update(entity, target, dt);
      layTrap.update(entity, target, dt);
    };
  
    return entity;
  }

  
  export function createEmeraldRegenerator() {
    const entity = { x: 0, y: 0, hp: 90, maxHp: 90, sprite: "emeraldRegenerator" };
  
    const slowApproach = new MovementBehavior({
      mode: "approach",
      speed: 1,
      distance: 3
    });
  
    const summonMinions = new SummonBehavior({
      cooldown: 6000,
      maxMinions: 3,
      createMinion: () => ({ x: 0, y: 0, hp: 15, maxHp: 15, sprite: "emeraldMinion" })
    });
  
    const healBuff = new BuffBehavior({
      stat: "defense", // or HP if your engine supports direct healing
      amount: 3,
      radius: 4,
      duration: 3000,
      cooldown: 8000
    });
  
    const areaDenialShoot = new AttackBehavior({
      cooldown: 3000,
      pattern: "wave",
      projectileSpeed: 2,
      damage: 8,
      numProjectiles: 3,
      spreadAngle: 0.4
    });
  
    entity.update = (target, dt) => {
      slowApproach.update(entity, target, dt);
      summonMinions.update(entity, target, dt);
      healBuff.update(entity, target, dt);
      areaDenialShoot.update(entity, target, dt);
    };
  
    return entity;
  }
  
  export function createNavyTurtle() {
    const entity = { x: 0, y: 0, hp: 150, maxHp: 150, sprite: "navyTurtle" };
  
    const slowMove = new MovementBehavior({
      mode: "approach",
      speed: 0.7,
      distance: 2
    });
  
    const barrierShoot = new AttackBehavior({
      cooldown: 2500,
      pattern: "straight",
      projectileSpeed: 2,
      damage: 12,
      piercing: true
    });
  
    const blockReflect = new DefensiveBehavior({
      mode: "reflect",
      cooldown: 7000,
      reflectDuration: 2500
    });
  
    // A simple "regen" trick: BuffBehavior that adds HP frequently
    const regen = new BuffBehavior({
      stat: "hp",
      amount: 5, 
      radius: 0,
      duration: 1,
      cooldown: 2000
    });
  
    entity.update = (target, dt) => {
      slowMove.update(entity, target, dt);
      barrierShoot.update(entity, target, dt);
      blockReflect.update(entity, target, dt);
      regen.update(entity, target, dt);
    };
  
    return entity;
  }

  
  export function createMaroonVampire() {
    const entity = { x: 0, y: 0, hp: 70, maxHp: 70, sprite: "maroonVampire" };
  
    const approach = new MovementBehavior({
      mode: "approach",
      speed: 2,
      distance: 1
    });
  
    const retreat = new MovementBehavior({
      mode: "flee",
      speed: 2.5,
      fleeDistance: 1.0
    });
  
    const spiralLeechShoot = new AttackBehavior({
      cooldown: 1500,
      pattern: "spiral",
      numProjectiles: 2,
      spreadAngle: 0.3,
      damage: 8,
      leechOnHit: true,
      leechAmount: 5
    });
  
    entity.update = (target, dt) => {
      approach.update(entity, target, dt);
      retreat.update(entity, target, dt);
      spiralLeechShoot.update(entity, target, dt);
    };
  
    return entity;
  }
  
  export function createSilverySniper() {
    const entity = { x: 0, y: 0, hp: 60, maxHp: 60, sprite: "silverySniper" };
  
    const maintainDistance = new MovementBehavior({
      mode: "maintainDistance",
      speed: 2.5,
      distance: 10
    });
  
    const preciseShoot = new AttackBehavior({
      cooldown: 2000,
      pattern: "straight",
      damage: 20,
      projectileSpeed: 6
    });
  
    const teleport = new DefensiveBehavior({
      mode: "teleport",
      cooldown: 6000,
      teleportRange: 5
    });
  
    const disguise = new DefensiveBehavior({
      mode: "disguise",
      cooldown: 10000,
      disguiseDuration: 4000,
      disguisedSprite: "rock.png"
    });
  
    entity.update = (target, dt) => {
      maintainDistance.update(entity, target, dt);
      preciseShoot.update(entity, target, dt);
      teleport.update(entity, target, dt);
      disguise.update(entity, target, dt);
    };
  
    return entity;
  }

  export function createOrangeMadman() {
    const entity = { x: 0, y: 0, hp: 60, maxHp: 60, sprite: "orangeMadman" };
  
    const randomMove = new MovementBehavior({
      mode: "random",
      speed: 2,
      changeInterval: 1000
    });
  
    const randomShoot = new AttackBehavior({
      cooldown: 1200,
      pattern: "random",
      numProjectiles: 4,
      spreadAngle: 1.0,
      damage: 8
    });
  
    const layTrap = new LayTrapBehavior({
      cooldown: 4000,
      trapData: { damage: 10, radius: 1, duration: 3000 }
    });
  
    entity.update = (target, dt) => {
      randomMove.update(entity, target, dt);
      randomShoot.update(entity, target, dt);
      layTrap.update(entity, target, dt);
    };
  
    return entity;
  }
  
  export function createBrownTank() {
    const entity = { x: 0, y: 0, hp: 140, maxHp: 140, sprite: "brownTank" };
  
    const slowApproach = new MovementBehavior({
      mode: "approach",
      speed: 1,
      distance: 2
    });
  
    const shortBurstShoot = new AttackBehavior({
      cooldown: 2000,
      numProjectiles: 3,
      spreadAngle: 0.2,
      damage: 10,
      projectileSpeed: 4
    });
  
    const blockReflect = new DefensiveBehavior({
      mode: "reflect",
      cooldown: 6000,
      reflectDuration: 2000
    });
  
    entity.update = (target, dt) => {
      slowApproach.update(entity, target, dt);
      shortBurstShoot.update(entity, target, dt);
      blockReflect.update(entity, target, dt);
    };
  
    return entity;
  }

  
  export function createTealScout() {
    const entity = { x: 0, y: 0, hp: 40, maxHp: 40, sprite: "tealScout" };
  
    const strafe = new MovementBehavior({
      mode: "strafe",
      speed: 3,
      distance: 5
    });
  
    const flee = new MovementBehavior({
      mode: "flee",
      speed: 4,
      fleeDistance: 2
    });
  
    const singleShot = new AttackBehavior({
      cooldown: 800,
      pattern: "straight",
      damage: 6,
      projectileSpeed: 6
    });
  
    entity.update = (target, dt) => {
      strafe.update(entity, target, dt);
      flee.update(entity, target, dt);
      singleShot.update(entity, target, dt);
    };
  
    return entity;
  }
  
  export function createGreenSpawner() {
    const entity = { x: 0, y: 0, hp: 50, maxHp: 50, sprite: "greenSpawner" };
  
    const flee = new MovementBehavior({
      mode: "flee",
      speed: 2.5,
      fleeDistance: 3
    });
  
    const summon = new SummonBehavior({
      cooldown: 5000,
      maxMinions: 4,
      createMinion: () => ({ x: 0, y: 0, hp: 10, maxHp: 10, sprite: "greenMinion" })
    });
  
    const lowDamageShoot = new AttackBehavior({
      cooldown: 1500,
      damage: 5,
      pattern: "straight",
      projectileSpeed: 4
    });
  
    const layTrap = new LayTrapBehavior({
      cooldown: 8000,
      trapData: { damage: 8, radius: 1, duration: 4000 }
    });
  
    entity.update = (target, dt) => {
      flee.update(entity, target, dt);
      summon.update(entity, target, dt);
      lowDamageShoot.update(entity, target, dt);
      layTrap.update(entity, target, dt);
    };
  
    return entity;
  }

  export function createBlueFodder() {
    const entity = { x: 0, y: 0, hp: 30, maxHp: 30, sprite: "blueFodder" };
  
    const simpleApproach = new MovementBehavior({
      mode: "approach",
      speed: 1.5,
      distance: 1
    });
  
    const linearShoot = new AttackBehavior({
      cooldown: 1000,
      pattern: "straight",
      damage: 5,
      projectileSpeed: 5
    });
  
    entity.update = (target, dt) => {
      simpleApproach.update(entity, target, dt);
      linearShoot.update(entity, target, dt);
    };
  
    return entity;
  }
  

