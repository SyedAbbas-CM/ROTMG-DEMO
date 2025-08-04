import fs from 'fs';
import path from 'path';
import Ajv from 'ajv';
import { BehaviorState } from './BehaviorState.js';
import * as Behaviors from './Behaviors.js';
import * as Transitions from './Transitions.js';

const ajv = new Ajv({ allErrors: true, strict:false });
const __dirname = path.dirname(new URL(import.meta.url).pathname);
let schemaPath = path.join(__dirname,'schema','enemySchema.json');
if(!fs.existsSync(schemaPath)){
  // fallback to cwd
  schemaPath = path.join(process.cwd(),'src','schema','enemySchema.json');
}
const schema = JSON.parse(fs.readFileSync(schemaPath,'utf8'));
const validate = ajv.compile(schema);

const behaviourFactory = {
  wander: p=> new Behaviors.Wander(p.speed||1,p.duration||3),
  follow: p=> new Behaviors.Follow(p.speed||1,p.acquireRange||120,p.loseRange||160,p.stopDistance||20),
  shoot:  p=> new Behaviors.Shoot(p.cooldownMultiplier||1,p.projectileCount||1,p.spread||0,p.inaccuracy||0),
  protect:p=> new Behaviors.Protect(p.protectType||0,p.speed||1,p.acquireRange||10,p.protectionRange||2,p.reprotectRange||1),
  teleportToTarget:p=> new Behaviors.TeleportToTarget(p.range||8,p.cooldown||2),
  tossObject:p=> new Behaviors.TossObject(p.childType||0,p.range||5,p.cooldown||3),
  suicide: ()=> new Behaviors.Suicide()
};

/**
 * Load *.enemy.json files from a directory, validate, and return array of defs.
 * @param {string} dir
 */
export function loadEnemyDefinitions(dir){
  if(!fs.existsSync(dir)) return [];
  const files = fs.readdirSync(dir).filter(f=>f.endsWith('.enemy.json'));
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

export function compileEnemy(def){
  const stateObjs = new Map();
  Object.entries(def.states).forEach(([name,spec])=>{
    const bs = new BehaviorState();
    (spec.behaviours||[]).forEach(b=>{
      let inst=null;
      if(typeof b==='string') inst = behaviourFactory[b]?.({}) ;
      else if(typeof b==='object'){
        const type=b.type.toLowerCase();
        inst = behaviourFactory[type]?.(b.params||{});
      }
      if(inst) bs.addBehavior(inst);
    });
    stateObjs.set(name, bs);
  });
  // transitions mapping
  Object.entries(def.states).forEach(([name,spec])=>{
    const bs = stateObjs.get(name);
    (spec.transitions||[]).forEach(tr=>{
      const targetState = stateObjs.get(tr.target);
      if(!targetState) return;
      switch(tr.type){
        case 'timed':
          bs.addTransition(new Transitions.TimedTransition(tr.time||1000,targetState));
          break;
        case 'playerWithin':
          bs.addTransition(new Transitions.PlayerWithinRange(tr.range||8,targetState));
          break;
      }
    });
  });
  const rootState = stateObjs.values().next().value;
  const template = {
    id: def.id, name:def.name, spriteName:def.sprite, maxHealth:def.hp, speed:def.speed||10,
    shootRange: def.attack?.range||120, shootCooldown:(def.attack?.cooldown||1000)/1000,
    bulletSpeed:def.attack?.speed||20, projectileCount:def.attack?.projectileCount||1,
    spread:(def.attack?.spread||0)*Math.PI/180, damage:def.attack?.damage||10
  };
  return {template, rootState};
} 