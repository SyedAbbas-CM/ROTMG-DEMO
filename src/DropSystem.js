// src/DropSystem.js
import { logger } from './utils/logger.js';

// Bag colour priority table same as RotMG (0-6 used for now)
export const BAG_COLOUR_PRIORITY = [0,1,2,3,4,5,6];

/**
 * Evaluate drop table for an enemy type.
 * For now we support simple fixed probability entries.
 * dropTable example:
 *   [ { "id":1001, "prob":0.3, "bagType":0, "soulbound":false } ]
 * @param {Array} dropTable
 * @param {function} rng – optional RNG returning [0,1)
 * @returns {{items:number[], bagType:number}}
 */
export function rollDropTable(dropTable, rng=Math.random){
  if(!Array.isArray(dropTable) || dropTable.length===0) return {items:[],bagType:0};
  const rolled=[];
  let maxBag=0;
  dropTable.forEach(def=>{
    if(rng()< (def.prob ?? 1)){
      rolled.push(def.id);
      if((def.bagType??0) > maxBag) maxBag = def.bagType??0;
    }
  });
  return {items:rolled, bagType:maxBag};
}

export function getBagColourSprite(bagType){
  switch(bagType){
    case 0: return 'items_sprite_lootbag_white';
    case 1: return 'items_sprite_lootbag_brown';
    case 2: return 'items_sprite_lootbag_purple';
    case 3: return 'items_sprite_lootbag_orange';
    case 4: return 'items_sprite_lootbag_cyan';
    case 5: return 'items_sprite_lootbag_blue';
    case 6: return 'items_sprite_lootbag_red';
    default: return 'items_sprite_lootbag_white';
  }
}

logger.info('DropSystem','module loaded'); 