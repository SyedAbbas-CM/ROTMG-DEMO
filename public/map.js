// map.js

import { assets } from './assets.js';

export const mapSize = 100;
export let map = new Uint8Array(mapSize * mapSize);
export const adjacentNWES = [-mapSize, -1, +1, +mapSize];

// Initialize texMap
export let texMap = new Map();

texMap.set(0, {
  x: 48,
  y: 8,
  tex: assets.envi,
  wall: false,
  solid: false,
  x2: 48,
  y2: 8,
  deco: false,
});
texMap.set(1, {
  x: 32,
  y: 8,
  tex: assets.envi,
  wall: true,
  solid: true,
  x2: 8,
  y2: 8,
  deco: false,
});

// Function to set the map (used in networking.js)
export function setMap(newMap) {
  map = newMap;
}

// Function to set texMap
export function setTexMap(newTexMap) {
  texMap = newTexMap;
}
