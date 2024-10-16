// constants.js

// Constants for game dimensions and rendering
export const tSize = 50;
export const tHalf = tSize / 2;
export const screen = 600;
export const xCenter = 300;
export let yCenter = 324;
export let offCenter = false;
export const offCenterDiff = 450 - 324;
export const min = -tSize * 1.5;
export const max = screen + tSize * 2;
export const WALL_SIZE = 1;

// Constants for rendering
export const c3w = 128; // Canvas 3 width

// Added missing constants
export const scanC1 = -9;
export const scanC2 = 11;
export const scanOC1 = -11;
export const scanOC2 = 13;
export let scan1 = scanC1;
export let scan2 = scanC2;

export const hitbox = 0.25;

// For updating scan values based on offCenter
export function updateScanValues() {
  if (offCenter) {
    scan1 = scanOC1;
    scan2 = scanOC2;
    yCenter = 450;
  } else {
    scan1 = scanC1;
    scan2 = scanC2;
    yCenter = 324;
  }
}
