// utils.js

import { mouseX, mouseY, mouseDown } from './input.js';

// === UTILITY FUNCTIONS ===
export function rankArray(arr) {
  let indexedArray = arr.map((value, index) => ({ value, index }));

  indexedArray.sort((a, b) => b.value - a.value);

  let rankArrayResult = new Array(arr.length);

  indexedArray.forEach((item, rank) => {
    rankArrayResult[item.index] = rank;
  });

  return rankArrayResult;
}

export function within(x, y, w, h, state = 0, px, py) {
  switch (state) {
    case 0:
      return mouseX > x && mouseX < x + w && mouseY > y && mouseY < y + h;
    case 1:
      return px > x && px < x + w && py > y && py < y + h;
    case 2:
      return mouseX > x && mouseX < x + w && mouseY > y && mouseY < y + h && mouseDown;
    case 3:
      return px > x && px < x + w && py > y && py < y + h && mouseDown;
    default:
      return false;
  }
}
