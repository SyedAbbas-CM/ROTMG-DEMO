/**
 * public/src/screens/previewScreen.js
 */

import { spriteManager } from '../assets/spriteManager.js';

const container = document.createElement('div');
container.style.background = '#444';
container.style.padding = '10px';
container.style.margin = '10px';
document.body.appendChild(container);

const title = document.createElement('h2');
title.textContent = 'Preview Screen';
container.appendChild(title);

const previewCanvas = document.createElement('canvas');
previewCanvas.width = 400;
previewCanvas.height = 300;
previewCanvas.style.border = '1px solid #999';
container.appendChild(previewCanvas);

const ctx = previewCanvas.getContext('2d');

let currentGroup = null;
let frameIndex = 0;
let frameInterval = 250;
let loopId = null;

/**
 * Render the current frame
 */
function renderFrame() {
  ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
  if (!currentGroup) return;

  const spriteKeys = spriteManager.getGroupSprites(currentGroup);
  if (!spriteKeys || spriteKeys.length === 0) return;

  const spriteKey = spriteKeys[frameIndex];
  const sprite = spriteManager.getSprite(spriteKey);
  if (!sprite) return;

  const image = spriteManager.getSheetImage(spriteKey);
  if (!image) return;

  // draw in the center of the canvas
  const dx = previewCanvas.width / 2 - sprite.width / 2;
  const dy = previewCanvas.height / 2 - sprite.height / 2;

  ctx.drawImage(
    image,
    sprite.x, sprite.y, sprite.width, sprite.height,
    dx, dy, sprite.width, sprite.height
  );

  frameIndex = (frameIndex + 1) % spriteKeys.length;
}

/**
 * Start animation
 */
function startAnimation() {
  if (loopId) clearInterval(loopId);
  loopId = setInterval(renderFrame, frameInterval);
}

/**
 * Stop animation
 */
function stopAnimation() {
  if (loopId) clearInterval(loopId);
  loopId = null;
}

/**
 * UI controls
 */
const controlsDiv = document.createElement('div');
controlsDiv.style.marginTop = '10px';
container.appendChild(controlsDiv);

// group selector
const groupSelect = document.createElement('select');
groupSelect.style.marginRight = '5px';
function populateGroupList() {
  groupSelect.innerHTML = '';
  Object.keys(spriteManager.groups).forEach(gName => {
    const opt = document.createElement('option');
    opt.value = gName;
    opt.textContent = gName;
    groupSelect.appendChild(opt);
  });
}
populateGroupList();
controlsDiv.appendChild(groupSelect);

// speed input
const speedLabel = document.createElement('label');
speedLabel.textContent = ' Interval (ms): ';
controlsDiv.appendChild(speedLabel);

const speedInput = document.createElement('input');
speedInput.type = 'number';
speedInput.value = frameInterval.toString();
speedLabel.appendChild(speedInput);

// start button
const startBtn = document.createElement('button');
startBtn.textContent = 'Play';
startBtn.style.marginLeft = '5px';
startBtn.addEventListener('click', () => {
  currentGroup = groupSelect.value;
  frameInterval = parseInt(speedInput.value, 10) || 250;
  frameIndex = 0;
  startAnimation();
});
controlsDiv.appendChild(startBtn);

// stop button
const stopBtn = document.createElement('button');
stopBtn.textContent = 'Stop';
stopBtn.style.marginLeft = '5px';
stopBtn.addEventListener('click', stopAnimation);
controlsDiv.appendChild(stopBtn);

/**
 * If you add or rename groups after this script loaded,
 * just call populateGroupList() again to refresh the dropdown.
 */
