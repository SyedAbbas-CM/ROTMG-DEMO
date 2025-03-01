// src/screens/spriteEditor.js

import { spriteManager } from '../assets/spriteManager.js';

// We'll wrap all UI creation in a container so we can show/hide the editor.
let editorContainer = null;

export function openSpriteEditor() {
  // If the editor is already open, do nothing.
  if (document.getElementById('spriteEditorContainer')) return;

  // Create a container for the sprite editor UI.
  editorContainer = document.createElement('div');
  editorContainer.id = 'spriteEditorContainer';
  editorContainer.style.position = 'fixed';
  editorContainer.style.top = '50px';
  editorContainer.style.left = '50px';
  editorContainer.style.background = '#222';
  editorContainer.style.padding = '10px';
  editorContainer.style.zIndex = '10000';
  editorContainer.style.maxHeight = '90vh';
  editorContainer.style.overflowY = 'auto';
  document.body.appendChild(editorContainer);

  // Create and append a title.
  const title = document.createElement('h2');
  title.textContent = 'Sprite Editor';
  title.style.color = 'white';
  editorContainer.appendChild(title);

  // Create the editor canvas.
  const editorCanvas = document.createElement('canvas');
  editorCanvas.id = 'spriteEditorCanvas';
  editorCanvas.width = 800;
  editorCanvas.height = 600;
  editorCanvas.style.border = '1px solid #999';
  editorContainer.appendChild(editorCanvas);
  const ctx = editorCanvas.getContext('2d');

  // Variables to hold current sheet data.
  let currentSheetName = null; // e.g. "hero_sprites"
  let spriteData = { name: '', path: '', sprites: [], groups: {} };
  let boundingBoxes = [];

  let isDrawing = false;
  let startX = 0;
  let startY = 0;

  // --- UI for choosing a sprite sheet ---
  async function loadSpriteSheetList() {
    try {
      const sheetNames = await fetch('/assets/spritesheets').then(r => r.json());
      const dropdown = document.createElement('select');
      dropdown.id = 'spriteSheetDropdown';

      const placeholderOpt = document.createElement('option');
      placeholderOpt.value = '';
      placeholderOpt.textContent = 'Select a sprite sheet...';
      dropdown.appendChild(placeholderOpt);

      sheetNames.forEach(name => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        dropdown.appendChild(option);
      });

      dropdown.addEventListener('change', e => {
        if (!e.target.value) return;
        loadSpriteSheet(e.target.value);
      });

      editorContainer.appendChild(dropdown);
    } catch (err) {
      console.error('Failed to load sprite sheet list:', err);
    }
  }

  // --- Load a specific sprite sheet (image + JSON data) ---
  async function loadSpriteSheet(sheetName) {
    currentSheetName = sheetName;

    // (a) Load the image into spriteManager.
    const config = { name: sheetName, path: `/assets/spritesheets/${sheetName}.png`, 
      defaultSpriteWidth: 8, defaultSpriteHeight: 8, spritesPerRow: 16, spritesPerColumn: 16 
    };
    await spriteManager.loadSpriteSheet(config);

    // (b) Try to fetch existing JSON data.
    const jsonUrl = `/assets/spritesheets/${sheetName}.json`;
    try {
      const data = await fetch(jsonUrl).then(r => r.json());
      spriteData = data;
    } catch {
      // If no JSON is found, initialize with an empty structure.
      spriteData = { name: sheetName, path: config.path, sprites: [], groups: {} };
    }

    // (c) Convert sprite definitions into bounding boxes for the editor.
    boundingBoxes = spriteData.sprites.map(s => ({
      x: s.x, y: s.y,
      width: s.width, height: s.height,
      name: s.name || ''
    }));

    renderSpriteSheet();
  }

  // --- Render the sprite sheet with bounding boxes ---
  function renderSpriteSheet() {
    if (!currentSheetName) return;
    ctx.clearRect(0, 0, editorCanvas.width, editorCanvas.height);

    const sheetObj = spriteManager.getSpriteSheet(currentSheetName);
    if (!sheetObj) return;

    // Draw the sprite sheet image.
    ctx.drawImage(sheetObj.image, 0, 0);

    // Draw bounding boxes.
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    boundingBoxes.forEach(box => {
      ctx.strokeRect(box.x, box.y, box.width, box.height);
      if (box.name) {
        ctx.font = '12px sans-serif';
        ctx.fillStyle = 'yellow';
        ctx.fillText(box.name, box.x + 2, box.y - 4);
      }
    });
  }

  // --- Handle mouse events for drawing bounding boxes ---
  editorCanvas.addEventListener('mousedown', e => {
    isDrawing = true;
    const rect = editorCanvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
  });

  editorCanvas.addEventListener('mousemove', e => {
    if (!isDrawing) return;
    renderSpriteSheet();
    const rect = editorCanvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    ctx.strokeStyle = 'blue';
    ctx.strokeRect(startX, startY, mouseX - startX, mouseY - startY);
  });

  editorCanvas.addEventListener('mouseup', e => {
    if (!isDrawing) return;
    isDrawing = false;
    const rect = editorCanvas.getBoundingClientRect();
    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;
    const width = endX - startX;
    const height = endY - startY;
    const boxName = prompt('Enter sprite name:', `sprite_${boundingBoxes.length}`);
    boundingBoxes.push({ x: startX, y: startY, width, height, name: boxName || '' });
    renderSpriteSheet();
  });

  // --- Save sprite data to server ---
  function saveSpriteData() {
    if (!currentSheetName) return;
    spriteData.sprites = boundingBoxes.map((b, i) => ({
      id: i,
      name: b.name,
      x: b.x, y: b.y,
      width: b.width, height: b.height
    }));
    fetch(`/assets/spritesheets/${currentSheetName}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(spriteData),
    })
      .then(r => r.json())
      .then(resp => {
        if (resp.success) {
          alert('Sprite data saved!');
          // Update the SpriteManager with the new definitions.
          spriteManager.defineSprites(currentSheetName, spriteData);
        } else {
          alert('Failed to save sprite data.');
        }
      })
      .catch(err => {
        console.error('Save error:', err);
        alert('Error saving sprite data.');
      });
  }

  // --- Create a Save button ---
  const saveBtn = document.createElement('button');
  saveBtn.textContent = 'Save Sprites';
  saveBtn.style.display = 'block';
  saveBtn.style.marginTop = '10px';
  saveBtn.addEventListener('click', saveSpriteData);
  editorContainer.appendChild(saveBtn);

  // Load the sprite sheet list to populate the dropdown.
  loadSpriteSheetList();

  // Optionally, add a Close button to dismiss the editor.
  const closeBtn = document.createElement('button');
  closeBtn.textContent = 'Close Sprite Editor';
  closeBtn.style.display = 'block';
  closeBtn.style.marginTop = '10px';
  closeBtn.addEventListener('click', closeSpriteEditor);
  editorContainer.appendChild(closeBtn);
}

export function closeSpriteEditor() {
  const container = document.getElementById('spriteEditorContainer');
  if (container) {
    container.remove();
  }
}
