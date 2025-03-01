/**
 * public/src/screens/groupManager.js
 */

import { spriteManager } from '../assets/spriteManager.js';

const container = document.createElement('div');
container.style.background = '#444';
container.style.padding = '10px';
container.style.margin = '10px';
document.body.appendChild(container);

const title = document.createElement('h2');
title.textContent = 'Group Manager';
container.appendChild(title);

// group listing
const groupList = document.createElement('div');
container.appendChild(groupList);

let currentGroup = null;

function renderGroups() {
  groupList.innerHTML = '';
  Object.keys(spriteManager.groups).forEach(gName => {
    const gDiv = document.createElement('div');
    gDiv.style.cursor = 'pointer';
    gDiv.textContent = `${gName} (${spriteManager.groups[gName].length} sprites)`;
    gDiv.addEventListener('click', () => {
      currentGroup = gName;
      renderSpritesInGroup(gName);
    });
    groupList.appendChild(gDiv);
  });
}

const spriteListDiv = document.createElement('div');
spriteListDiv.style.border = '1px solid #666';
spriteListDiv.style.marginTop = '10px';
spriteListDiv.style.padding = '5px';
container.appendChild(spriteListDiv);

function renderSpritesInGroup(gName) {
  spriteListDiv.innerHTML = `<h4>Sprites in group: ${gName}</h4>`;
  spriteManager.groups[gName].forEach(sk => {
    const item = document.createElement('div');
    item.textContent = sk;
    spriteListDiv.appendChild(item);
  });
}

/**
 * Add new group
 */
const newGroupBtn = document.createElement('button');
newGroupBtn.textContent = 'New Group';
newGroupBtn.addEventListener('click', () => {
  const gName = prompt('Enter new group name:');
  if (!gName) return;
  spriteManager.groups[gName] = [];
  renderGroups();
});
container.appendChild(newGroupBtn);

/**
 * Add a sprite to current group
 */
const addSpriteInput = document.createElement('input');
addSpriteInput.placeholder = 'sheetName_spriteName';
addSpriteInput.style.marginLeft = '10px';
container.appendChild(addSpriteInput);

const addSpriteBtn = document.createElement('button');
addSpriteBtn.textContent = 'Add Sprite';
addSpriteBtn.style.marginLeft = '5px';
addSpriteBtn.addEventListener('click', () => {
  if (!currentGroup) {
    alert('Select or create a group first!');
    return;
  }
  const sk = addSpriteInput.value.trim();
  if (!spriteManager.getSprite(sk)) {
    alert(`Sprite key "${sk}" not found in manager.`);
    return;
  }
  if (!spriteManager.groups[currentGroup].includes(sk)) {
    spriteManager.groups[currentGroup].push(sk);
  }
  addSpriteInput.value = '';
  renderGroups();
  renderSpritesInGroup(currentGroup);
});
container.appendChild(addSpriteBtn);

/**
 * Save all group data to "globalGroups" on the server
 * (adjust to your own approach if you want per-sheet storage)
 */
const saveGroupsBtn = document.createElement('button');
saveGroupsBtn.textContent = 'Save Groups';
saveGroupsBtn.style.display = 'block';
saveGroupsBtn.style.marginTop = '10px';
saveGroupsBtn.addEventListener('click', () => {
  const data = { groups: spriteManager.groups };
  fetch('/assets/spritesheets/globalGroups', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
    .then(r => r.json())
    .then(resp => {
      if (resp.success) {
        alert('Groups saved successfully!');
      } else {
        alert('Failed to save groups.');
      }
    })
    .catch(err => {
      console.error(err);
      alert('Error saving groups data.');
    });
});
container.appendChild(saveGroupsBtn);

renderGroups();
