/**
 * public/src/screens/settingsScreen.js
 */

export const defaultSettings = {
    gridSize: 32,
    autosave: false,
    assetPath: '/assets/spritesheets'
  };
  
  export function initSettingsScreen() {
    const container = document.createElement('div');
    container.id = 'settingsContainer';
    container.style.position = 'fixed';
    container.style.top = '10px';
    container.style.right = '10px';
    container.style.background = '#333';
    container.style.color = '#fff';
    container.style.padding = '10px';
    container.style.zIndex = '9999';
  
    const heading = document.createElement('h3');
    heading.textContent = 'Settings';
    container.appendChild(heading);
  
    // Grid Size
    const gridLabel = document.createElement('label');
    gridLabel.textContent = 'Grid Size: ';
    const gridInput = document.createElement('input');
    gridInput.type = 'number';
    gridInput.value = defaultSettings.gridSize.toString();
    gridInput.addEventListener('change', () => {
      defaultSettings.gridSize = parseInt(gridInput.value, 10);
    });
    gridLabel.appendChild(gridInput);
    container.appendChild(gridLabel);
    container.appendChild(document.createElement('br'));
  
    // Autosave
    const autosaveLabel = document.createElement('label');
    autosaveLabel.textContent = 'Autosave: ';
    const autosaveChk = document.createElement('input');
    autosaveChk.type = 'checkbox';
    autosaveChk.checked = defaultSettings.autosave;
    autosaveChk.addEventListener('change', () => {
      defaultSettings.autosave = autosaveChk.checked;
    });
    autosaveLabel.appendChild(autosaveChk);
    container.appendChild(autosaveLabel);
    container.appendChild(document.createElement('br'));
  
    // Asset Path
    const pathLabel = document.createElement('label');
    pathLabel.textContent = 'Asset Path: ';
    const pathInput = document.createElement('input');
    pathInput.type = 'text';
    pathInput.value = defaultSettings.assetPath;
    pathInput.addEventListener('change', () => {
      defaultSettings.assetPath = pathInput.value;
    });
    pathLabel.appendChild(pathInput);
    container.appendChild(pathLabel);
    container.appendChild(document.createElement('br'));
  
    // Close
    const closeBtn = document.createElement('button');
    closeBtn.textContent = 'Close Settings';
    closeBtn.addEventListener('click', () => {
      document.body.removeChild(container);
    });
    container.appendChild(closeBtn);
  
    document.body.appendChild(container);
  }
  