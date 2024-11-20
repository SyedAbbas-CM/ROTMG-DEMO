// src/assets/assets.js

export const assets = {};

export function loadAssets() {
  return new Promise((resolve, reject) => {
    let assetsLoaded = 0;
    const totalAssets = 4; // Number of sprite sheets to load

    function checkAllAssetsLoaded() {
      assetsLoaded++;
      if (assetsLoaded === totalAssets) resolve();
    }

    // Load character sprite sheet
    const characterSpriteSheet = new Image();
    characterSpriteSheet.src = 'assets/images/Oryx/lofi_char.png';
    characterSpriteSheet.onload = () => {
      assets.characterSpriteSheet = characterSpriteSheet;
      checkAllAssetsLoaded();
    };
    characterSpriteSheet.onerror = () => reject(new Error('Failed to load 8-Bit_Remaster_Character.png'));

    // Load enemy sprite sheet
    const enemySpriteSheet = new Image();
    enemySpriteSheet.src = 'assets/images/Oryx/8-Bit_Remaster_Character.png'; // Ensure this path is correct
    enemySpriteSheet.onload = () => {
      assets.enemySpriteSheet = enemySpriteSheet;
      checkAllAssetsLoaded();
    };
    enemySpriteSheet.onerror = () => reject(new Error('Failed to load 8-Bit_Remaster_Enemy.png'));

    // Load tile sprite sheet
    const tileSpriteSheet = new Image();
    tileSpriteSheet.src = 'assets/images/Oryx/8-Bit_Remaster_World.png';
    tileSpriteSheet.onload = () => {
      assets.tileSpriteSheet = tileSpriteSheet;
      checkAllAssetsLoaded();
    };
    tileSpriteSheet.onerror = () => reject(new Error('Failed to load 8-Bit_Remaster_World.png'));

    // Load wall sprite sheet
    const wallSpriteSheet = new Image();
    wallSpriteSheet.src = 'assets/images/Oryx/8-Bit_Remaster_World.png'; // Ensure this path is correct
    wallSpriteSheet.onload = () => {
      assets.wallSpriteSheet = wallSpriteSheet;
      checkAllAssetsLoaded();
    };
    wallSpriteSheet.onerror = () => reject(new Error('Failed to load 8-Bit_Remaster_Walls.png'));
  });
}
