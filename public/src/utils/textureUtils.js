// src/utils/textureUtils.js

/**
 * Creates a material with a specific sprite from a loaded sprite sheet.
 * @param {Image} image - The loaded Image object of the sprite sheet.
 * @param {Object} spritePosition - { x, y } position of the sprite in pixels.
 * @param {number} tileSize - Size of the tile in pixels.
 * @returns {THREE.MeshStandardMaterial} - The created material.
 */
export function createSpriteMaterial(image, spritePosition, tileSize) {
  if (!image || !image.width || !image.height) {
    console.error('Invalid image provided to createSpriteMaterial.');
    return new THREE.MeshStandardMaterial();
  }

  const texture = new THREE.Texture(image);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  
  const tilesPerRow = image.width / tileSize;
  const tilesPerColumn = image.height / tileSize;
  
  texture.repeat.set(1 / tilesPerRow, 1 / tilesPerColumn);
  
  texture.offset.set(
    spritePosition.x / image.width,
    1 - (spritePosition.y + tileSize) / image.height
  ); // Flip Y axis
  
  texture.needsUpdate = true;

  return new THREE.MeshStandardMaterial({ map: texture, transparent: true });
}

/**
 * Gets the UV offset for a specific tile type in the texture atlas.
 * @param {number} tileType - The type of tile (e.g., FLOOR, WALL).
 * @returns {Object} - UV offset for the tile in the texture atlas.
 */
export function getUVOffsetForTileType(tileType) {
  const spritePos = TILE_SPRITES[tileType];
  const uvOffset = {
    u: spritePos.x / assets.tileSpriteSheet.width,
    v: spritePos.y / assets.tileSpriteSheet.height,
  };
  return uvOffset;
}