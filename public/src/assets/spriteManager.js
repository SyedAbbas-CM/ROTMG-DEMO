export class SpriteManager {
  constructor() {
    // Loaded sprite sheets: { sheetName: { image, config } }
    this.spriteSheets = {};
    // Global sprite definitions: keys as "sheetName_spriteName"
    this.sprites = {};
    // Groups (for animations, etc.)
    this.groups = {};
  }

  /**
   * Loads a sprite sheet from a configuration object.
   * The config must include: name, path, defaultSpriteWidth, defaultSpriteHeight, spritesPerRow, spritesPerColumn.
   * Optionally, config.sprites can be provided.
   */
  async loadSpriteSheet(config) {
    const image = await this._loadImage(config.path);
    this.spriteSheets[config.name] = { image, config };

    // Auto-extract if no sprites provided.
    if (!config.sprites || config.sprites.length === 0) {
      config.sprites = this.autoExtractSprites(image, config);
    }

    this.defineSprites(config.name, config);
    return this.spriteSheets[config.name];
  }

  autoExtractSprites(image, config) {
    const sprites = [];
    const { defaultSpriteWidth, defaultSpriteHeight, spritesPerRow, spritesPerColumn } = config;
    let id = 0;
    for (let row = 0; row < spritesPerColumn; row++) {
      for (let col = 0; col < spritesPerRow; col++) {
        sprites.push({
          id: id,
          name: `sprite_${id}`,
          x: col * defaultSpriteWidth,
          y: row * defaultSpriteHeight,
          width: defaultSpriteWidth,
          height: defaultSpriteHeight
        });
        id++;
      }
    }
    return sprites;
  }

  _loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.src = url;
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`Failed to load image at ${url}`));
    });
  }

  /**
   * Saves sprite definitions and groups into the manager.
   */
  defineSprites(sheetName, spriteData) {
    const sheetObj = this.spriteSheets[sheetName];
    if (!sheetObj) {
      console.warn(`No sprite sheet found with name "${sheetName}".`);
      return;
    }
    // Update the configuration
    sheetObj.config = spriteData;

    // Remove previous definitions for this sheet
    Object.keys(this.sprites).forEach(key => {
      if (key.startsWith(`${sheetName}_`)) {
        delete this.sprites[key];
      }
    });

    // Define each sprite using key format "sheetName_spriteName"
    spriteData.sprites.forEach(sprite => {
      const key = `${sheetName}_${sprite.name || `sprite_${sprite.id}`}`;
      this.sprites[key] = {
        sheetName: sheetName,
        x: sprite.x,
        y: sprite.y,
        width: sprite.width,
        height: sprite.height
      };
    });

    // Process groups if provided
    Object.keys(spriteData.groups || {}).forEach(groupName => {
      if (!this.groups[groupName]) {
        this.groups[groupName] = [];
      }
      spriteData.groups[groupName].forEach(idOrName => {
        const found = spriteData.sprites.find(
          spr => spr.id === idOrName || spr.name === idOrName
        );
        if (found) {
          const key = `${sheetName}_${found.name || `sprite_${found.id}`}`;
          if (!this.groups[groupName].includes(key)) {
            this.groups[groupName].push(key);
          }
        }
      });
    });
  }

  /**
   * Returns the sprite definition by its key.
   * Falls back to the first sprite (8x8) of the first sheet if not found.
   */
  getSprite(spriteKey) {
    if (this.sprites[spriteKey]) {
      return this.sprites[spriteKey];
    }
    // Fallback: use the first loaded sprite sheet's first sprite (8x8)
    const sheetNames = Object.keys(this.spriteSheets);
    if (sheetNames.length > 0) {
      const firstSheet = this.spriteSheets[sheetNames[0]];
      if (!firstSheet.config.sprites || firstSheet.config.sprites.length === 0) {
        const defaultSprite = { id: 0, name: 'sprite_0', x: 0, y: 0, width: 8, height: 8 };
        firstSheet.config.sprites = [defaultSprite];
        this.sprites[`${sheetNames[0]}_sprite_0`] = { sheetName: sheetNames[0], ...defaultSprite };
      }
      return this.sprites[`${sheetNames[0]}_sprite_0`];
    }
    return null;
  }

  /**
   * Returns the sprite sheet object (with image and config) by name.
   */
  getSpriteSheet(sheetName) {
    return this.spriteSheets[sheetName] || null;
  }

  /**
   * Returns an array of sprite keys for a given group.
   */
  getGroupSprites(groupName) {
    return this.groups[groupName] || [];
  }

  /**
   * For a given sprite key, returns the associated image.
   */
  getSheetImage(spriteKey) {
    const spr = this.getSprite(spriteKey);
    if (!spr) return null;
    const sheet = this.getSpriteSheet(spr.sheetName);
    return sheet ? sheet.image : null;
  }
}

export const spriteManager = new SpriteManager();
