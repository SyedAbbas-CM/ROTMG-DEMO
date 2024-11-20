// src/assets/spriteManager.js

export class SpriteManager {
    constructor() {
      this.spriteSheets = {};
      this.sprites = {};
    }
  
    async loadSpriteSheets(configPath) {
      const response = await fetch(configPath);
      const spriteSheetsConfig = await response.json();
  
      const loadPromises = spriteSheetsConfig.map(config => this.loadSpriteSheet(config));
  
      await Promise.all(loadPromises);
      console.log('All sprite sheets loaded.');
    }
  
    async loadSpriteSheet(config) {
      return new Promise((resolve, reject) => {
        const image = new Image();
        image.src = config.path;
        image.onload = () => {
          this.spriteSheets[config.name] = { image, config };
          this.extractSprites(config.name);
          resolve();
        };
        image.onerror = () => reject(new Error(`Failed to load ${config.path}`));
      });
    }
  
    extractSprites(sheetName) {
      const { image, config } = this.spriteSheets[sheetName];
      const { defaultSpriteWidth, defaultSpriteHeight, spritesPerRow, spritesPerColumn } = config;
  
      if (config.sprites && config.sprites.length > 0) {
        // Process irregular sprites
        config.sprites.forEach(spriteConfig => {
          const key = `${sheetName}_${spriteConfig.index}`;
          this.sprites[key] = {
            sheetName,
            x: spriteConfig.x,
            y: spriteConfig.y,
            width: spriteConfig.width || defaultSpriteWidth,
            height: spriteConfig.height || defaultSpriteHeight,
          };
        });
      } else {
        // Process uniform grid
        let index = 0;
        for (let row = 0; row < spritesPerColumn; row++) {
          for (let col = 0; col < spritesPerRow; col++) {
            const key = `${sheetName}_${index}`;
            this.sprites[key] = {
              sheetName,
              x: col * defaultSpriteWidth,
              y: row * defaultSpriteHeight,
              width: defaultSpriteWidth,
              height: defaultSpriteHeight,
            };
            index++;
          }
        }
      }
    }
  
    getSprite(spriteKey) {
      return this.sprites[spriteKey];
    }
  
    getSpriteSheet(sheetName) {
      return this.spriteSheets[sheetName];
    }
  }
  