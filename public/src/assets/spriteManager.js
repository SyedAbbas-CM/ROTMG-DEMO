export class SpriteManager {
  constructor() {
    // Loaded sprite sheets: { sheetName: { image, config } }
    this.spriteSheets = {};
    // Global sprite definitions: keys as "sheetName_spriteName"
    this.sprites = {};
    // Groups (for animations, etc.)
    this.groups = {};
    // Map of alias -> { sheetName, spriteName }
    this.aliases = {};
  }

  /**
   * Loads a sprite sheet from a configuration object.
   * The config must include: name, path, and optionally defaultSpriteWidth and defaultSpriteHeight.
   * If width/height are not specified, auto-detection will be used.
   */
  async loadSpriteSheet(config) {
    // Create a default config with sensible defaults
    const defaultConfig = {
      defaultSpriteWidth: 8,
      defaultSpriteHeight: 8,
      autoDetect: true
    };

    // Merge provided config with defaults
    const mergedConfig = { ...defaultConfig, ...config };
    
    try {
      // Load the image
      const image = await this._loadImage(mergedConfig.path);
      this.spriteSheets[mergedConfig.name] = { image, config: mergedConfig };
      
      // Process sprites based on config format
      if (mergedConfig.sprites && mergedConfig.sprites.length > 0) {
        // Convert row/col format to x/y format if needed
        mergedConfig.sprites = mergedConfig.sprites.map((sprite, index) => {
          if (sprite.row !== undefined && sprite.col !== undefined) {
            return {
              id: sprite.id || index,
              name: sprite.name || `sprite_${index}`,
              x: sprite.col * mergedConfig.defaultSpriteWidth,
              y: sprite.row * mergedConfig.defaultSpriteHeight,
              width: sprite.width || mergedConfig.defaultSpriteWidth,
              height: sprite.height || mergedConfig.defaultSpriteHeight
            };
          }
          return sprite;
        });
      }
      // Extract sprites automatically if not provided
      else if (mergedConfig.autoDetect) {
        mergedConfig.sprites = await this.autoDetectSprites(image, mergedConfig);
      } 
      // Fall back to grid-based extraction if sprites not provided and autoDetect is false
      else if (!mergedConfig.sprites || mergedConfig.sprites.length === 0) {
        if (!mergedConfig.spritesPerRow || !mergedConfig.spritesPerColumn) {
          // If grid dimensions not specified, estimate them based on image size
          mergedConfig.spritesPerRow = Math.floor(image.width / mergedConfig.defaultSpriteWidth);
          mergedConfig.spritesPerColumn = Math.floor(image.height / mergedConfig.defaultSpriteHeight);
        }
        mergedConfig.sprites = this.autoExtractSprites(image, mergedConfig);
      }
      
      // Define sprites in our manager
      this.defineSprites(mergedConfig.name, mergedConfig);
      console.log(`Loaded sprite sheet "${mergedConfig.name}" with ${mergedConfig.sprites.length} sprites`);
      
      return this.spriteSheets[mergedConfig.name];
    } catch (error) {
      console.error(`Failed to load sprite sheet "${mergedConfig.name}":`, error);
      throw error;
    }
  }

  /**
   * Auto-detect sprites by scanning the image for non-transparent regions.
   * This uses an HTML canvas to analyze the image data.
   */
  async autoDetectSprites(image, config) {
    return new Promise((resolve) => {
      // Create a canvas to analyze the image
      const canvas = document.createElement('canvas');
      canvas.width = image.width;
      canvas.height = image.height;
      
      const ctx = canvas.getContext('2d');
      ctx.drawImage(image, 0, 0);
      
      // Get the image data for analysis
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      
      // Step 1: Create an alpha map (true for non-transparent pixels)
      const alphaMap = [];
      for (let y = 0; y < canvas.height; y++) {
        alphaMap[y] = [];
        for (let x = 0; x < canvas.width; x++) {
          const pixelIndex = (y * canvas.width + x) * 4;
          // A pixel is non-transparent if its alpha value is greater than 0
          alphaMap[y][x] = data[pixelIndex + 3] > 0;
        }
      }
      
      // Step 2: Use a modified flood-fill algorithm to find contiguous sprite regions
      const visited = Array(canvas.height).fill().map(() => Array(canvas.width).fill(false));
      const sprites = [];
      let id = 0;
      
      // Helper function for flood fill
      const floodFill = (startX, startY) => {
        if (!alphaMap[startY][startX] || visited[startY][startX]) return null;
        
        // Use a queue-based flood fill
        const queue = [{x: startX, y: startY}];
        visited[startY][startX] = true;
        
        // Track bounds of the contiguous region
        let minX = startX, maxX = startX, minY = startY, maxY = startY;
        
        while (queue.length > 0) {
          const {x, y} = queue.shift();
          
          // Check all 4 adjacent pixels
          const neighbors = [
            {x: x+1, y: y}, {x: x-1, y: y}, 
            {x: x, y: y+1}, {x: x, y: y-1}
          ];
          
          for (const neighbor of neighbors) {
            const nx = neighbor.x;
            const ny = neighbor.y;
            
            // Skip out-of-bounds pixels
            if (nx < 0 || nx >= canvas.width || ny < 0 || ny >= canvas.height) continue;
            
            // Skip transparent or already visited pixels
            if (!alphaMap[ny][nx] || visited[ny][nx]) continue;
            
            // Mark as visited and add to queue
            visited[ny][nx] = true;
            queue.push({x: nx, y: ny});
            
            // Update bounding box
            minX = Math.min(minX, nx);
            maxX = Math.max(maxX, nx);
            minY = Math.min(minY, ny);
            maxY = Math.max(maxY, ny);
          }
        }
        
        // Return the sprite definition
        return {
          id,
          name: `sprite_${id}`,
          x: minX,
          y: minY,
          width: maxX - minX + 1,
          height: maxY - minY + 1
        };
      };
      
      // Scan the image and find sprites
      for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
          const sprite = floodFill(x, y);
          if (sprite) {
            sprites.push(sprite);
            id++;
          }
        }
      }
      
      // Filter out sprites that are too small (likely noise)
      const minSize = 3; // Minimum dimension in pixels
      const filteredSprites = sprites.filter(sprite => 
        sprite.width >= minSize && sprite.height >= minSize);
      
      resolve(filteredSprites.length > 0 ? filteredSprites : this.fallbackToGridExtraction(image, config));
    });
  }
  
  /**
   * Fall back to grid-based extraction if auto-detection finds no sprites.
   */
  fallbackToGridExtraction(image, config) {
    console.warn(`Auto-detection found no sprites for "${config.name}", falling back to grid extraction`);
    // Determine grid dimensions if not provided
    if (!config.spritesPerRow || !config.spritesPerColumn) {
      config.spritesPerRow = Math.floor(image.width / config.defaultSpriteWidth);
      config.spritesPerColumn = Math.floor(image.height / config.defaultSpriteHeight);
    }
    return this.autoExtractSprites(image, config);
  }

  /**
   * Extract sprites in a grid pattern based on the provided config.
   */
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
    const defaultWidth = spriteData.defaultSpriteWidth || 8;
    const defaultHeight = spriteData.defaultSpriteHeight || 8;

    spriteData.sprites.forEach(sprite => {
      const spriteName = sprite.name || `sprite_${sprite.id}`;
      const key = `${sheetName}_${spriteName}`;

      // Compute x/y from row/col if not provided directly
      let x = sprite.x;
      let y = sprite.y;
      if (x === undefined && sprite.col !== undefined) {
        x = sprite.col * (sprite.width || defaultWidth);
      }
      if (y === undefined && sprite.row !== undefined) {
        y = sprite.row * (sprite.height || defaultHeight);
      }

      this.sprites[key] = {
        sheetName: sheetName,
        x: x || 0,
        y: y || 0,
        width: sprite.width || defaultWidth,
        height: sprite.height || defaultHeight
      };

      // Auto-register the bare sprite name as an alias if it is unique across the project.
      // This allows code to simply request fetchSprite('knight') without worrying about
      // which sheet it lives on, provided there are no name collisions.
      if (!this.aliases[spriteName]) {
        this.aliases[spriteName] = { sheetName, spriteName };
      }
    });

    // Process groups if provided
    if (spriteData.groups) {
      Object.keys(spriteData.groups).forEach(groupName => {
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
    const sprite = this.getSprite(spriteKey);
    if (!sprite) return null;
    return this.spriteSheets[sprite.sheetName]?.image || null;
  }
  
  /**
   * Creates sprite animation groups based on provided criteria.
   * @param {string} sheetName - Name of the sprite sheet
   * @param {string} groupName - Name for the new group
   * @param {Object} options - Grouping options
   */
  createSpriteGroup(sheetName, groupName, options = {}) {
    const sheet = this.getSpriteSheet(sheetName);
    if (!sheet) {
      console.warn(`Cannot create group: Sheet "${sheetName}" not found`);
      return;
    }
    
    // Filter sprites based on options
    const matchingSprites = sheet.config.sprites.filter(sprite => {
      let match = true;
      
      // Match by size
      if (options.width && sprite.width !== options.width) match = false;
      if (options.height && sprite.height !== options.height) match = false;
      
      // Match by position
      if (options.row !== undefined) {
        const spriteRow = Math.floor(sprite.y / sheet.config.defaultSpriteHeight);
        if (spriteRow !== options.row) match = false;
      }
      
      if (options.column !== undefined) {
        const spriteCol = Math.floor(sprite.x / sheet.config.defaultSpriteWidth);
        if (spriteCol !== options.column) match = false;
      }
      
      // Custom matcher function
      if (options.matcher && typeof options.matcher === 'function') {
        if (!options.matcher(sprite)) match = false;
      }
      
      return match;
    });
    
    // Create or update the group
    this.groups[groupName] = matchingSprites.map(sprite => 
      `${sheetName}_${sprite.name || `sprite_${sprite.id}`}`
    );
    
    return this.groups[groupName];
  }

  /**
   * Draw a sprite from a sprite sheet at the given position
   * @param {CanvasRenderingContext2D} ctx - The canvas context to draw on
   * @param {string} sheetName - Name of the sprite sheet
   * @param {number} spriteX - X position of the sprite in the sheet
   * @param {number} spriteY - Y position of the sprite in the sheet
   * @param {number} destX - X position to draw at
   * @param {number} destY - Y position to draw at
   * @param {number} destWidth - Width to draw (will scale)
   * @param {number} destHeight - Height to draw (will scale)
   * @param {number} [sourceWidth] - Optional width of source sprite (defaults to destWidth)
   * @param {number} [sourceHeight] - Optional height of source sprite (defaults to destHeight)
   */
  drawSprite(ctx, sheetName, spriteX, spriteY, destX, destY, destWidth, destHeight, sourceWidth, sourceHeight) {
    // Get the sprite sheet
    const sheet = this.getSpriteSheet(sheetName);
    if (!sheet || !sheet.image) {
      console.warn(`Cannot draw sprite: Sheet "${sheetName}" not found or image not loaded`);
      return;
    }

    // Handle default source dimensions
    sourceWidth = sourceWidth || sheet.config.defaultSpriteWidth || destWidth;
    sourceHeight = sourceHeight || sheet.config.defaultSpriteHeight || destHeight;

    try {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(
        sheet.image,
        spriteX, spriteY,
        sourceWidth, sourceHeight,
        destX, destY,
        destWidth, destHeight
      );
    } catch (error) {
      console.error(`Error drawing sprite from ${sheetName} at (${spriteX}, ${spriteY}):`, error);
      
      // Fallback: draw a colored rectangle for debugging
      ctx.fillStyle = 'magenta';
      ctx.fillRect(destX, destY, destWidth, destHeight);
    }
  }

  /**
   * Convenience helper – compute a sprite region by grid coordinates rather
   * than relying on the auto-generated sprite registry. This makes it trivial
   * to address sprites in classic tile sheets where every tile is the same
   * size (e.g. 8×8, 10×10, etc.) but the sheet as a whole might use any cell
   * size and number of rows/columns.
   *
   * It returns an object shaped like those coming from `getSprite`, allowing
   * it to be passed straight into the existing `drawSprite` helpers.
   *
   *   const region = spriteManager.getGridSprite('enemy_sprites', 3, 5);
   *   ctx.drawImage(region.image, region.x, region.y, region.width, region.height, ...);
   *
   * If `spriteWidth` / `spriteHeight` are omitted the sheet's default
   * dimensions are used (those provided in the `loadSpriteSheet` config).
   */
  getGridSprite(sheetName, row, col, spriteWidth = null, spriteHeight = null) {
    const sheetObj = this.spriteSheets[sheetName];
    if (!sheetObj) {
      console.warn(`SpriteManager.getGridSprite: Sheet "${sheetName}" not loaded yet.`);
      return null;
    }

    const { image, config } = sheetObj;
    const w = spriteWidth  || config.defaultSpriteWidth;
    const h = spriteHeight || config.defaultSpriteHeight;

    return {
      sheetName,
      image,
      name: `r${row}c${col}`,
      x: col * w,
      y: row * h,
      width: w,
      height: h
    };
  }

  /**
   * Register a human-friendly alias for a sprite definition that already exists.
   * Example: spriteManager.registerAlias('orc', 'chars2', 'sprite_15');
   */
  registerAlias(alias, sheetName, spriteName) {
    const key = `${sheetName}_${spriteName}`;
    if (!this.sprites[key]) {
      console.warn(`registerAlias: sprite ${key} not found`);
      return false;
    }
    this.aliases[alias] = { sheetName, spriteName };
    return true;
  }

  /**
   * Retrieve a sprite via alias or direct key.
   * This lets game code request sprites by simple names like 'orc'.
   */
  fetchSprite(name) {
    // First check alias table
    if (this.aliases[name]) {
      const { sheetName, spriteName } = this.aliases[name];
      return this.getSprite(`${sheetName}_${spriteName}`);
    }
    // Fall back to direct key lookup (sheet_sprite)
    return this.getSprite(name);
  }

  /**
   * Convenience helper: ensure a grid-based sprite exists and return it.
   * This lets designers call spriteManager.fetchGridSprite('chars2', row, col, alias)
   */
  fetchGridSprite(sheetName, row, col, alias = null, spriteWidth = null, spriteHeight = null) {
    const gridSprite = this.getGridSprite(sheetName, row, col, spriteWidth, spriteHeight);
    // Ensure this grid sprite is stored in the main registry so later look-ups succeed
    if (gridSprite) {
      const key = `${sheetName}_${gridSprite.name}`;
      if (!this.sprites[key]) {
        this.sprites[key] = gridSprite;
      }
    }
    if (alias) {
      this.registerAlias(alias, sheetName, gridSprite.name);
    }
    return gridSprite;
  }
}

export const spriteManager = new SpriteManager();