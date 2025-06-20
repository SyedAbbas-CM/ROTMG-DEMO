/**
 * SpriteDatabase - Optimized sprite lookup system
 * 
 * Features:
 * - Fast O(1) lookups by name
 * - Memory-efficient storage (no duplicate data)
 * - Precomputed coordinates for grid-based sprites
 * - Batch atlas loading
 * - Runtime sprite validation
 */
export class SpriteDatabase {
  constructor() {
    // Core data structures
    this.atlases = new Map();           // atlasName -> atlas config
    this.sprites = new Map();           // spriteName -> sprite data
    this.images = new Map();            // atlasName -> HTMLImageElement
    this.groups = new Map();            // groupName -> Set of sprite names
    
    // Performance optimizations
    this.spriteCache = new Map();       // spriteName -> cached sprite object
    this.coordinateCache = new Map();   // "atlas_row_col" -> {x, y, w, h}
    
    // Statistics
    this.stats = {
      atlasesLoaded: 0,
      spritesRegistered: 0,
      cacheHits: 0,
      cacheMisses: 0
    };

    this.entities = new Map();
  }

  /**
   * Load multiple atlases in parallel
   * @param {Array<string>} atlasPaths - Array of atlas JSON file paths
   * @returns {Promise<void>}
   */
  async loadAtlases(atlasPaths) {
    console.log(`[SpriteDB] Attempting to load ${atlasPaths.length} sprite atlases...`);
    const startTime = performance.now();

    // ---- Fetch atlas JSON files (tolerate failures) ----
    const fetchPromises = atlasPaths.map(path =>
      fetch(path)
        .then(r => {
          if (!r.ok) throw new Error(`HTTP ${r.status} while fetching ${path}`);
          return r.json();
        })
    );

    const fetchResults = await Promise.allSettled(fetchPromises);

    const successfulAtlasConfigs = [];
    fetchResults.forEach((res, idx) => {
      if (res.status === 'fulfilled') {
        successfulAtlasConfigs.push(res.value);
      } else {
        console.warn(`[SpriteDB] Skipping atlas '${atlasPaths[idx]}' – fetch failed:`, res.reason?.message || res.reason);
      }
    });

    if (successfulAtlasConfigs.length === 0) {
      console.warn('[SpriteDB] No atlases could be loaded. The game may render without sprites.');
      return;
    }

    // ---- Load corresponding images (also tolerate failures) ----
    const imagePromises = successfulAtlasConfigs.map(cfg => {
      // Determine image path property (editor may save as meta.image)
      const imgPath = cfg.path || cfg.image || (cfg.meta && cfg.meta.image);
      if (!imgPath) {
        console.warn('[SpriteDB] Atlas config missing image path:', cfg.name || '[unnamed]');
        return Promise.resolve(null);
      }
      return this._loadImage(imgPath).catch(err => {
        console.warn(`[SpriteDB] Failed to load image '${imgPath}':`, err);
        return null;
      });
    });

    const imageResults = await Promise.all(imagePromises);

    // ---- Register only successful atlas-image pairs ----
    for (let i = 0; i < successfulAtlasConfigs.length; i++) {
      const cfg = successfulAtlasConfigs[i];
      const img = imageResults[i];
      if (!img) {
        console.warn(`[SpriteDB] Image missing for atlas '${cfg?.name || 'unknown'}' – skipping registration.`);
        continue;
      }
      this.registerAtlas(cfg, img);
    }

    const loadTime = performance.now() - startTime;
    console.log(`[SpriteDB] ✅ Finished loading. Atlases: ${this.stats.atlasesLoaded}, Sprites: ${this.stats.spritesRegistered}. (${loadTime.toFixed(2)}ms)`);
  }

  /**
   * Register a single atlas with its image
   * @param {Object} config - Atlas configuration
   * @param {HTMLImageElement} image - Loaded image
   */
  registerAtlas(config, image) {
    let { name, sprites = [], groups = {} } = config;
    if (!name) {
      // Derive name from image filename or generic index
      const imgPath = config.path || config.image || (config.meta && config.meta.image) || 'atlas_'+this.stats.atlasesLoaded;
      name = imgPath.split('/').pop().replace(/\.(png|jpg|jpeg|gif)$/i,'');
    }
    config.name = name;
    
    // Ensure default sprite dimensions present
    if (!('defaultSpriteWidth' in config) || !('defaultSpriteHeight' in config)) {
      const first = (config.sprites && config.sprites[0]) || {};
      if (!('defaultSpriteWidth' in config)) config.defaultSpriteWidth = first.width || 8;
      if (!('defaultSpriteHeight' in config)) config.defaultSpriteHeight = first.height || 8;
    }

    // Store atlas config and image
    this.atlases.set(name, config);
    this.images.set(name, image);
    
    // Register all sprites from this atlas
    sprites.forEach(spriteConfig => {
      this.registerSprite(name, spriteConfig, config);
    });
    
    // Register groups
    Object.entries(groups).forEach(([groupName, spriteNames]) => {
      if (!this.groups.has(groupName)) {
        this.groups.set(groupName, new Set());
      }
      const group = this.groups.get(groupName);
      spriteNames.forEach(spriteName => group.add(spriteName));
    });
    
    this.stats.atlasesLoaded++;
    console.log(`Registered atlas "${name}" with ${sprites.length} sprites`);
  }

  /**
   * Register a single sprite
   * @param {string} atlasName - Name of the atlas
   * @param {Object} spriteConfig - Sprite configuration
   * @param {Object} atlasConfig - Atlas configuration
   */
  registerSprite(atlasName, spriteConfig, atlasConfig) {
    const { name, row, col, x, y, width, height } = spriteConfig;
    const { defaultSpriteWidth, defaultSpriteHeight } = atlasConfig;
    
    // Calculate coordinates (support both row/col and x/y formats)
    let spriteX, spriteY, spriteW, spriteH;
    
    if (row !== undefined && col !== undefined) {
      // Grid-based coordinates
      spriteX = col * defaultSpriteWidth;
      spriteY = row * defaultSpriteHeight;
      spriteW = width || defaultSpriteWidth;
      spriteH = height || defaultSpriteHeight;
    } else {
      // Absolute coordinates
      spriteX = x || 0;
      spriteY = y || 0;
      spriteW = width || defaultSpriteWidth;
      spriteH = height || defaultSpriteHeight;
    }
    
    // Store sprite data
    const spriteData = {
      name,
      atlasName,
      x: spriteX,
      y: spriteY,
      width: spriteW,
      height: spriteH,
      row: row,
      col: col
    };
    
    this.sprites.set(name, spriteData);
    this.stats.spritesRegistered++;
  }

  /**
   * Get sprite data by name (optimized lookup)
   * @param {string} spriteName - Name of the sprite
   * @returns {Object|null} Sprite data or null if not found
   */
  getSprite(spriteName) {
    // Check cache first
    if (this.spriteCache.has(spriteName)) {
      this.stats.cacheHits++;
      return this.spriteCache.get(spriteName);
    }
    
    // Look up sprite data
    const spriteData = this.sprites.get(spriteName);
    if (!spriteData) {
      this.stats.cacheMisses++;
      return null;
    }
    
    // Get the image
    const image = this.images.get(spriteData.atlasName);
    if (!image) {
      console.warn(`Image not loaded for atlas: ${spriteData.atlasName}`);
      return null;
    }
    
    // Create sprite object
    const sprite = {
      name: spriteData.name,
      atlasName: spriteData.atlasName,
      image: image,
      x: spriteData.x,
      y: spriteData.y,
      width: spriteData.width,
      height: spriteData.height,
      row: spriteData.row,
      col: spriteData.col
    };
    
    // Cache the result
    this.spriteCache.set(spriteName, sprite);
    this.stats.cacheMisses++;
    
    return sprite;
  }

  /**
   * Get sprite by grid coordinates (for dynamic access)
   * @param {string} atlasName - Name of the atlas
   * @param {number} row - Row index
   * @param {number} col - Column index
   * @returns {Object|null} Sprite data or null if not found
   */
  getSpriteByGrid(atlasName, row, col) {
    const cacheKey = `${atlasName}_${row}_${col}`;
    
    // Check coordinate cache
    if (this.coordinateCache.has(cacheKey)) {
      const coords = this.coordinateCache.get(cacheKey);
      const image = this.images.get(atlasName);
      
      return {
        name: `${atlasName}_r${row}c${col}`,
        atlasName,
        image,
        row,
        col,
        ...coords
      };
    }
    
    // Calculate coordinates
    const atlas = this.atlases.get(atlasName);
    if (!atlas) return null;
    
    const { defaultSpriteWidth, defaultSpriteHeight } = atlas;
    const coords = {
      x: col * defaultSpriteWidth,
      y: row * defaultSpriteHeight,
      width: defaultSpriteWidth,
      height: defaultSpriteHeight
    };
    
    // Cache coordinates
    this.coordinateCache.set(cacheKey, coords);
    
    const image = this.images.get(atlasName);
    return {
      name: `${atlasName}_r${row}c${col}`,
      atlasName,
      image,
      row,
      col,
      ...coords
    };
  }

  /**
   * Get all sprites in a group
   * @param {string} groupName - Name of the group
   * @returns {Array<Object>} Array of sprite objects
   */
  getGroup(groupName) {
    const group = this.groups.get(groupName);
    if (!group) return [];
    
    return Array.from(group).map(spriteName => this.getSprite(spriteName)).filter(Boolean);
  }

  /**
   * Get a random sprite from a group
   * @param {string} groupName - Name of the group
   * @returns {Object|null} Random sprite or null if group is empty
   */
  getRandomFromGroup(groupName) {
    const groupSprites = this.getGroup(groupName);
    if (groupSprites.length === 0) return null;
    
    const randomIndex = Math.floor(Math.random() * groupSprites.length);
    return groupSprites[randomIndex];
  }

  /**
   * Check if a sprite exists
   * @param {string} spriteName - Name of the sprite
   * @returns {boolean} True if sprite exists
   */
  hasSprite(spriteName) {
    return this.sprites.has(spriteName);
  }

  /**
   * Get all sprite names
   * @returns {Array<string>} Array of all sprite names
   */
  getAllSpriteNames() {
    return Array.from(this.sprites.keys());
  }

  /**
   * Get all group names
   * @returns {Array<string>} Array of all group names
   */
  getAllGroupNames() {
    return Array.from(this.groups.keys());
  }

  /**
   * Get performance statistics
   * @returns {Object} Performance stats
   */
  getStats() {
    const totalLookups = this.stats.cacheHits + this.stats.cacheMisses;
    const hitRate = totalLookups === 0 ? 0 : ((this.stats.cacheHits / totalLookups) * 100).toFixed(1);
    return {
      ...this.stats,
      cacheSize: this.spriteCache.size,
      coordinateCacheSize: this.coordinateCache.size,
      totalSprites: this.sprites.size,
      totalAtlases: this.atlases.size,
      totalLookups,
      hitRate
    };
  }

  /**
   * Clear all caches (useful for memory management)
   */
  clearCaches() {
    this.spriteCache.clear();
    this.coordinateCache.clear();
    this.stats.cacheHits = 0;
    this.stats.cacheMisses = 0;
  }

  /**
   * Load an image with promise
   * @private
   */
  _loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`Failed to load image: ${url}`));
      img.src = url;
    });
  }

  /**
   * Draw a sprite to a canvas context
   * @param {CanvasRenderingContext2D} ctx - Canvas context
   * @param {string} spriteName - Name of the sprite
   * @param {number} destX - Destination X
   * @param {number} destY - Destination Y
   * @param {number} destWidth - Destination width
   * @param {number} destHeight - Destination height
   * @returns {boolean} True if drawn successfully
   */
  drawSprite(ctx, spriteName, destX, destY, destWidth, destHeight) {
    const sprite = this.getSprite(spriteName);
    if (!sprite) {
      console.warn(`Sprite not found: ${spriteName}`);
      return false;
    }
    
    try {
      ctx.drawImage(
        sprite.image,
        sprite.x, sprite.y, sprite.width, sprite.height,
        destX, destY, destWidth, destHeight
      );
      return true;
    } catch (error) {
      console.error(`Error drawing sprite ${spriteName}:`, error);
      return false;
    }
  }

  async loadEntities() {
    try {
      const res = await fetch('/api/entities');
      const json = await res.json();
      ['tiles', 'objects', 'enemies'].forEach(group => {
        (json[group] || []).forEach(ent => {
          this.entities.set(ent.id, ent);
          // attach sprite frame if available
          if (ent.sprite) ent.spriteFrame = this.get(ent.sprite);
        });
      });
      console.log(`[SpriteDB] Loaded ${this.entities.size} entities`);
    } catch (err) {
      console.error('[SpriteDB] loadEntities failed', err);
    }
  }

  getEntity(id) {
    return this.entities.get(id);
  }

  get(spriteName) {
    // Alias for getSprite to maintain backward compatibility
    return this.getSprite(spriteName);
  }
}

// Create singleton instance
export const spriteDatabase = new SpriteDatabase(); 