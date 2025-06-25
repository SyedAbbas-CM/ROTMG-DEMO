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
      // Normalise image path – many atlas JSONs store paths without a leading "/" which causes
      // the browser to resolve them relative to the current page (e.g. /tools/map-editor.html).
      // We want them to be resolved from the web-root instead, so we prepend "/" unless the path
      // already starts with a slash or with a protocol (http/https).
      let imgPath = cfg.path || cfg.image || (cfg.meta && cfg.meta.image);
      if (imgPath && !imgPath.startsWith('/') && !/^https?:\/\//i.test(imgPath)) {
        imgPath = '/' + imgPath;
      }
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

    // --- Group / tagging support ----------------------------------------
    //  Sprite editors can add either:
    //    "group": "tiles"               (single group string)
    // or "groups": ["tiles","walls"]   (array)
    // or "tags":   ["tiles","walls"]   (alias)
    // We normalise all of them into this.groups map.
    const tagSet = new Set();
    if (typeof spriteConfig.group === 'string') tagSet.add(spriteConfig.group);
    if (Array.isArray(spriteConfig.groups)) spriteConfig.groups.forEach(g => tagSet.add(g));
    if (Array.isArray(spriteConfig.tags)) spriteConfig.tags.forEach(g => tagSet.add(g));

    // Heuristic fallback – many raw atlases mark everything as "auto"
    if (tagSet.size===0 || (tagSet.size===1 && tagSet.has('auto'))){
      tagSet.delete('auto');
      const atlasLower = atlasName.toLowerCase();
      if(/world|environment|terrain/.test(atlasLower)) tagSet.add('tiles');
      else if(/obj|object|prop/.test(atlasLower)) tagSet.add('objects');
      else if(/char|enemy|creature|monster/.test(atlasLower)) tagSet.add('enemies');
      else tagSet.add('misc');
    }

    tagSet.forEach(g => {
      if (!this.groups.has(g)) this.groups.set(g, new Set());
      this.groups.get(g).add(name);
    });

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
   * Check if an atlas has been registered.  Added for compatibility with
   * legacy code that previously called spriteDatabase.hasAtlas(name).
   */
  hasAtlas(atlasName) {
    return this.atlases.has(atlasName);
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

  /**
   * Compatibility helper replicating the old spriteManager.fetchGridSprite API.
   * It returns a sprite extracted from grid coordinates and, if an alias is
   * provided, registers that alias for future quick look-ups.
   */
  fetchGridSprite(atlasName, row, col, alias = null, spriteWidth = null, spriteHeight = null) {
    const baseSprite = this.getSpriteByGrid(atlasName, row, col);
    if (!baseSprite) return null;

    // If the caller requested a larger logical cell (e.g. 12×12 while the
    // underlying atlas frame is 8×8) we build a *padded canvas* so the sprite
    // still occupies the full tile.  This prevents the 8-pixel art sitting
    // inside a 12-pixel black square.
    if (spriteWidth || spriteHeight) {
      const targetW = spriteWidth  || baseSprite.width;
      const targetH = spriteHeight || baseSprite.height;

      // Only upscale / pad when the requested size is bigger than the source.
      if (targetW !== baseSprite.width || targetH !== baseSprite.height) {
        const canvas   = document.createElement('canvas');
        canvas.width   = targetW;
        canvas.height  = targetH;
        const ctx      = canvas.getContext('2d');
        // Keep crisp pixel art when scaling
        ctx.imageSmoothingEnabled = false;

        // -----------------------------------------------------------------
        // 1) Copy JUST the native frame (8×8, 10×10, …) into a temp canvas.
        // -----------------------------------------------------------------
        const srcW = baseSprite.width;
        const srcH = baseSprite.height;
        const tmpCvs = document.createElement('canvas');
        tmpCvs.width  = srcW;
        tmpCvs.height = srcH;
        const tmpCtx  = tmpCvs.getContext('2d');
        tmpCtx.imageSmoothingEnabled = false;
        tmpCtx.clearRect(0,0,srcW,srcH);
        tmpCtx.drawImage(
          baseSprite.image,
          baseSprite.x,
          baseSprite.y,
          srcW,
          srcH,
          0,
          0,
          srcW,
          srcH
        );

        // ---------------------------------------------------------------
        // 2) Paste it centred into the 12×12 (or requested) destination.
        //    Any leftover pixels around stay transparent so no neighbour
        //    bleed occurs.
        // ---------------------------------------------------------------
        const padX = Math.floor((targetW - srcW) / 2);
        const padY = Math.floor((targetH - srcH) / 2);
        ctx.clearRect(0,0,targetW,targetH);
        ctx.drawImage(tmpCvs, 0, 0, srcW, srcH, padX, padY, srcW, srcH);

        // Replace the sprite definition with the up-scaled canvas region
        const paddedSprite = {
          ...baseSprite,
          image: canvas,
          x: 0,
          y: 0,
          width: targetW,
          height: targetH
        };

        // Use new definition for aliasing / return value
        if (alias) {
          this.sprites.set(alias, paddedSprite);
        }
        return paddedSprite;
      } else {
        // Source sprite already matches requested size (e.g. 12×12) but some
        // Oryx sheets include a 1-pixel transparent frame that shows up as a
        // dark halo against the map background.  We crop that outer pixel on
        // all sides and re-draw the central 10×10 (or 11×11) region so the
        // sprite fills the tile cleanly.

        const CROP = 1;
        if (targetW >= 12) {
          const canvas = document.createElement('canvas');
          canvas.width  = targetW;
          canvas.height = targetH;
          const ctx = canvas.getContext('2d');
          ctx.imageSmoothingEnabled = false;
          ctx.clearRect(0,0,targetW,targetH);

          ctx.drawImage(
            baseSprite.image,
            baseSprite.x   + CROP,
            baseSprite.y   + CROP,
            baseSprite.width  - 2*CROP,
            baseSprite.height - 2*CROP,
            0,
            0,
            targetW,
            targetH
          );

          baseSprite.image  = canvas;
          baseSprite.x      = 0;
          baseSprite.y      = 0;
          baseSprite.width  = targetW;
          baseSprite.height = targetH;
        }

        // Requested size matches source – just update meta width/height so
        // renderers scale correctly.
        baseSprite.width  = targetW;
        baseSprite.height = targetH;
      }
    }

    // Register alias mapping if requested and not already present
    if (alias) {
      this.sprites.set(alias, baseSprite); // unconditionally overwrite
    }

    return baseSprite;
  }
}

// Create singleton instance
export const spriteDatabase = new SpriteDatabase(); 