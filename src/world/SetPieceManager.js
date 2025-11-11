/**
 * SetPieceManager - Loads and randomly places predefined map chunks
 *
 * Set pieces are predefined structures (temples, camps, ruins) that spawn
 * randomly during map generation to add variety and points of interest.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class SetPieceManager {
  constructor() {
    this.setPieces = new Map(); // Map of setPieceId -> setpiece data
    this.loadedCount = 0;
  }

  /**
   * Load all set pieces from the maps directory
   * @param {string} mapsDir - Directory containing map JSON files
   */
  loadSetPieces(mapsDir = '/Users/az/Desktop/Rotmg-Pservers/ROTMG-DEMO/public/maps') {
    try {
      const files = fs.readdirSync(mapsDir);
      const jsonFiles = files.filter(f => f.endsWith('.json'));

      console.log(`[SETPIECE] Loading ${jsonFiles.length} set pieces from ${mapsDir}`);

      for (const file of jsonFiles) {
        const filePath = path.join(mapsDir, file);
        const setPieceId = file.replace('.json', '');

        try {
          const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));

          // Store the set piece
          this.setPieces.set(setPieceId, {
            id: setPieceId,
            width: data.width || 32,
            height: data.height || 32,
            data: data,
            fileName: file
          });

          this.loadedCount++;
          console.log(`[SETPIECE] Loaded: ${setPieceId} (${data.width}x${data.height})`);
        } catch (err) {
          console.error(`[SETPIECE] Failed to load ${file}:`, err.message);
        }
      }

      console.log(`[SETPIECE] Successfully loaded ${this.loadedCount} set pieces`);
    } catch (err) {
      console.error('[SETPIECE] Failed to load set pieces directory:', err.message);
    }
  }

  /**
   * Get a random set piece
   * @returns {Object|null} Random set piece or null if none loaded
   */
  getRandomSetPiece() {
    if (this.setPieces.size === 0) return null;

    const pieces = Array.from(this.setPieces.values());
    return pieces[Math.floor(Math.random() * pieces.length)];
  }

  /**
   * Get a specific set piece by ID
   * @param {string} id - Set piece ID
   * @returns {Object|null} Set piece or null if not found
   */
  getSetPiece(id) {
    return this.setPieces.get(id) || null;
  }

  /**
   * Generate random placement locations for set pieces across the map
   * @param {number} mapWidth - Map width in tiles
   * @param {number} mapHeight - Map height in tiles
   * @param {number} count - Number of set pieces to place
   * @param {number} minDistance - Minimum distance between set pieces
   * @returns {Array} Array of {x, y, setPieceId} placement locations
   */
  generatePlacements(mapWidth, mapHeight, count = 5, minDistance = 200) {
    const placements = [];
    const margin = 100; // Don't place near edges
    const maxAttempts = 100;

    for (let i = 0; i < count; i++) {
      const setPiece = this.getRandomSetPiece();
      if (!setPiece) continue;

      let placed = false;
      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        // Generate random position with margin
        const x = margin + Math.floor(Math.random() * (mapWidth - 2 * margin - setPiece.width));
        const y = margin + Math.floor(Math.random() * (mapHeight - 2 * margin - setPiece.height));

        // Check distance from other set pieces
        const tooClose = placements.some(p => {
          const dx = x - p.x;
          const dy = y - p.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          return distance < minDistance;
        });

        if (!tooClose) {
          placements.push({
            x,
            y,
            setPieceId: setPiece.id,
            width: setPiece.width,
            height: setPiece.height
          });
          placed = true;
          console.log(`[SETPIECE] Placed ${setPiece.id} at (${x}, ${y})`);
          break;
        }
      }

      if (!placed) {
        console.warn(`[SETPIECE] Could not place ${setPiece.id} after ${maxAttempts} attempts`);
      }
    }

    return placements;
  }

  /**
   * Apply a set piece to a map's tile grid
   * @param {Object} mapManager - MapManager instance
   * @param {number} originX - Top-left X coordinate to place set piece
   * @param {number} originY - Top-left Y coordinate to place set piece
   * @param {string} setPieceId - ID of set piece to place
   */
  applySetPiece(mapManager, originX, originY, setPieceId) {
    const setPiece = this.getSetPiece(setPieceId);
    if (!setPiece) {
      console.error(`[SETPIECE] Set piece not found: ${setPieceId}`);
      return;
    }

    const { data } = setPiece;

    // Apply layers if they exist (new format)
    if (data.layers && Array.isArray(data.layers)) {
      for (const layer of data.layers) {
        if (!layer.grid || !layer.visible) continue;

        for (let y = 0; y < layer.grid.length; y++) {
          for (let x = 0; x < layer.grid[y].length; x++) {
            const cell = layer.grid[y][x];
            if (!cell || !cell.sprite) continue;

            const worldX = originX + x;
            const worldY = originY + y;

            // Set the tile in the map
            // Note: MapManager.setTile() needs to exist for this to work
            if (typeof mapManager.setTile === 'function') {
              mapManager.setTile(worldX, worldY, {
                sprite: cell.sprite,
                rotation: cell.rot || 0,
                walkable: true // Default walkable, may need more logic
              });
            }
          }
        }
      }
    }
    // Apply ground layer if it exists (old format)
    else if (data.ground && Array.isArray(data.ground)) {
      for (let y = 0; y < data.ground.length; y++) {
        for (let x = 0; x < data.ground[y].length; x++) {
          const cell = data.ground[y][x];
          if (!cell || !cell.sprite) continue;

          const worldX = originX + x;
          const worldY = originY + y;

          if (typeof mapManager.setTile === 'function') {
            mapManager.setTile(worldX, worldY, {
              sprite: cell.sprite,
              rotation: cell.rot || 0,
              walkable: true
            });
          }
        }
      }
    }

    console.log(`[SETPIECE] Applied ${setPieceId} at (${originX}, ${originY})`);
  }

  /**
   * Get all available set piece IDs
   * @returns {Array<string>} Array of set piece IDs
   */
  getAvailableSetPieces() {
    return Array.from(this.setPieces.keys());
  }

  /**
   * Get count of loaded set pieces
   * @returns {number} Number of loaded set pieces
   */
  getCount() {
    return this.setPieces.size;
  }
}

export default SetPieceManager;
