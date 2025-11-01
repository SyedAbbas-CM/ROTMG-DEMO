/**
 * OverworldConfig.js
 * Defines the 4x4 overworld region grid with biome types and generation weights
 *
 * World Structure:
 * - 4x4 grid of regions (16 total regions)
 * - Each region = 128x128 tiles
 * - Total world size = 512x512 tiles
 * - Each region has unique biome weights for procedural generation
 */

export const OVERWORLD_CONFIG = {
    // Grid configuration
    gridSize: 4,        // 4x4 regions
    regionSize: 128,    // 128x128 tiles per region
    chunkSize: 16,      // Keep existing 16x16 tile chunks

    // Total world size: 4 * 128 = 512 tiles in each dimension
    get worldSize() {
        return this.gridSize * this.regionSize;
    },

    /**
     * 4x4 Region Grid Layout
     * Distance from center (1.5, 1.5) determines region type
     *
     *     0       1       2       3
     * 0 [Plains][Forest][Forest][Mountain]
     * 1 [Plains][Desert][Forest][Mountain]
     * 2 [Water ][Desert][Plains][Hills   ]
     * 3 [Ocean ][Water ][Plains][Desert  ]
     */
    regions: [
        // Row 0 (North)
        { x: 0, y: 0, type: 'plains',   name: 'Northern Plains' },
        { x: 1, y: 0, type: 'forest',   name: 'Greenwood Forest' },
        { x: 2, y: 0, type: 'forest',   name: 'Deep Forest' },
        { x: 3, y: 0, type: 'mountain', name: 'Northern Peaks' },

        // Row 1
        { x: 0, y: 1, type: 'plains',   name: 'Central Plains' },
        { x: 1, y: 1, type: 'desert',   name: 'Arid Wastes' },
        { x: 2, y: 1, type: 'forest',   name: 'Wildwood' },
        { x: 3, y: 1, type: 'mountain', name: 'Eastern Peaks' },

        // Row 2
        { x: 0, y: 2, type: 'water',    name: 'Western Lakes' },
        { x: 1, y: 2, type: 'desert',   name: 'Scorched Lands' },
        { x: 2, y: 2, type: 'plains',   name: 'Heartland' },
        { x: 3, y: 2, type: 'hills',    name: 'Rolling Hills' },

        // Row 3 (South)
        { x: 0, y: 3, type: 'ocean',    name: 'Deep Waters' },
        { x: 1, y: 3, type: 'water',    name: 'Coastal Waters' },
        { x: 2, y: 3, type: 'plains',   name: 'Southern Plains' },
        { x: 3, y: 3, type: 'desert',   name: 'Southern Waste' },
    ],

    /**
     * Biome weights for each region type
     * These modify the tile selection probabilities during procedural generation
     * Higher values = more likely to appear
     */
    biomeWeights: {
        // Plains: Mostly grass with NO TREES (only sparse boulders)
        plains: {
            grass: 0.65,
            grass_yellow: 0.15,
            grass_dark: 0.05,
            tree: 0.10,
            water_1: 0.03,
            mountain: 0.02,

            // Object density - NO TREES on plains per user requirement
            objectDensity: {
                tree: 0.0,         // NO TREES on plains
                boulder: 0.002,    // Very sparse boulders (0.2%)
                flowers: 0.015     // Moderate flowers
            }
        },

        // Forest: Dense trees, darker grass
        forest: {
            grass_dark: 0.40,
            tree: 0.35,
            grass: 0.15,
            water_1: 0.05,
            mountain: 0.05,

            objectDensity: {
                tree: 0.20,        // 20% chance - dense forest
                boulder: 0.02,
                flowers: 0.01
            }
        },

        // Desert: Sand, dead trees, sparse rocks
        desert: {
            sand_1: 0.50,
            sand_2: 0.30,
            tree_dead: 0.10,
            grass_yellow: 0.05,
            boulder: 0.05,

            objectDensity: {
                tree_dead: 0.03,
                boulder: 0.04,
                flowers: 0.0
            }
        },

        // Mountain: Rocky terrain, boulders
        mountain: {
            mountain: 0.50,
            cobblestone: 0.25,
            boulder: 0.15,
            grass_dark: 0.05,
            tree: 0.05,

            objectDensity: {
                boulder: 0.15,     // 15% chance - rocky mountain terrain
                tree: 0.02,
                flowers: 0.0
            }
        },

        // Hills: Mix of grass and rocks
        hills: {
            grass: 0.40,
            grass_dark: 0.20,
            mountain: 0.20,
            boulder: 0.10,
            tree: 0.10,

            objectDensity: {
                boulder: 0.08,
                tree: 0.06,
                flowers: 0.02
            }
        },

        // Water: Shallow water with rocks
        water: {
            water_1: 0.50,
            water_2: 0.30,
            grass: 0.10,
            sand_1: 0.05,
            deep_water: 0.05,

            objectDensity: {
                boulder: 0.02,     // Rocks in water
                tree: 0.0,
                flowers: 0.0
            }
        },

        // Ocean: Deep water
        ocean: {
            deep_water: 0.60,
            deep_water_2: 0.30,
            water_1: 0.10,

            objectDensity: {
                boulder: 0.01,
                tree: 0.0,
                flowers: 0.0
            }
        },
    },

    /**
     * Get region configuration for a given region coordinate
     * @param {number} regionX - Region X coordinate (0-3)
     * @param {number} regionY - Region Y coordinate (0-3)
     * @returns {Object} Region configuration
     */
    getRegion(regionX, regionY) {
        const index = regionY * this.gridSize + regionX;
        return this.regions[index] || this.regions[0]; // Fallback to first region
    },

    /**
     * Get region coordinates from world coordinates
     * @param {number} worldX - World X coordinate (0-511)
     * @param {number} worldY - World Y coordinate (0-511)
     * @returns {Object} { regionX, regionY, localX, localY }
     */
    worldToRegion(worldX, worldY) {
        const regionX = Math.floor(worldX / this.regionSize);
        const regionY = Math.floor(worldY / this.regionSize);
        const localX = worldX % this.regionSize;
        const localY = worldY % this.regionSize;

        return { regionX, regionY, localX, localY };
    },

    /**
     * Get region coordinates from chunk coordinates
     * @param {number} chunkX - Chunk X coordinate
     * @param {number} chunkY - Chunk Y coordinate
     * @returns {Object} { regionX, regionY }
     */
    chunkToRegion(chunkX, chunkY) {
        const tilesPerRegion = this.regionSize;
        const chunksPerRegion = tilesPerRegion / this.chunkSize; // 128 / 16 = 8 chunks per region

        const regionX = Math.floor(chunkX / chunksPerRegion);
        const regionY = Math.floor(chunkY / chunksPerRegion);

        return { regionX, regionY };
    },

    /**
     * Get biome weights for a region
     * @param {string} regionType - Region type (plains, forest, desert, etc.)
     * @returns {Object} Biome weights
     */
    getBiomeWeights(regionType) {
        return this.biomeWeights[regionType] || this.biomeWeights.plains;
    }
};

// Export as default for ES6 import
export default OVERWORLD_CONFIG;
