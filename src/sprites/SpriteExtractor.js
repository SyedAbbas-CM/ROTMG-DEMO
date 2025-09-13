/**
 * SpriteExtractor - Extract 8x8 sprites from the lofiEnvironment spritesheet
 */

export class SpriteExtractor {
    constructor() {
        this.spriteSize = 8;
        this.sheetPath = 'public/assets/images/lofiEnvironment.png';
        
        // Define the terrain sprites we want from columns 5-6, rows 7-9
        this.terrainSprites = {
            // Column 5 (index 4 in 0-based)
            grass1: { col: 5, row: 7 },      // 5,7 - grass/plains
            stone1: { col: 5, row: 8 },      // 5,8 - stone/mountain  
            water1: { col: 5, row: 9 },      // 5,9 - water/river
            
            // Column 6 (index 5 in 0-based)
            grass2: { col: 6, row: 7 },      // 6,7 - different grass
            stone2: { col: 6, row: 8 },      // 6,8 - different stone
            dirt1: { col: 6, row: 9 },       // 6,9 - dirt/wasteland
        };
        
        // Map terrain types to sprite choices
        this.terrainMap = {
            plains: ['grass1', 'grass2'],
            mountains: ['stone1', 'stone2'], 
            water: ['water1'],
            wasteland: ['dirt1'],
            forest: ['grass1', 'grass2'], // Use grass variants for now
            desert: ['dirt1', 'stone1']
        };
    }
    
    /**
     * Get sprite coordinates for a terrain type
     * @param {string} terrainType - Type of terrain
     * @param {number} x - World X for procedural selection
     * @param {number} y - World Y for procedural selection
     * @returns {Object} Sprite coordinates {col, row}
     */
    getTerrainSprite(terrainType, x = 0, y = 0) {
        const sprites = this.terrainMap[terrainType] || this.terrainMap.plains;
        
        // Use coordinates for procedural selection
        const seed = ((x * 73856093) ^ (y * 19349663)) >>> 0;
        const index = seed % sprites.length;
        const spriteName = sprites[index];
        
        return this.terrainSprites[spriteName] || this.terrainSprites.grass1;
    }
    
    /**
     * Get all available terrain types
     * @returns {Array} Array of terrain type names
     */
    getTerrainTypes() {
        return Object.keys(this.terrainMap);
    }
    
    /**
     * Convert sprite coordinates to pixel coordinates in the spritesheet
     * @param {number} col - Column (1-based)
     * @param {number} row - Row (1-based) 
     * @returns {Object} Pixel coordinates {x, y, width, height}
     */
    spriteToPixels(col, row) {
        return {
            x: (col - 1) * this.spriteSize,
            y: (row - 1) * this.spriteSize,
            width: this.spriteSize,
            height: this.spriteSize
        };
    }
    
    /**
     * Generate ASCII representation using simple characters
     * @param {string} terrainType - Terrain type
     * @returns {string} ASCII character
     */
    getASCIIChar(terrainType) {
        const asciiMap = {
            plains: '.',
            mountains: '^',
            water: '~',
            wasteland: 'x',
            forest: 'T',
            desert: 's'
        };
        return asciiMap[terrainType] || '.';
    }
    
    /**
     * Generate colored terminal representation
     * @param {string} terrainType - Terrain type
     * @returns {string} Colored character with ANSI codes
     */
    getColoredChar(terrainType) {
        const colorMap = {
            plains: '\x1b[32m.\x1b[0m',      // Green
            mountains: '\x1b[37m^\x1b[0m',   // White
            water: '\x1b[34m~\x1b[0m',       // Blue
            wasteland: '\x1b[31mx\x1b[0m',   // Red
            forest: '\x1b[32mT\x1b[0m',      // Green
            desert: '\x1b[33ms\x1b[0m'       // Yellow
        };
        return colorMap[terrainType] || '.';
    }
}

export default SpriteExtractor;