/**
 * EfficientWorldManager - Isolated Overworld System Demo
 * 5k x 5k world with 100x100 tile chunks
 * Completely standalone - no dependencies on main game
 */

export class EfficientWorldManager {
    constructor(options = {}) {
        // Reasonable world size for demo
        this.worldWidth = options.worldWidth || 5000;   
        this.worldHeight = options.worldHeight || 5000; 
        this.chunkSize = options.chunkSize || 100;      
        
        // Calculate chunk grid dimensions
        this.chunksX = Math.ceil(this.worldWidth / this.chunkSize);   
        this.chunksY = Math.ceil(this.worldHeight / this.chunkSize);  
        
        this.seed = options.seed || Math.random();
        
        // Active loaded chunks
        this.loadedChunks = new Map();
        this.maxLoadedChunks = options.maxLoadedChunks || 25;
        
        // Terrain sprite mappings for lofiEnvironment.png
        this.terrainSprites = {
            grass1: { col: 5, row: 7 },      // Plains
            stone1: { col: 5, row: 8 },      // Mountains  
            water1: { col: 5, row: 9 },      // Water
            grass2: { col: 6, row: 7 },      // Forest
            stone2: { col: 6, row: 8 },      // Desert
            dirt1: { col: 6, row: 9 },       // Wasteland
        };
        
        // Map terrain types to sprite choices
        this.terrainMap = {
            plains: ['grass1', 'grass2'],
            mountains: ['stone1', 'stone2'], 
            water: ['water1'],
            wasteland: ['dirt1'],
            forest: ['grass1', 'grass2'],
            desert: ['dirt1', 'stone2']
        };
        
        console.log(`[OverworldDemo] Initialized ${this.worldWidth}x${this.worldHeight} world`);
        console.log(`[OverworldDemo] ${this.chunksX}x${this.chunksY} chunks (${this.chunksX * this.chunksY} total)`);
    }
    
    worldToChunk(worldX, worldY) {
        const chunkX = Math.floor(worldX / this.chunkSize);
        const chunkY = Math.floor(worldY / this.chunkSize);
        const localX = worldX % this.chunkSize;
        const localY = worldY % this.chunkSize;
        return { chunkX, chunkY, localX, localY };
    }
    
    loadChunk(chunkX, chunkY) {
        const chunkId = `${chunkX},${chunkY}`;
        
        if (this.loadedChunks.has(chunkId)) {
            const chunk = this.loadedChunks.get(chunkId);
            chunk.lastAccessed = Date.now();
            return chunk;
        }
        
        const chunk = this.generateChunk(chunkX, chunkY);
        this.loadedChunks.set(chunkId, {
            ...chunk,
            loadedAt: Date.now(),
            lastAccessed: Date.now()
        });
        
        this.manageChunkMemory();
        console.log(`[OverworldDemo] Loaded chunk (${chunkX}, ${chunkY})`);
        return chunk;
    }
    
    generateChunk(chunkX, chunkY) {
        const chunkSeed = this.hashCoords(chunkX, chunkY, this.seed);
        const rng = this.createSeededRNG(chunkSeed);
        const terrainType = this.selectChunkTerrain(chunkX, chunkY, rng);
        
        return {
            id: `${chunkX},${chunkY}`,
            chunkX, chunkY, terrainType,
            generated: true, seed: chunkSeed
        };
    }
    
    selectChunkTerrain(chunkX, chunkY, rng) {
        const centerX = this.chunksX / 2;
        const centerY = this.chunksY / 2;
        const distanceFromCenter = Math.sqrt(
            Math.pow(chunkX - centerX, 2) + Math.pow(chunkY - centerY, 2)
        );
        
        if (distanceFromCenter < 10) {
            return this.weightedChoice(['plains', 'forest', 'water'], [60, 30, 10], rng);
        } else if (distanceFromCenter < 20) {
            return this.weightedChoice(['plains', 'forest', 'mountains', 'water'], [40, 30, 20, 10], rng);
        } else {
            return this.weightedChoice(['mountains', 'wasteland', 'desert', 'water'], [40, 30, 20, 10], rng);
        }
    }
    
    getTerrainAt(worldX, worldY) {
        const { chunkX, chunkY } = this.worldToChunk(worldX, worldY);
        const chunk = this.loadChunk(chunkX, chunkY);
        const sprite = this.getTerrainSprite(chunk.terrainType, worldX, worldY);
        
        return {
            terrainType: chunk.terrainType,
            sprite: sprite,
            chunkId: chunk.id
        };
    }
    
    getTerrainSprite(terrainType, x = 0, y = 0) {
        const sprites = this.terrainMap[terrainType] || this.terrainMap.plains;
        const seed = ((x * 73856093) ^ (y * 19349663)) >>> 0;
        const index = seed % sprites.length;
        const spriteName = sprites[index];
        return this.terrainSprites[spriteName] || this.terrainSprites.grass1;
    }
    
    manageChunkMemory() {
        if (this.loadedChunks.size <= this.maxLoadedChunks) return;
        
        const chunks = Array.from(this.loadedChunks.entries())
            .sort(([,a], [,b]) => a.lastAccessed - b.lastAccessed);
        
        const toUnload = chunks.slice(0, chunks.length - this.maxLoadedChunks);
        toUnload.forEach(([chunkId]) => {
            this.loadedChunks.delete(chunkId);
        });
        
        if (toUnload.length > 0) {
            console.log(`[OverworldDemo] Unloaded ${toUnload.length} old chunks`);
        }
    }
    
    getStats() {
        return {
            worldSize: `${this.worldWidth}x${this.worldHeight}`,
            chunkGrid: `${this.chunksX}x${this.chunksY}`,
            chunkSize: `${this.chunkSize}x${this.chunkSize}`,
            totalChunks: this.chunksX * this.chunksY,
            loadedChunks: this.loadedChunks.size,
            memoryUsage: `${(this.loadedChunks.size * 0.1).toFixed(1)} KB`
        };
    }
    
    // Utility functions
    hashCoords(x, y, seed) {
        return ((x * 73856093) ^ (y * 19349663) ^ (seed * 83492791)) >>> 0;
    }
    
    createSeededRNG(seed) {
        let state = seed;
        return function() {
            state = ((state * 9301 + 49297) % 233280) >>> 0;
            return state / 233280;
        };
    }
    
    weightedChoice(items, weights, rng) {
        const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
        let randomValue = rng() * totalWeight;
        
        for (let i = 0; i < items.length; i++) {
            randomValue -= weights[i];
            if (randomValue <= 0) {
                return items[i];
            }
        }
        
        return items[items.length - 1];
    }
}