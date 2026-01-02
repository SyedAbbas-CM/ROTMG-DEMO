/**
 * Database.js - SQLite database for player persistence
 * Stores: players, characters, inventory, items
 */

import Database from 'better-sqlite3';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Database file location
const DB_PATH = path.join(__dirname, '../../data/game.db');

class GameDatabase {
    constructor() {
        this.db = null;
    }

    /**
     * Initialize database connection and create tables
     */
    init() {
        // Ensure data directory exists
        const dataDir = path.dirname(DB_PATH);
        if (!fs.existsSync(dataDir)) {
            fs.mkdirSync(dataDir, { recursive: true });
        }

        this.db = new Database(DB_PATH);
        this.db.pragma('journal_mode = WAL'); // Better performance

        this.createTables();
        console.log('[Database] Initialized at', DB_PATH);
        return this;
    }

    /**
     * Create database tables if they don't exist
     */
    createTables() {
        // Players table - account info
        this.db.exec(`
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                email TEXT UNIQUE,
                password_hash TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_login DATETIME,
                is_banned INTEGER DEFAULT 0,
                ban_reason TEXT
            )
        `);

        // Characters table - each player can have multiple characters
        this.db.exec(`
            CREATE TABLE IF NOT EXISTS characters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                class TEXT NOT NULL DEFAULT 'warrior',
                level INTEGER DEFAULT 1,
                experience INTEGER DEFAULT 0,
                health INTEGER DEFAULT 200,
                max_health INTEGER DEFAULT 200,
                mana INTEGER DEFAULT 100,
                max_mana INTEGER DEFAULT 100,
                attack INTEGER DEFAULT 10,
                defense INTEGER DEFAULT 5,
                speed INTEGER DEFAULT 5,
                dexterity INTEGER DEFAULT 5,
                vitality INTEGER DEFAULT 5,
                wisdom INTEGER DEFAULT 5,
                x REAL DEFAULT 30.0,
                y REAL DEFAULT 30.0,
                world_id TEXT DEFAULT 'map_1',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_played DATETIME,
                is_dead INTEGER DEFAULT 0,
                fame INTEGER DEFAULT 0,
                FOREIGN KEY (player_id) REFERENCES players(id)
            )
        `);

        // Inventory table - items per character
        this.db.exec(`
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id INTEGER NOT NULL,
                slot INTEGER NOT NULL,
                item_id TEXT NOT NULL,
                quantity INTEGER DEFAULT 1,
                tier INTEGER DEFAULT 0,
                enchantments TEXT,
                FOREIGN KEY (character_id) REFERENCES characters(id),
                UNIQUE(character_id, slot)
            )
        `);

        // Equipment table - equipped items
        this.db.exec(`
            CREATE TABLE IF NOT EXISTS equipment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id INTEGER NOT NULL,
                slot TEXT NOT NULL,
                item_id TEXT NOT NULL,
                tier INTEGER DEFAULT 0,
                enchantments TEXT,
                FOREIGN KEY (character_id) REFERENCES characters(id),
                UNIQUE(character_id, slot)
            )
        `);

        // Create indexes for faster lookups
        this.db.exec(`
            CREATE INDEX IF NOT EXISTS idx_characters_player ON characters(player_id);
            CREATE INDEX IF NOT EXISTS idx_inventory_character ON inventory(character_id);
            CREATE INDEX IF NOT EXISTS idx_equipment_character ON equipment(character_id);
            CREATE INDEX IF NOT EXISTS idx_players_email ON players(email);
        `);

        console.log('[Database] Tables created/verified');
    }

    // ==================== PLAYER OPERATIONS ====================

    /**
     * Create a new player account
     */
    createPlayer(name, email, passwordHash = null) {
        const stmt = this.db.prepare(`
            INSERT INTO players (name, email, password_hash, last_login)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        `);

        try {
            const result = stmt.run(name, email, passwordHash);
            console.log(`[Database] Created player: ${name} (ID: ${result.lastInsertRowid})`);
            return { id: result.lastInsertRowid, name, email };
        } catch (err) {
            if (err.message.includes('UNIQUE constraint')) {
                console.log(`[Database] Player ${name} or email ${email} already exists`);
                return null;
            }
            throw err;
        }
    }

    /**
     * Get player by name
     */
    getPlayerByName(name) {
        const stmt = this.db.prepare('SELECT * FROM players WHERE name = ?');
        return stmt.get(name);
    }

    /**
     * Get player by email
     */
    getPlayerByEmail(email) {
        const stmt = this.db.prepare('SELECT * FROM players WHERE email = ?');
        return stmt.get(email);
    }

    /**
     * Get player by ID
     */
    getPlayerById(id) {
        const stmt = this.db.prepare('SELECT * FROM players WHERE id = ?');
        return stmt.get(id);
    }

    /**
     * Update player's last login time
     */
    updateLastLogin(playerId) {
        const stmt = this.db.prepare('UPDATE players SET last_login = CURRENT_TIMESTAMP WHERE id = ?');
        stmt.run(playerId);
    }

    // ==================== CHARACTER OPERATIONS ====================

    /**
     * Create a new character for a player
     */
    createCharacter(playerId, name, characterClass = 'warrior') {
        // Get class defaults
        const classDefaults = this.getClassDefaults(characterClass);

        const stmt = this.db.prepare(`
            INSERT INTO characters (
                player_id, name, class, health, max_health, mana, max_mana,
                attack, defense, speed, dexterity, vitality, wisdom, last_played
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        `);

        const result = stmt.run(
            playerId, name, characterClass,
            classDefaults.health, classDefaults.maxHealth,
            classDefaults.mana, classDefaults.maxMana,
            classDefaults.attack, classDefaults.defense,
            classDefaults.speed, classDefaults.dexterity,
            classDefaults.vitality, classDefaults.wisdom
        );

        console.log(`[Database] Created character: ${name} (${characterClass}) for player ${playerId}`);
        return this.getCharacterById(result.lastInsertRowid);
    }

    /**
     * Get class default stats
     */
    getClassDefaults(characterClass) {
        const defaults = {
            warrior: { health: 200, maxHealth: 200, mana: 100, maxMana: 100, attack: 15, defense: 10, speed: 5, dexterity: 5, vitality: 10, wisdom: 5 },
            archer: { health: 150, maxHealth: 150, mana: 100, maxMana: 100, attack: 12, defense: 5, speed: 7, dexterity: 10, vitality: 5, wisdom: 5 },
            wizard: { health: 100, maxHealth: 100, mana: 200, maxMana: 200, attack: 8, defense: 3, speed: 5, dexterity: 5, vitality: 3, wisdom: 15 },
            priest: { health: 120, maxHealth: 120, mana: 180, maxMana: 180, attack: 6, defense: 5, speed: 5, dexterity: 5, vitality: 5, wisdom: 12 },
            rogue: { health: 130, maxHealth: 130, mana: 100, maxMana: 100, attack: 10, defense: 3, speed: 10, dexterity: 12, vitality: 5, wisdom: 5 },
            knight: { health: 250, maxHealth: 250, mana: 50, maxMana: 50, attack: 12, defense: 15, speed: 3, dexterity: 5, vitality: 15, wisdom: 3 },
        };
        return defaults[characterClass] || defaults.warrior;
    }

    /**
     * Get character by ID
     */
    getCharacterById(id) {
        const stmt = this.db.prepare('SELECT * FROM characters WHERE id = ?');
        return stmt.get(id);
    }

    /**
     * Get all characters for a player
     */
    getCharactersByPlayerId(playerId) {
        const stmt = this.db.prepare('SELECT * FROM characters WHERE player_id = ? ORDER BY last_played DESC');
        return stmt.all(playerId);
    }

    /**
     * Update character stats/position
     */
    updateCharacter(characterId, updates) {
        const allowedFields = ['health', 'max_health', 'mana', 'max_mana', 'level', 'experience',
                              'x', 'y', 'world_id', 'is_dead', 'fame', 'attack', 'defense',
                              'speed', 'dexterity', 'vitality', 'wisdom'];

        const fields = [];
        const values = [];

        for (const [key, value] of Object.entries(updates)) {
            if (allowedFields.includes(key)) {
                fields.push(`${key} = ?`);
                values.push(value);
            }
        }

        if (fields.length === 0) return false;

        fields.push('last_played = CURRENT_TIMESTAMP');
        values.push(characterId);

        const stmt = this.db.prepare(`UPDATE characters SET ${fields.join(', ')} WHERE id = ?`);
        stmt.run(...values);
        return true;
    }

    /**
     * Save character position (called frequently)
     */
    saveCharacterPosition(characterId, x, y, worldId) {
        const stmt = this.db.prepare('UPDATE characters SET x = ?, y = ?, world_id = ?, last_played = CURRENT_TIMESTAMP WHERE id = ?');
        stmt.run(x, y, worldId, characterId);
    }

    /**
     * Mark character as dead
     */
    killCharacter(characterId) {
        const stmt = this.db.prepare('UPDATE characters SET is_dead = 1, last_played = CURRENT_TIMESTAMP WHERE id = ?');
        stmt.run(characterId);
    }

    // ==================== INVENTORY OPERATIONS ====================

    /**
     * Get character's inventory
     */
    getInventory(characterId) {
        const stmt = this.db.prepare('SELECT * FROM inventory WHERE character_id = ? ORDER BY slot');
        return stmt.all(characterId);
    }

    /**
     * Set item in inventory slot
     */
    setInventorySlot(characterId, slot, itemId, quantity = 1, tier = 0, enchantments = null) {
        const stmt = this.db.prepare(`
            INSERT OR REPLACE INTO inventory (character_id, slot, item_id, quantity, tier, enchantments)
            VALUES (?, ?, ?, ?, ?, ?)
        `);
        stmt.run(characterId, slot, itemId, quantity, tier, enchantments ? JSON.stringify(enchantments) : null);
    }

    /**
     * Clear inventory slot
     */
    clearInventorySlot(characterId, slot) {
        const stmt = this.db.prepare('DELETE FROM inventory WHERE character_id = ? AND slot = ?');
        stmt.run(characterId, slot);
    }

    /**
     * Get equipped items
     */
    getEquipment(characterId) {
        const stmt = this.db.prepare('SELECT * FROM equipment WHERE character_id = ?');
        return stmt.all(characterId);
    }

    /**
     * Equip item
     */
    equipItem(characterId, slot, itemId, tier = 0, enchantments = null) {
        const stmt = this.db.prepare(`
            INSERT OR REPLACE INTO equipment (character_id, slot, item_id, tier, enchantments)
            VALUES (?, ?, ?, ?, ?)
        `);
        stmt.run(characterId, slot, itemId, tier, enchantments ? JSON.stringify(enchantments) : null);
    }

    /**
     * Unequip item from slot
     */
    unequipItem(characterId, slot) {
        const stmt = this.db.prepare('DELETE FROM equipment WHERE character_id = ? AND slot = ?');
        stmt.run(characterId, slot);
    }

    // ==================== UTILITY ====================

    /**
     * Close database connection
     */
    close() {
        if (this.db) {
            this.db.close();
            console.log('[Database] Connection closed');
        }
    }

    /**
     * Get database stats
     */
    getStats() {
        const players = this.db.prepare('SELECT COUNT(*) as count FROM players').get();
        const characters = this.db.prepare('SELECT COUNT(*) as count FROM characters').get();
        return {
            players: players.count,
            characters: characters.count
        };
    }
}

// Singleton instance
let dbInstance = null;

export function initDatabase() {
    if (!dbInstance) {
        dbInstance = new GameDatabase();
        dbInstance.init();
    }
    return dbInstance;
}

export function getDatabase() {
    if (!dbInstance) {
        throw new Error('Database not initialized. Call initDatabase() first.');
    }
    return dbInstance;
}

export default GameDatabase;
