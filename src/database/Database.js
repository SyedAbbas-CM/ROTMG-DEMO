/**
 * Database.js - SQLite database for player persistence
 * Uses sql.js (pure JS SQLite) for cross-platform compatibility
 */

import initSqlJs from 'sql.js';
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
        this.SQL = null;
    }

    /**
     * Initialize database connection and create tables
     */
    async init() {
        // Ensure data directory exists
        const dataDir = path.dirname(DB_PATH);
        if (!fs.existsSync(dataDir)) {
            fs.mkdirSync(dataDir, { recursive: true });
        }

        // Initialize SQL.js
        this.SQL = await initSqlJs();

        // Load existing database or create new one
        if (fs.existsSync(DB_PATH)) {
            const fileBuffer = fs.readFileSync(DB_PATH);
            this.db = new this.SQL.Database(fileBuffer);
            console.log('[Database] Loaded existing database');
        } else {
            this.db = new this.SQL.Database();
            console.log('[Database] Created new database');
        }

        this.createTables();
        this.save(); // Save initial structure
        console.log('[Database] Initialized at', DB_PATH);
        return this;
    }

    /**
     * Save database to disk
     */
    save() {
        if (!this.db) return;
        const data = this.db.export();
        const buffer = Buffer.from(data);
        fs.writeFileSync(DB_PATH, buffer);
    }

    /**
     * Create database tables if they don't exist
     */
    createTables() {
        // Players table - account info
        this.db.run(`
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
        this.db.run(`
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
        this.db.run(`
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
        this.db.run(`
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
        this.db.run(`CREATE INDEX IF NOT EXISTS idx_characters_player ON characters(player_id)`);
        this.db.run(`CREATE INDEX IF NOT EXISTS idx_inventory_character ON inventory(character_id)`);
        this.db.run(`CREATE INDEX IF NOT EXISTS idx_equipment_character ON equipment(character_id)`);
        this.db.run(`CREATE INDEX IF NOT EXISTS idx_players_email ON players(email)`);

        console.log('[Database] Tables created/verified');
    }

    // Helper to run query and get first result as object
    _get(sql, params = []) {
        const stmt = this.db.prepare(sql);
        stmt.bind(params);
        if (stmt.step()) {
            const row = stmt.getAsObject();
            stmt.free();
            return row;
        }
        stmt.free();
        return null;
    }

    // Helper to run query and get all results
    _all(sql, params = []) {
        const results = [];
        const stmt = this.db.prepare(sql);
        stmt.bind(params);
        while (stmt.step()) {
            results.push(stmt.getAsObject());
        }
        stmt.free();
        return results;
    }

    // Helper to run a query that modifies data
    _run(sql, params = []) {
        this.db.run(sql, params);
        this.save(); // Auto-save after modifications
        return {
            lastInsertRowid: this.db.exec("SELECT last_insert_rowid()")[0]?.values[0][0] || 0,
            changes: this.db.getRowsModified()
        };
    }

    // ==================== PLAYER OPERATIONS ====================

    /**
     * Create a new player account
     */
    createPlayer(name, email, passwordHash = null) {
        try {
            const result = this._run(
                `INSERT INTO players (name, email, password_hash, last_login) VALUES (?, ?, ?, datetime('now'))`,
                [name, email, passwordHash]
            );
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
        return this._get('SELECT * FROM players WHERE name = ?', [name]);
    }

    /**
     * Get player by email
     */
    getPlayerByEmail(email) {
        return this._get('SELECT * FROM players WHERE email = ?', [email]);
    }

    /**
     * Get player by ID
     */
    getPlayerById(id) {
        return this._get('SELECT * FROM players WHERE id = ?', [id]);
    }

    /**
     * Update player's last login time
     */
    updateLastLogin(playerId) {
        this._run("UPDATE players SET last_login = datetime('now') WHERE id = ?", [playerId]);
    }

    // ==================== CHARACTER OPERATIONS ====================

    /**
     * Create a new character for a player
     */
    createCharacter(playerId, name, characterClass = 'warrior') {
        const classDefaults = this.getClassDefaults(characterClass);

        const result = this._run(`
            INSERT INTO characters (
                player_id, name, class, health, max_health, mana, max_mana,
                attack, defense, speed, dexterity, vitality, wisdom, last_played
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        `, [
            playerId, name, characterClass,
            classDefaults.health, classDefaults.maxHealth,
            classDefaults.mana, classDefaults.maxMana,
            classDefaults.attack, classDefaults.defense,
            classDefaults.speed, classDefaults.dexterity,
            classDefaults.vitality, classDefaults.wisdom
        ]);

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
        return this._get('SELECT * FROM characters WHERE id = ?', [id]);
    }

    /**
     * Get all characters for a player
     */
    getCharactersByPlayerId(playerId) {
        return this._all('SELECT * FROM characters WHERE player_id = ? ORDER BY last_played DESC', [playerId]);
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

        fields.push("last_played = datetime('now')");
        values.push(characterId);

        this._run(`UPDATE characters SET ${fields.join(', ')} WHERE id = ?`, values);
        return true;
    }

    /**
     * Save character position (called frequently)
     */
    saveCharacterPosition(characterId, x, y, worldId) {
        this._run("UPDATE characters SET x = ?, y = ?, world_id = ?, last_played = datetime('now') WHERE id = ?",
            [x, y, worldId, characterId]);
    }

    /**
     * Mark character as dead
     */
    killCharacter(characterId) {
        this._run("UPDATE characters SET is_dead = 1, last_played = datetime('now') WHERE id = ?", [characterId]);
    }

    /**
     * Delete character permanently (permadeath)
     * Also deletes associated inventory and equipment
     */
    deleteCharacter(characterId) {
        // Delete inventory first (foreign key)
        this._run('DELETE FROM inventory WHERE character_id = ?', [characterId]);
        // Delete equipment
        this._run('DELETE FROM equipment WHERE character_id = ?', [characterId]);
        // Delete character
        this._run('DELETE FROM characters WHERE id = ?', [characterId]);
        console.log(`[Database] Deleted character ${characterId} and all items`);
    }

    // ==================== INVENTORY OPERATIONS ====================

    /**
     * Get character's inventory
     */
    getInventory(characterId) {
        return this._all('SELECT * FROM inventory WHERE character_id = ? ORDER BY slot', [characterId]);
    }

    /**
     * Set item in inventory slot
     */
    setInventorySlot(characterId, slot, itemId, quantity = 1, tier = 0, enchantments = null) {
        this._run(`
            INSERT OR REPLACE INTO inventory (character_id, slot, item_id, quantity, tier, enchantments)
            VALUES (?, ?, ?, ?, ?, ?)
        `, [characterId, slot, itemId, quantity, tier, enchantments ? JSON.stringify(enchantments) : null]);
    }

    /**
     * Clear inventory slot
     */
    clearInventorySlot(characterId, slot) {
        this._run('DELETE FROM inventory WHERE character_id = ? AND slot = ?', [characterId, slot]);
    }

    /**
     * Get equipped items
     */
    getEquipment(characterId) {
        return this._all('SELECT * FROM equipment WHERE character_id = ?', [characterId]);
    }

    /**
     * Equip item
     */
    equipItem(characterId, slot, itemId, tier = 0, enchantments = null) {
        this._run(`
            INSERT OR REPLACE INTO equipment (character_id, slot, item_id, tier, enchantments)
            VALUES (?, ?, ?, ?, ?)
        `, [characterId, slot, itemId, tier, enchantments ? JSON.stringify(enchantments) : null]);
    }

    /**
     * Unequip item from slot
     */
    unequipItem(characterId, slot) {
        this._run('DELETE FROM equipment WHERE character_id = ? AND slot = ?', [characterId, slot]);
    }

    // ==================== UTILITY ====================

    /**
     * Close database connection
     */
    close() {
        if (this.db) {
            this.save();
            this.db.close();
            console.log('[Database] Connection closed');
        }
    }

    /**
     * Get database stats
     */
    getStats() {
        const players = this._get('SELECT COUNT(*) as count FROM players');
        const characters = this._get('SELECT COUNT(*) as count FROM characters');
        return {
            players: players?.count || 0,
            characters: characters?.count || 0
        };
    }
}

// Singleton instance
let dbInstance = null;

export async function initDatabase() {
    if (!dbInstance) {
        dbInstance = new GameDatabase();
        await dbInstance.init();
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
