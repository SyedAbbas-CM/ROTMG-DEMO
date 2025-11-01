// Enemy Editor Routes - Save/Update enemies to enemies.json database
import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import { entityDatabase } from '../assets/EntityDatabase.js';

const router = Router();

// Path to enemies database file
const ENEMIES_DB_PATH = path.join(process.cwd(), 'public', 'assets', 'entities', 'enemies.json');

// Helper: Read enemies from disk
function readEnemiesFromDisk() {
  try {
    if (!fs.existsSync(ENEMIES_DB_PATH)) {
      return [];
    }
    const content = fs.readFileSync(ENEMIES_DB_PATH, 'utf8');
    return JSON.parse(content);
  } catch (err) {
    console.error('[EnemyEditor] Failed to read enemies.json:', err);
    return [];
  }
}

// Helper: Write enemies to disk
function writeEnemiesToDisk(enemies) {
  try {
    const dir = path.dirname(ENEMIES_DB_PATH);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(ENEMIES_DB_PATH, JSON.stringify(enemies, null, 2));

    // Reload into EntityDatabase
    entityDatabase.groups.enemies.clear();
    enemies.forEach(enemy => {
      if (enemy && enemy.id) {
        entityDatabase.groups.enemies.set(enemy.id, enemy);
      }
    });

    console.log(`[EnemyEditor] Saved ${enemies.length} enemies to database`);
    return true;
  } catch (err) {
    console.error('[EnemyEditor] Failed to write enemies.json:', err);
    return false;
  }
}

// GET /api/enemy-editor/list - List all enemies
router.get('/list', (req, res) => {
  try {
    const enemies = readEnemiesFromDisk();
    res.json({ success: true, enemies });
  } catch (err) {
    console.error('[EnemyEditor] Failed to list enemies:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// GET /api/enemy-editor/get/:id - Get specific enemy
router.get('/get/:id', (req, res) => {
  try {
    const { id } = req.params;
    const enemies = readEnemiesFromDisk();
    const enemy = enemies.find(e => e.id === id);

    if (!enemy) {
      return res.status(404).json({
        success: false,
        error: `Enemy ${id} not found`
      });
    }

    res.json({ success: true, enemy });
  } catch (err) {
    console.error('[EnemyEditor] Failed to get enemy:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// POST /api/enemy-editor/save - Save/update enemy
router.post('/save', (req, res) => {
  try {
    const { enemy } = req.body;

    if (!enemy || !enemy.id) {
      return res.status(400).json({
        success: false,
        error: 'Missing enemy or enemy.id'
      });
    }

    const enemies = readEnemiesFromDisk();
    const existingIndex = enemies.findIndex(e => e.id === enemy.id);

    if (existingIndex >= 0) {
      // Update existing
      enemies[existingIndex] = enemy;
      console.log(`[EnemyEditor] Updated enemy: ${enemy.id}`);
    } else {
      // Add new
      enemies.push(enemy);
      console.log(`[EnemyEditor] Created new enemy: ${enemy.id}`);
    }

    const success = writeEnemiesToDisk(enemies);

    if (success) {
      res.json({
        success: true,
        message: `Enemy ${enemy.id} saved`,
        enemy
      });
    } else {
      res.status(500).json({
        success: false,
        error: 'Failed to write to database'
      });
    }
  } catch (err) {
    console.error('[EnemyEditor] Failed to save enemy:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// DELETE /api/enemy-editor/delete/:id - Delete enemy
router.delete('/delete/:id', (req, res) => {
  try {
    const { id } = req.params;
    const enemies = readEnemiesFromDisk();
    const filteredEnemies = enemies.filter(e => e.id !== id);

    if (enemies.length === filteredEnemies.length) {
      return res.status(404).json({
        success: false,
        error: `Enemy ${id} not found`
      });
    }

    const success = writeEnemiesToDisk(filteredEnemies);

    if (success) {
      res.json({
        success: true,
        message: `Enemy ${id} deleted`
      });
    } else {
      res.status(500).json({
        success: false,
        error: 'Failed to write to database'
      });
    }
  } catch (err) {
    console.error('[EnemyEditor] Failed to delete enemy:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// POST /api/enemy-editor/bulk-save - Save multiple enemies at once
router.post('/bulk-save', (req, res) => {
  try {
    const { enemies } = req.body;

    if (!Array.isArray(enemies)) {
      return res.status(400).json({
        success: false,
        error: 'enemies must be an array'
      });
    }

    const success = writeEnemiesToDisk(enemies);

    if (success) {
      res.json({
        success: true,
        message: `Saved ${enemies.length} enemies`,
        count: enemies.length
      });
    } else {
      res.status(500).json({
        success: false,
        error: 'Failed to write to database'
      });
    }
  } catch (err) {
    console.error('[EnemyEditor] Failed to bulk save enemies:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

export default router;
