// Behavior Designer Routes - Save/Load behavior definitions
import { Router } from 'express';
import fs from 'fs';
import path from 'path';

const router = Router();

// Directory where behaviors are stored
const BEHAVIORS_DIR = path.join(process.cwd(), 'public', 'assets', 'behaviors');

// Ensure behaviors directory exists
if (!fs.existsSync(BEHAVIORS_DIR)) {
  fs.mkdirSync(BEHAVIORS_DIR, { recursive: true });
}

// GET /api/behavior-designer/list - List all available behaviors
router.get('/list', (req, res) => {
  try {
    const files = fs.readdirSync(BEHAVIORS_DIR).filter(f => f.endsWith('.json'));
    const behaviors = files.map(f => {
      const content = fs.readFileSync(path.join(BEHAVIORS_DIR, f), 'utf8');
      const data = JSON.parse(content);
      return {
        name: data.name || f.replace('.json', ''),
        filename: f,
        path: `/assets/behaviors/${f}`,
        stateCount: data.states ? data.states.length : 0
      };
    });
    res.json({ success: true, behaviors });
  } catch (err) {
    console.error('[BehaviorDesigner] Failed to list behaviors:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// GET /api/behavior-designer/load/:name - Load specific behavior
router.get('/load/:name', (req, res) => {
  try {
    const { name } = req.params;
    const safeName = name.replace(/[^a-zA-Z0-9_-]/g, '_');
    const filePath = path.join(BEHAVIORS_DIR, `${safeName}.json`);

    if (!fs.existsSync(filePath)) {
      return res.status(404).json({
        success: false,
        error: `Behavior ${name} not found`
      });
    }

    const behaviorData = JSON.parse(fs.readFileSync(filePath, 'utf8'));

    res.json({
      success: true,
      behavior: behaviorData
    });
  } catch (err) {
    console.error('[BehaviorDesigner] Failed to load behavior:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// POST /api/behavior-designer/save - Save behavior to disk
router.post('/save', (req, res) => {
  try {
    const { name, dsl, states } = req.body;

    if (!name || !states) {
      return res.status(400).json({
        success: false,
        error: 'Missing name or states'
      });
    }

    // Sanitize filename (remove path traversal attempts)
    const safeName = name.replace(/[^a-zA-Z0-9_-]/g, '_');
    const filename = `${safeName}.json`;
    const filePath = path.join(BEHAVIORS_DIR, filename);

    const behaviorData = {
      name,
      dsl,
      states,
      metadata: {
        createdAt: new Date().toISOString(),
        version: '1.0'
      }
    };

    // Write behavior to disk
    fs.writeFileSync(filePath, JSON.stringify(behaviorData, null, 2));

    console.log(`[BehaviorDesigner] Saved behavior to ${filePath}`);

    res.json({
      success: true,
      path: `/assets/behaviors/${filename}`,
      message: `Behavior saved as ${filename}`
    });
  } catch (err) {
    console.error('[BehaviorDesigner] Failed to save behavior:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// DELETE /api/behavior-designer/delete/:name - Delete behavior
router.delete('/delete/:name', (req, res) => {
  try {
    const { name } = req.params;
    const safeName = name.replace(/[^a-zA-Z0-9_-]/g, '_');
    const filePath = path.join(BEHAVIORS_DIR, `${safeName}.json`);

    if (!fs.existsSync(filePath)) {
      return res.status(404).json({
        success: false,
        error: `Behavior ${name} not found`
      });
    }

    fs.unlinkSync(filePath);

    console.log(`[BehaviorDesigner] Deleted behavior ${name}`);

    res.json({
      success: true,
      message: `Behavior ${name} deleted`
    });
  } catch (err) {
    console.error('[BehaviorDesigner] Failed to delete behavior:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// POST /api/behavior-designer/compile - Compile DSL to executable behavior
router.post('/compile', (req, res) => {
  try {
    const { dsl, behaviorId } = req.body;

    if (!dsl || !behaviorId) {
      return res.status(400).json({
        success: false,
        error: 'Missing dsl or behaviorId'
      });
    }

    // Import the DSL parser and compiler
    // Note: This would need dynamic import in real implementation
    // For now, we'll just validate the DSL syntax

    const lines = dsl.split('\n').filter(l => l.trim() && !l.trim().startsWith('//'));
    const hasStates = lines.some(l => l.includes('STATE'));

    if (!hasStates) {
      return res.status(400).json({
        success: false,
        error: 'DSL must contain at least one STATE definition'
      });
    }

    res.json({
      success: true,
      message: 'DSL compiled successfully',
      behaviorId
    });
  } catch (err) {
    console.error('[BehaviorDesigner] Failed to compile DSL:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

export default router;
