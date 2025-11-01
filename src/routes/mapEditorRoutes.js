// Map Editor Routes - Save/Load maps to disk
import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const router = Router();

// Directory where maps are stored
const MAPS_DIR = path.join(process.cwd(), 'public', 'maps');

// Ensure maps directory exists
if (!fs.existsSync(MAPS_DIR)) {
  fs.mkdirSync(MAPS_DIR, { recursive: true });
}

// GET /api/map-editor/maps - List all available maps
router.get('/maps', (req, res) => {
  try {
    const files = fs.readdirSync(MAPS_DIR).filter(f => f.endsWith('.json'));
    const maps = files.map(f => ({
      name: f.replace('.json', ''),
      filename: f,
      path: `/maps/${f}`
    }));
    res.json({ success: true, maps });
  } catch (err) {
    console.error('[MapEditor] Failed to list maps:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// POST /api/map-editor/save - Save map to disk
router.post('/save', (req, res) => {
  try {
    const { mapName, mapData } = req.body;

    if (!mapName || !mapData) {
      return res.status(400).json({
        success: false,
        error: 'Missing mapName or mapData'
      });
    }

    // Sanitize filename (remove path traversal attempts)
    const safeName = mapName.replace(/[^a-zA-Z0-9_-]/g, '_');
    const filename = `${safeName}.json`;
    const filePath = path.join(MAPS_DIR, filename);

    // Write map to disk
    fs.writeFileSync(filePath, JSON.stringify(mapData, null, 2));

    console.log(`[MapEditor] Saved map to ${filePath}`);

    res.json({
      success: true,
      path: `/maps/${filename}`,
      message: `Map saved as ${filename}`
    });
  } catch (err) {
    console.error('[MapEditor] Failed to save map:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// GET /api/map-editor/load/:name - Load specific map
router.get('/load/:name', (req, res) => {
  try {
    const { name } = req.params;
    const safeName = name.replace(/[^a-zA-Z0-9_-]/g, '_');
    const filePath = path.join(MAPS_DIR, `${safeName}.json`);

    if (!fs.existsSync(filePath)) {
      return res.status(404).json({
        success: false,
        error: `Map ${name} not found`
      });
    }

    const mapData = JSON.parse(fs.readFileSync(filePath, 'utf8'));

    res.json({
      success: true,
      mapName: name,
      mapData
    });
  } catch (err) {
    console.error('[MapEditor] Failed to load map:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// DELETE /api/map-editor/delete/:name - Delete map
router.delete('/delete/:name', (req, res) => {
  try {
    const { name } = req.params;
    const safeName = name.replace(/[^a-zA-Z0-9_-]/g, '_');
    const filePath = path.join(MAPS_DIR, `${safeName}.json`);

    if (!fs.existsSync(filePath)) {
      return res.status(404).json({
        success: false,
        error: `Map ${name} not found`
      });
    }

    fs.unlinkSync(filePath);

    console.log(`[MapEditor] Deleted map ${name}`);

    res.json({
      success: true,
      message: `Map ${name} deleted`
    });
  } catch (err) {
    console.error('[MapEditor] Failed to delete map:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

export default router;
