// src/routes/hotReloadRoutes.js
import express from 'express';
import { loadCapabilities } from '../registry/DirectoryLoader.js';
import { registry } from '../registry/index.js';

const router = express.Router();

router.post('/api/hot-reload', async (_req, res) => {
  try {
    const payload = await loadCapabilities();
    registry.merge(payload);
    res.json({ ok: true, loaded: Object.keys(payload.validators).length });
  } catch (err) {
    console.error('[HotReload] failed', err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

export default router; 