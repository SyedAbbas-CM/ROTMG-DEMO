// File: /src/routes/mapRoutes.js
// NOTE: This route file was using CommonJS in an ESM project. It appears unused by the server.
// Converting to ESM and keeping minimal functionality in case the editor or tools reference it.
import express from 'express';
const router = express.Router();

// Example tile-based map
const mapSize = 100*100;
let map = new Array(mapSize).fill(0);

router.get('/', (req, res) => {
  res.json({ map });
});

router.post('/change', (req, res) => {
  const { location, block } = req.body;
  if (location >= 0 && location < map.length) {
    map[location] = block;
    return res.json({ success: true });
  }
  res.json({ success: false });
});

export default router;
