// File: /src/routes/mapRoutes.js
const express = require('express');
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

module.exports = router;
