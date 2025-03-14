// File: /src/routes/spriteRoutes.js
const express = require('express');
const router = express.Router();
const fs = require('fs');
const path = require('path');

router.get('/', (req, res) => {
  // read dir etc.
});

router.post('/:sheetName', (req, res) => {
  // save sprite data
});

module.exports = router;
