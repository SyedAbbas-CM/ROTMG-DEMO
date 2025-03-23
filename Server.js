// server.js

const express = require('express');
const http = require('http');
const path = require('path');
const fs = require('fs');
const mapRoutes = require('./routes/mapRoutes');
const spriteRoutes = require('./routes/spriteRoutes');

// Import GameManager (the main game coordinator)
const GameManager = require('./src/GameManager').default;

// Create Express app and HTTP server
const app = express();
const server = http.createServer(app);

// Set up middleware
app.use(express.json());
app.use(express.static('public'));

// Route handlers
app.use('/map', mapRoutes);
app.use('/assets/spritesheets', spriteRoutes);

// New route to serve the WASM file
app.get('/wasm/collision.wasm', (req, res) => {
  const wasmPath = path.join(__dirname, 'public', 'wasm', 'collision.wasm');
  
  // Set proper MIME type for WASM
  res.setHeader('Content-Type', 'application/wasm');
  
  // Check if file exists
  if (fs.existsSync(wasmPath)) {
    res.sendFile(wasmPath);
  } else {
    res.status(404).send('WASM file not found');
  }
});

// Create and initialize GameManager
const gameManager = new GameManager(server);

// Start the game loop
gameManager.start();

// Listen on port 3000
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\nShutting down server...');
  gameManager.cleanup();
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});