#!/usr/bin/env node

/**
 * Simple HTTP server for the overworld demo
 * Run with: node server.js
 * Then open: http://localhost:3001
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 3001;

// MIME types
const mimeTypes = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml'
};

const server = http.createServer((req, res) => {
    let filePath = req.url === '/' ? '/index.html' : req.url;
    
    // Handle requests for main game assets
    if (req.url.startsWith('/assets/')) {
        const mainGamePath = path.join(__dirname, '..', 'public', req.url);
        
        fs.access(mainGamePath, fs.constants.F_OK, (err) => {
            if (!err) {
                // File exists in main game, serve it
                const ext = path.extname(mainGamePath).toLowerCase();
                const contentType = mimeTypes[ext] || 'application/octet-stream';
                
                res.setHeader('Access-Control-Allow-Origin', '*');
                res.setHeader('Access-Control-Allow-Methods', 'GET');
                res.setHeader('Content-Type', contentType);
                
                const stream = fs.createReadStream(mainGamePath);
                stream.on('error', (error) => {
                    res.writeHead(500);
                    res.end('Server error accessing main game asset');
                });
                stream.pipe(res);
                return;
            }
            
            // Fall through to normal handling if not found in main game
            handleNormalFile();
        });
    } else {
        handleNormalFile();
    }
    
    function handleNormalFile() {
        // Resolve to current directory
        filePath = path.join(__dirname, filePath);
        
        // Check if file exists
        fs.access(filePath, fs.constants.F_OK, (err) => {
            if (err) {
                res.writeHead(404);
                res.end('File not found');
                return;
            }
            
            // Get file extension for MIME type
            const ext = path.extname(filePath).toLowerCase();
            const contentType = mimeTypes[ext] || 'application/octet-stream';
            
            // Set CORS headers for local development
            res.setHeader('Access-Control-Allow-Origin', '*');
            res.setHeader('Access-Control-Allow-Methods', 'GET');
            res.setHeader('Content-Type', contentType);
            
            // Stream the file
            const stream = fs.createReadStream(filePath);
            stream.on('error', (error) => {
                res.writeHead(500);
                res.end('Server error');
            });
            
            stream.pipe(res);
        });
    }
});

server.listen(PORT, () => {
    console.log(`🌍 Overworld Demo Server running at:`);
    console.log(`   http://localhost:${PORT}`);
    console.log(`\n📁 Serving files from: ${__dirname}`);
    console.log(`\n🎮 Controls:`);
    console.log(`   WASD - Move around`);
    console.log(`   Mouse Wheel - Zoom`);
    console.log(`   Click - Teleport to position`);
    console.log(`\n⭐ Features:`);
    console.log(`   • 5000x5000 tile procedural world`);
    console.log(`   • Chunk-based loading (100x100 tiles per chunk)`);
    console.log(`   • 6 terrain types with sprite rendering`);
    console.log(`   • Memory management (max 25 chunks loaded)`);
    console.log(`\n🚀 Ready for testing!`);
});

server.on('error', (error) => {
    if (error.code === 'EADDRINUSE') {
        console.error(`❌ Port ${PORT} is already in use. Try a different port or stop the conflicting process.`);
    } else {
        console.error('❌ Server error:', error.message);
    }
    process.exit(1);
});