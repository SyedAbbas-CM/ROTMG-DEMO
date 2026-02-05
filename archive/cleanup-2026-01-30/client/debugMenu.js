createButton('Toggle Debug Lines') {
    gameState.debug.showDebugLines = !gameState.debug.showDebugLines;
},

createButton('Visualize Map') {
    if (gameState.clientMapManager) {
        gameState.clientMapManager.visualizeMap();
    } else {
        console.error('Client map manager not available');
    }
}, 