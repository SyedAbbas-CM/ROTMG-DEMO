/**
 * Initialize the sprite editor
 */
export function initSpriteEditor() {
    const editorContainer = document.getElementById('spriteEditorContainer');
    if (!editorContainer) {
        console.error('Sprite editor container not found');
        return;
    }
    
    // Create a close button
    const closeButton = document.createElement('button');
    closeButton.textContent = 'Close Editor (ESC)';
    closeButton.className = 'sprite-editor-close-button';
    closeButton.style.position = 'absolute';
    closeButton.style.top = '10px';
    closeButton.style.right = '10px';
    closeButton.style.padding = '5px 10px';
    closeButton.style.backgroundColor = '#d33';
    closeButton.style.color = 'white';
    closeButton.style.border = 'none';
    closeButton.style.borderRadius = '3px';
    closeButton.style.cursor = 'pointer';
    
    closeButton.addEventListener('click', closeSpriteEditor);
    editorContainer.appendChild(closeButton);
    
    // Add ESC key handler to close the editor
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && editorContainer.style.display === 'block') {
            closeSpriteEditor();
        }
    });
    
    // ... existing initialization code ...
}

/**
 * Open the sprite editor
 */
export function openSpriteEditor() {
    const editorContainer = document.getElementById('spriteEditorContainer');
    if (!editorContainer) {
        console.error('Sprite editor container not found');
        return;
    }
    
    editorContainer.style.display = 'block';
    
    // Populate sprite sheets dropdown
    populateSpriteSheets();
    
    console.log('Sprite editor opened');
}

/**
 * Close the sprite editor
 */
export function closeSpriteEditor() {
    const editorContainer = document.getElementById('spriteEditorContainer');
    if (!editorContainer) {
        console.error('Sprite editor container not found');
        return;
    }
    
    editorContainer.style.display = 'none';
    
    // Restore focus to the game canvas
    const gameCanvas = document.getElementById('gameCanvas');
    if (gameCanvas) {
        gameCanvas.focus();
    }
    
    console.log('Sprite editor closed');
} 