/**
 * ClientInventoryManager.js
 * Client-side inventory UI and management
 */

import { UIManager } from '../ui/UIManager.js';
import { ItemType, ItemRarity } from '../../src/ItemManager.js';

class ClientInventoryManager {
    constructor() {
        this.inventory = null;
        this.ui = null;
        this.draggedItem = null;
        this.dragStartSlot = -1;
    }
    
    /**
     * Initialize the inventory manager
     * @param {Object} uiManager - UI manager instance
     */
    init(uiManager) {
        this.uiManager = uiManager;
        this._createInventoryUI();
    }
    
    /**
     * Create the inventory UI
     * @private
     */
    _createInventoryUI() {
        // Create inventory container
        this.ui = this.uiManager.createComponent('inventory', {
            width: 400,
            height: 500,
            background: '#1a1a1a',
            border: '2px solid #333',
            borderRadius: '8px',
            padding: '10px',
            display: 'none' // Hidden by default
        });
        
        // Create slots grid
        this.slots = [];
        const grid = this.uiManager.createComponent('inventory-grid', {
            display: 'grid',
            gridTemplateColumns: 'repeat(5, 1fr)',
            gap: '5px',
            margin: '10px'
        });
        
        for (let i = 0; i < 20; i++) {
            const slot = this.uiManager.createComponent(`slot-${i}`, {
                width: '64px',
                height: '64px',
                background: '#333',
                border: '1px solid #444',
                position: 'relative'
            });
            
            // Add drag and drop events
            slot.addEventListener('mousedown', (e) => this._onSlotMouseDown(e, i));
            slot.addEventListener('mouseup', (e) => this._onSlotMouseUp(e, i));
            slot.addEventListener('mouseover', (e) => this._onSlotMouseOver(e, i));
            
            this.slots.push(slot);
            grid.appendChild(slot);
        }
        
        this.ui.appendChild(grid);
        
        // Create close button
        const closeButton = this.uiManager.createComponent('close-button', {
            position: 'absolute',
            top: '10px',
            right: '10px',
            width: '24px',
            height: '24px',
            background: '#ff4444',
            border: 'none',
            borderRadius: '50%',
            cursor: 'pointer'
        });
        
        closeButton.addEventListener('click', () => this.hide());
        this.ui.appendChild(closeButton);
    }
    
    /**
     * Show the inventory
     */
    show() {
        this.ui.style.display = 'block';
    }
    
    /**
     * Hide the inventory
     */
    hide() {
        this.ui.style.display = 'none';
    }
    
    /**
     * Update the inventory UI
     * @param {Object} inventory - Inventory data
     */
    updateInventory(inventory) {
        this.inventory = inventory;
        
        // Update slot contents
        for (let i = 0; i < this.slots.length; i++) {
            const slot = this.slots[i];
            const item = inventory.slots[i];
            
            // Clear slot
            while (slot.firstChild) {
                slot.removeChild(slot.firstChild);
            }
            
            if (item) {
                // Create item element
                const itemElement = this.uiManager.createComponent(`item-${item.id}`, {
                    width: '100%',
                    height: '100%',
                    background: this._getRarityColor(item.rarity),
                    position: 'relative'
                });
                
                // Add item sprite
                const sprite = this.uiManager.createComponent('item-sprite', {
                    width: '48px',
                    height: '48px',
                    margin: '8px',
                    backgroundImage: `url(assets/sprites/items.png)`,
                    backgroundPosition: `${-item.spriteIndex * 48}px 0`
                });
                
                // Add stack size
                if (item.stackSize > 1) {
                    const stackSize = this.uiManager.createComponent('stack-size', {
                        position: 'absolute',
                        bottom: '2px',
                        right: '2px',
                        background: 'rgba(0,0,0,0.5)',
                        color: 'white',
                        padding: '2px 4px',
                        borderRadius: '4px',
                        fontSize: '12px'
                    });
                    stackSize.textContent = item.stackSize;
                    itemElement.appendChild(stackSize);
                }
                
                itemElement.appendChild(sprite);
                slot.appendChild(itemElement);
            }
        }
    }
    
    /**
     * Handle slot mouse down
     * @private
     */
    _onSlotMouseDown(e, slotIndex) {
        const item = this.inventory.slots[slotIndex];
        if (!item) return;
        
        this.draggedItem = item;
        this.dragStartSlot = slotIndex;
        
        // Create drag preview
        this.dragPreview = this.uiManager.createComponent('drag-preview', {
            position: 'fixed',
            pointerEvents: 'none',
            zIndex: 1000
        });
        
        const itemElement = this.slots[slotIndex].firstChild.cloneNode(true);
        this.dragPreview.appendChild(itemElement);
        document.body.appendChild(this.dragPreview);
        
        // Update preview position
        this._updateDragPreview(e);
        
        // Add move event listener
        document.addEventListener('mousemove', this._onMouseMove);
    }
    
    /**
     * Handle slot mouse up
     * @private
     */
    _onSlotMouseUp(e, slotIndex) {
        if (!this.draggedItem) return;
        
        // Remove drag preview
        if (this.dragPreview) {
            document.body.removeChild(this.dragPreview);
            this.dragPreview = null;
        }
        
        // Move item
        if (slotIndex !== this.dragStartSlot) {
            this._moveItem(this.dragStartSlot, slotIndex);
        }
        
        this.draggedItem = null;
        this.dragStartSlot = -1;
        
        // Remove move event listener
        document.removeEventListener('mousemove', this._onMouseMove);
    }
    
    /**
     * Handle slot mouse over
     * @private
     */
    _onSlotMouseOver(e, slotIndex) {
        if (!this.draggedItem) return;
        
        // Highlight slot
        this.slots[slotIndex].style.background = '#444';
    }
    
    /**
     * Handle mouse move
     * @private
     */
    _onMouseMove = (e) => {
        if (!this.dragPreview) return;
        this._updateDragPreview(e);
    };
    
    /**
     * Update drag preview position
     * @private
     */
    _updateDragPreview(e) {
        this.dragPreview.style.left = `${e.clientX - 32}px`;
        this.dragPreview.style.top = `${e.clientY - 32}px`;
    }
    
    /**
     * Move item between slots
     * @private
     */
    _moveItem(fromSlot, toSlot) {
        // Update local copy optimistically
        const temp = this.inventory.slots[fromSlot];
        this.inventory.slots[fromSlot] = this.inventory.slots[toSlot];
        this.inventory.slots[toSlot] = temp;
        this.updateInventory(this.inventory); // re-render
        // Emit to server
        if(window.networkManager && typeof window.networkManager.sendMoveItem==='function'){
          window.networkManager.sendMoveItem(fromSlot,toSlot);
        }
        this.uiManager.emit('inventory-move', { fromSlot, toSlot });
    }
    
    /**
     * Get rarity color
     * @private
     */
    _getRarityColor(rarity) {
        switch (rarity) {
            case ItemRarity.COMMON: return '#ffffff';
            case ItemRarity.UNCOMMON: return '#00ff00';
            case ItemRarity.RARE: return '#0088ff';
            case ItemRarity.EPIC: return '#aa00ff';
            case ItemRarity.LEGENDARY: return '#ffaa00';
            default: return '#ffffff';
        }
    }
}

export { ClientInventoryManager }; 