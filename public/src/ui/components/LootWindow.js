// public/src/ui/components/LootWindow.js
import { UIComponent } from '../UIManager.js';
import { spriteDatabase } from '../../assets/SpriteDatabase.js';

export class LootWindow extends UIComponent {
  constructor(gameState, manager) {
    super(gameState, manager);
    this.bag = null; // bag DTO currently shown
  }

  async init() {
    this.element = document.createElement('div');
    this.element.className = 'loot-window';
    Object.assign(this.element.style, {
      position: 'absolute',
      background: 'rgba(0,0,0,0.85)',
      border: '1px solid #666',
      padding: '4px',
      display: 'none',
      zIndex: 200,
      pointerEvents: 'auto'
    });
    return this.element;
  }

  openForBag(bagDto, screenX, screenY) {
    this.bag = bagDto;
    this.renderContents();
    this.element.style.left = `${screenX}px`;
    this.element.style.top  = `${screenY}px`;
    this.show();
  }

  renderContents() {
    if (!this.bag) return;
    this.element.innerHTML = '';
    const grid = document.createElement('div');
    grid.style.display = 'grid';
    grid.style.gridTemplateColumns = 'repeat(4, 32px)';
    grid.style.gridGap = '2px';
    this.bag.items.forEach(itemId => {
      const itemDef = globalThis.itemManager?.itemDefinitions.get(itemId) || {};
      const spriteName = itemDef.sprite || 'items_sprite_0';
      const sprite = spriteDatabase.get(spriteName);
      const canvas = document.createElement('canvas');
      canvas.width = 32; canvas.height = 32;
      const ctx = canvas.getContext('2d');
      ctx.imageSmoothingEnabled = false;
      if (sprite) {
        ctx.drawImage(sprite.image, sprite.x, sprite.y, sprite.width, sprite.height, 0,0,32,32);
      } else {
        ctx.fillStyle='#AAA'; ctx.fillRect(0,0,32,32);
      }
      canvas.dataset.itemId = itemId;
      canvas.draggable = true;
      canvas.addEventListener('dragstart', (e)=>{
        e.dataTransfer.setData('text/plain', itemId);
      });
      grid.appendChild(canvas);
    });
    this.element.appendChild(grid);
  }
} 