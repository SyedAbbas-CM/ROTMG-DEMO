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
      // Left-click to pick up item
      canvas.addEventListener('click', (e)=>{
        e.stopPropagation();
        // Guard: prevent spam if inventory full
        const gm = window.gameManager;
        if(gm && gm.inventory && gm.inventory.filter(i=>i==null).length===0){
          showToast('Inventory full');
          return;
        }
        const net = window.networkManager;
        if(net && typeof net.sendPickupItem === 'function'){
          net.sendPickupItem(this.bag.id, itemId);
        }
        // util toast
        function showToast(msg){
          if(window.showToast){ window.showToast(msg); return; }
          let div = document.getElementById('toastContainer');
          if(!div){
            div = document.createElement('div');
            div.id = 'toastContainer';
            Object.assign(div.style,{position:'fixed',bottom:'20px',left:'50%',transform:'translateX(-50%)',zIndex:9999,fontFamily:'sans-serif'});
            document.body.appendChild(div);
          }
          const toast = document.createElement('div');
          toast.textContent = msg;
          Object.assign(toast.style,{background:'rgba(0,0,0,0.8)',color:'#fff',padding:'6px 10px',marginTop:'4px',borderRadius:'4px',fontSize:'14px',opacity:'1',transition:'opacity 0.5s'});
          div.appendChild(toast);
          setTimeout(()=>{ toast.style.opacity='0'; setTimeout(()=>toast.remove(),500); }, 2000);
          window.showToast=showToast;
        }
      });
      canvas.addEventListener('dragstart', (e)=>{
        e.dataTransfer.setData('text/plain', itemId);
      });
      grid.appendChild(canvas);
    });
    this.element.appendChild(grid);
  }
} 