/**
 * Render all active bullets
 * @param {CanvasRenderingContext2D} ctx - Canvas rendering context
 * @param {Object} camera - Camera object with x,y position
 */
render(ctx, camera) {
    // Early return if no bullets
    if (this.activeBullets === 0) return;
    
    // Get sprite manager if available
    const spriteManager = window.spriteManager || null;
    
    // Render each active bullet
    for (let i = 0; i < this.activeBullets; i++) {
        // Skip inactive bullets
        if (!this.active[i]) continue;
        
        // Get screen coordinates
        const screenX = this.x[i] - camera.x;
        const screenY = this.y[i] - camera.y;
        
        // Skip if off-screen
        if (screenX < -50 || screenX > ctx.canvas.width + 50 || 
            screenY < -50 || screenY > ctx.canvas.height + 50) {
            continue;
        }
        
        // Default size for bullets without dimensions
        const width = this.width[i] || 5;
        const height = this.height[i] || 5;
        
        // Check if we have sprite information
        if (spriteManager && this.spriteSheet[i]) {
            // Render with sprite
            spriteManager.drawSprite(
                ctx,
                this.spriteSheet[i],
                this.spriteX[i],
                this.spriteY[i],
                screenX - width/2,
                screenY - height/2,
                width,
                height
            );
        } else {
            // Fallback to simple circle
            ctx.fillStyle = this.ownerId[i] ? '#ffff00' : '#ff0000';
            ctx.beginPath();
            ctx.arc(screenX, screenY, width/2, 0, Math.PI * 2);
            ctx.fill();
        }
        
        // Debug display
        if (window.debugOverlay && window.debugOverlay.enabled) {
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.strokeRect(
                screenX - width/2,
                screenY - height/2,
                width,
                height
            );
            
            ctx.fillStyle = 'white';
            ctx.font = '8px Arial';
            ctx.fillText(`${this.ids[i]}`, screenX, screenY - 10);
        }
    }
};

/**
 * Add a bullet to the manager
 * @param {Object} bullet - Bullet data
 * @returns {number} Bullet ID
 */
addBullet(bullet) {
    // Find an available slot
    let index = -1;
    for (let i = 0; i < this.capacity; i++) {
        if (!this.active[i]) {
            index = i;
            break;
        }
    }
    
    // If no slots available, return error
    if (index === -1) {
        console.error('Bullet capacity exceeded');
        return -1;
    }
    
    // Generate ID
    const id = this.nextId++;
    
    // Store bullet data
    this.ids[index] = id;
    this.x[index] = bullet.x;
    this.y[index] = bullet.y;
    this.vx[index] = bullet.vx;
    this.vy[index] = bullet.vy;
    this.lifetime[index] = bullet.lifetime || 3.0;
    this.createTime[index] = Date.now();
    this.active[index] = true;
    this.ownerId[index] = bullet.ownerId || 0;
    this.damage[index] = bullet.damage || 10;
    this.width[index] = bullet.width || 5;
    this.height[index] = bullet.height || 5;
    
    // Store sprite info if provided
    this.spriteSheet[index] = bullet.spriteSheet || null;
    this.spriteX[index] = bullet.spriteX || 0;
    this.spriteY[index] = bullet.spriteY || 0;
    this.spriteWidth[index] = bullet.spriteWidth || this.width[index];
    this.spriteHeight[index] = bullet.spriteHeight || this.height[index];
    
    this.activeBullets++;
    
    return id;
};

/**
 * Create a new BulletManager
 * @param {number} capacity - Maximum number of bullets to track
 */
constructor(capacity = 1000) {
    this.capacity = capacity;
    
    // Initialize arrays
    this.active = new Array(capacity).fill(false);
    this.ids = new Array(capacity).fill(0);
    this.x = new Float32Array(capacity);
    this.y = new Float32Array(capacity);
    this.vx = new Float32Array(capacity);
    this.vy = new Float32Array(capacity);
    this.lifetime = new Float32Array(capacity);
    this.createTime = new Float64Array(capacity);
    this.ownerId = new Array(capacity).fill(0);
    this.damage = new Float32Array(capacity);
    this.width = new Float32Array(capacity);
    this.height = new Float32Array(capacity);
    
    // Add sprite information arrays
    this.spriteSheet = new Array(capacity).fill(null);
    this.spriteX = new Float32Array(capacity);
    this.spriteY = new Float32Array(capacity);
    this.spriteWidth = new Float32Array(capacity);
    this.spriteHeight = new Float32Array(capacity);
    
    this.activeBullets = 0;
    this.nextId = 1;
    
    console.log(`ClientBulletManager initialized with capacity: ${capacity}`);
} 