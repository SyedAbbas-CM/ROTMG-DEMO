/**
 * ClientBulletManager class handles all bullet creation, updates and rendering
 */
export class ClientBulletManager {
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
    
    /**
     * Render all active bullets
     * @param {CanvasRenderingContext2D} ctx - Canvas rendering context
     * @param {Object} camera - Camera object with x,y position
     */
    render(ctx, camera) {
        // Early return if no bullets
        if (this.activeBullets === 0) return;
        
        // Import constants if needed
        const TILE_SIZE = 24; // Ensure this matches your constant value from constants.js
        
        // Get sprite manager if available
        const spriteManager = window.spriteManager || null;
        
        // Get screen center coordinates
        const screenCenterX = ctx.canvas.width / 2;
        const screenCenterY = ctx.canvas.height / 2;
        
        // Render each active bullet - important to use this.capacity to check all slots
        for (let i = 0; i < this.capacity; i++) {
            // Skip inactive bullets
            if (!this.active[i]) continue;
            
            // Calculate screen coordinates properly using TILE_SIZE
            const screenX = (this.x[i] - camera.x) * TILE_SIZE + screenCenterX;
            const screenY = (this.y[i] - camera.y) * TILE_SIZE + screenCenterY;
            
            // Skip if off-screen (with slightly larger buffer for fast bullets)
            const buffer = 100;
            if (screenX < -buffer || screenX > ctx.canvas.width + buffer || 
                screenY < -buffer || screenY > ctx.canvas.height + buffer) {
                continue;
            }
            
            // Default size for bullets without dimensions
            const width = this.width[i] || 10; // Increased size
            const height = this.height[i] || 10; // Increased size
            
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
                // Fallback to simple circle with outline for better visibility
                ctx.fillStyle = this.ownerId[i] ? '#ffff00' : '#ff3333';
                ctx.beginPath();
                ctx.arc(screenX, screenY, width/2, 0, Math.PI * 2);
                ctx.fill();
                
                // Add white outline for better visibility
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 1;
                ctx.stroke();
            }
            
            // Debug display only when explicitly enabled
            if (window.debugOverlay && window.debugOverlay.enabled && window.debugOverlay.showBulletIDs) {
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
    }
    
    /**
     * Update all active bullets
     * @param {number} deltaTime - Time since last update in seconds
     */
    update(deltaTime) {
        // Early return if no bullets
        if (this.activeBullets === 0) return;
        
        const now = Date.now();
        
        // Update each active bullet
        for (let i = 0; i < this.capacity; i++) {
            // Skip inactive bullets
            if (!this.active[i]) continue;
            
            // Check if bullet has expired
            const elapsed = (now - this.createTime[i]) / 1000;
            if (elapsed >= this.lifetime[i]) {
                // Mark as inactive
                this.active[i] = false;
                this.activeBullets--;
                continue;
            }
            
            // Update position
            this.x[i] += this.vx[i] * deltaTime;
            this.y[i] += this.vy[i] * deltaTime;
        }
        
        // Debug info - update only occasionally to avoid console spam
        if (this.activeBullets > 0 && Math.random() < 0.01) {
            console.log(`Active bullets: ${this.activeBullets}`);
        }
    }
    
    /**
     * Remove a bullet by its ID
     * @param {number} bulletId - The ID of the bullet to remove
     * @returns {boolean} Whether the bullet was successfully removed
     */
    removeBullet(bulletId) {
        // Search for the bullet with the given ID
        for (let i = 0; i < this.capacity; i++) {
            if (this.active[i] && this.ids[i] === bulletId) {
                // Mark as inactive
                this.active[i] = false;
                this.activeBullets--;
                
                if (this.debug) {
                    console.log(`Removed bullet with ID: ${bulletId}`);
                }
                
                return true;
            }
        }
        
        // Bullet not found
        return false;
    }
    
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
    }
} 