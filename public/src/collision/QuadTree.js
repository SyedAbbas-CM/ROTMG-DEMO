/**
 * QuadTree.js
 * Spatial partitioning data structure for efficient 2D collision detection
 *
 * Performance Characteristics:
 * - Insert: O(log n) average case
 * - Query: O(log n + k) where k is number of results
 * - Memory: O(n)
 *
 * Used for:
 * - Bullet collision detection (broad-phase)
 * - Player-object collision
 * - Entity-entity collision
 * - Chunk-based spatial filtering
 */

export class QuadTree {
  /**
   * Create a QuadTree
   * @param {Object} bounds - {x, y, width, height} region covered by this node
   * @param {number} capacity - Max entities before subdivision (default: 8)
   * @param {number} maxDepth - Maximum tree depth (default: 8)
   * @param {number} depth - Current depth (internal, default: 0)
   */
  constructor(bounds, capacity = 8, maxDepth = 8, depth = 0) {
    this.bounds = bounds; // {x, y, width, height}
    this.capacity = capacity;
    this.maxDepth = maxDepth;
    this.depth = depth;

    this.entities = []; // Entities stored in this node
    this.divided = false; // Whether this node has been subdivided

    // Child quadrants (created on subdivision)
    this.northeast = null;
    this.northwest = null;
    this.southeast = null;
    this.southwest = null;
  }

  /**
   * Insert an entity into the quad tree
   * @param {Object} entity - Entity with bounds {x, y, width, height} and id
   * @returns {boolean} True if inserted successfully
   */
  insert(entity) {
    // Check if entity intersects this node's bounds
    if (!this._intersects(entity, this.bounds)) {
      return false;
    }

    // If we have capacity and haven't subdivided, add entity here
    if (this.entities.length < this.capacity && !this.divided) {
      this.entities.push(entity);
      return true;
    }

    // If we're at max depth, force add here (prevents infinite subdivision)
    if (this.depth >= this.maxDepth) {
      this.entities.push(entity);
      return true;
    }

    // Otherwise, subdivide if needed and insert into children
    if (!this.divided) {
      this._subdivide();
    }

    // Try to insert into child quadrants
    if (this.northeast.insert(entity)) return true;
    if (this.northwest.insert(entity)) return true;
    if (this.southeast.insert(entity)) return true;
    if (this.southwest.insert(entity)) return true;

    // If entity doesn't fit in any child (spans multiple quadrants),
    // store it in this node
    this.entities.push(entity);
    return true;
  }

  /**
   * Query all entities that intersect with the given bounds
   * @param {Object} range - {x, y, width, height} query bounds
   * @param {Array} found - Accumulator array (optional)
   * @returns {Array} Array of entities intersecting the range
   */
  query(range, found = []) {
    // If range doesn't intersect this node, return empty
    if (!this._intersects(range, this.bounds)) {
      return found;
    }

    // Check entities in this node
    for (const entity of this.entities) {
      if (this._intersects(entity, range)) {
        found.push(entity);
      }
    }

    // If subdivided, query children
    if (this.divided) {
      this.northeast.query(range, found);
      this.northwest.query(range, found);
      this.southeast.query(range, found);
      this.southwest.query(range, found);
    }

    return found;
  }

  /**
   * Query entities at a specific point
   * @param {number} x - X coordinate
   * @param {number} y - Y coordinate
   * @returns {Array} Array of entities at this point
   */
  queryPoint(x, y) {
    return this.query({
      x: x - 0.001,
      y: y - 0.001,
      width: 0.002,
      height: 0.002
    });
  }

  /**
   * Query entities in a circular region
   * @param {number} x - Center X
   * @param {number} y - Center Y
   * @param {number} radius - Radius
   * @returns {Array} Array of entities in circle
   */
  queryCircle(x, y, radius) {
    // Get AABB around circle first (broad-phase)
    const range = {
      x: x - radius,
      y: y - radius,
      width: radius * 2,
      height: radius * 2
    };

    const candidates = this.query(range);

    // Filter by actual circle distance (narrow-phase)
    return candidates.filter(entity => {
      const entityCenterX = entity.x + entity.width / 2;
      const entityCenterY = entity.y + entity.height / 2;
      const dx = entityCenterX - x;
      const dy = entityCenterY - y;
      const distSq = dx * dx + dy * dy;
      return distSq <= radius * radius;
    });
  }

  /**
   * Clear all entities from the tree
   */
  clear() {
    this.entities = [];

    if (this.divided) {
      this.northeast.clear();
      this.northwest.clear();
      this.southeast.clear();
      this.southwest.clear();

      this.northeast = null;
      this.northwest = null;
      this.southeast = null;
      this.southwest = null;

      this.divided = false;
    }
  }

  /**
   * Rebuild the tree from scratch with new entities
   * More efficient than clearing and inserting one by one
   * @param {Array} entities - Array of entities to insert
   */
  rebuild(entities) {
    this.clear();
    for (const entity of entities) {
      this.insert(entity);
    }
  }

  /**
   * Get total number of entities in tree (including children)
   * @returns {number} Total entity count
   */
  size() {
    let count = this.entities.length;

    if (this.divided) {
      count += this.northeast.size();
      count += this.northwest.size();
      count += this.southeast.size();
      count += this.southwest.size();
    }

    return count;
  }

  /**
   * Get statistics about the tree structure
   * @returns {Object} {nodes, depth, entities, avgEntitiesPerLeaf}
   */
  getStats() {
    const stats = {
      nodes: 1,
      maxDepth: this.depth,
      totalEntities: this.entities.length,
      leafNodes: this.divided ? 0 : 1,
      leafEntities: this.divided ? 0 : this.entities.length
    };

    if (this.divided) {
      const neStats = this.northeast.getStats();
      const nwStats = this.northwest.getStats();
      const seStats = this.southeast.getStats();
      const swStats = this.southwest.getStats();

      stats.nodes += neStats.nodes + nwStats.nodes + seStats.nodes + swStats.nodes;
      stats.maxDepth = Math.max(neStats.maxDepth, nwStats.maxDepth, seStats.maxDepth, swStats.maxDepth);
      stats.totalEntities += neStats.totalEntities + nwStats.totalEntities + seStats.totalEntities + swStats.totalEntities;
      stats.leafNodes += neStats.leafNodes + nwStats.leafNodes + seStats.leafNodes + swStats.leafNodes;
      stats.leafEntities += neStats.leafEntities + nwStats.leafEntities + seStats.leafEntities + swStats.leafEntities;
    }

    return stats;
  }

  /**
   * Subdivide this node into 4 child quadrants
   * @private
   */
  _subdivide() {
    const x = this.bounds.x;
    const y = this.bounds.y;
    const w = this.bounds.width / 2;
    const h = this.bounds.height / 2;
    const nextDepth = this.depth + 1;

    this.northeast = new QuadTree(
      { x: x + w, y: y, width: w, height: h },
      this.capacity,
      this.maxDepth,
      nextDepth
    );

    this.northwest = new QuadTree(
      { x: x, y: y, width: w, height: h },
      this.capacity,
      this.maxDepth,
      nextDepth
    );

    this.southeast = new QuadTree(
      { x: x + w, y: y + h, width: w, height: h },
      this.capacity,
      this.maxDepth,
      nextDepth
    );

    this.southwest = new QuadTree(
      { x: x, y: y + h, width: w, height: h },
      this.capacity,
      this.maxDepth,
      nextDepth
    );

    this.divided = true;

    // Redistribute entities to children
    const entitiesToRedistribute = this.entities;
    this.entities = [];

    for (const entity of entitiesToRedistribute) {
      // Try to insert into children
      let inserted = false;
      inserted = inserted || this.northeast.insert(entity);
      inserted = inserted || this.northwest.insert(entity);
      inserted = inserted || this.southeast.insert(entity);
      inserted = inserted || this.southwest.insert(entity);

      // If entity spans multiple quadrants, keep it here
      if (!inserted) {
        this.entities.push(entity);
      }
    }
  }

  /**
   * Check if two AABB bounds intersect
   * @private
   * @param {Object} a - {x, y, width, height}
   * @param {Object} b - {x, y, width, height}
   * @returns {boolean} True if intersecting
   */
  _intersects(a, b) {
    return !(
      a.x + a.width < b.x ||
      a.x > b.x + b.width ||
      a.y + a.height < b.y ||
      a.y > b.y + b.height
    );
  }

  /**
   * Draw debug visualization of quad tree (for development)
   * @param {CanvasRenderingContext2D} ctx - Canvas context
   * @param {number} cameraX - Camera X offset
   * @param {number} cameraY - Camera Y offset
   * @param {number} scale - Scale factor
   */
  debugDraw(ctx, cameraX = 0, cameraY = 0, scale = 1) {
    const screenX = (this.bounds.x - cameraX) * scale;
    const screenY = (this.bounds.y - cameraY) * scale;
    const screenW = this.bounds.width * scale;
    const screenH = this.bounds.height * scale;

    // Draw boundary
    ctx.strokeStyle = this.divided ? 'rgba(0, 255, 0, 0.3)' : 'rgba(255, 0, 0, 0.5)';
    ctx.lineWidth = 1;
    ctx.strokeRect(screenX, screenY, screenW, screenH);

    // Draw entity count
    if (this.entities.length > 0 && !this.divided) {
      ctx.fillStyle = 'yellow';
      ctx.font = '10px monospace';
      ctx.fillText(this.entities.length.toString(), screenX + 5, screenY + 15);
    }

    // Recursively draw children
    if (this.divided) {
      this.northeast.debugDraw(ctx, cameraX, cameraY, scale);
      this.northwest.debugDraw(ctx, cameraX, cameraY, scale);
      this.southeast.debugDraw(ctx, cameraX, cameraY, scale);
      this.southwest.debugDraw(ctx, cameraX, cameraY, scale);
    }
  }
}
