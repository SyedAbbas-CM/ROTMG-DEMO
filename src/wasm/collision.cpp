/**
 * collision.cpp
 * WebAssembly module for optimized collision detection
 * Compile with Emscripten:
 * emcc collision.cpp -O3 -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s EXPORTED_FUNCTIONS=['_testAABBCollision','_detectCollisions'] -o public/wasm/collision.wasm
 */
#include <emscripten.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

extern "C" {
  // Basic AABB collision check between two rectangles
  EMSCRIPTEN_KEEPALIVE
  bool testAABBCollision(
    float ax, float ay, float awidth, float aheight,
    float bx, float by, float bwidth, float bheight
  ) {
    return (
      ax < bx + bwidth &&
      ax + awidth > bx &&
      ay < by + bheight &&
      ay + aheight > by
    );
  }

  // Optimized grid-based collision detection for bullets vs enemies
  EMSCRIPTEN_KEEPALIVE
  int detectCollisions(
    // Bullet data arrays (Structure of Arrays format)
    float* bulletX, float* bulletY, 
    float* bulletWidth, float* bulletHeight,
    int32_t bulletCount,
    
    // Enemy data arrays (Structure of Arrays format)
    float* enemyX, float* enemyY,
    float* enemyWidth, float* enemyHeight,
    int32_t enemyCount,
    
    // Grid parameters
    float gridCellSize,
    
    // Output collision pairs [bulletIdx0, enemyIdx0, bulletIdx1, enemyIdx1, ...]
    int32_t* collisionPairs,
    int32_t maxCollisions
  ) {
    int32_t collisionCount = 0;
    
    // For each bullet, check potential collisions
    for (int32_t i = 0; i < bulletCount && collisionCount < maxCollisions; i++) {
      // Calculate grid cells this bullet might intersect
      int32_t minCellX = (int32_t)(bulletX[i] / gridCellSize);
      int32_t minCellY = (int32_t)(bulletY[i] / gridCellSize);
      int32_t maxCellX = (int32_t)((bulletX[i] + bulletWidth[i]) / gridCellSize);
      int32_t maxCellY = (int32_t)((bulletY[i] + bulletHeight[i]) / gridCellSize);
      
      // Check each enemy for possible collision
      for (int32_t j = 0; j < enemyCount && collisionCount < maxCollisions; j++) {
        // Grid-based early rejection
        int32_t enemyMinCellX = (int32_t)(enemyX[j] / gridCellSize);
        int32_t enemyMinCellY = (int32_t)(enemyY[j] / gridCellSize);
        int32_t enemyMaxCellX = (int32_t)((enemyX[j] + enemyWidth[j]) / gridCellSize);
        int32_t enemyMaxCellY = (int32_t)((enemyY[j] + enemyHeight[j]) / gridCellSize);
        
        // Check for grid cell overlap - early rejection
        bool cellsOverlap = !(
          maxCellX < enemyMinCellX || 
          minCellX > enemyMaxCellX || 
          maxCellY < enemyMinCellY || 
          minCellY > enemyMaxCellY
        );
        
        // If overlapping cells, do precise AABB check
        if (cellsOverlap && testAABBCollision(
          bulletX[i], bulletY[i], bulletWidth[i], bulletHeight[i],
          enemyX[j], enemyY[j], enemyWidth[j], enemyHeight[j]
        )) {
          // Store collision pair indices
          collisionPairs[collisionCount * 2] = i;
          collisionPairs[collisionCount * 2 + 1] = j;
          collisionCount++;
        }
      }
    }
    
    return collisionCount;
  }
}