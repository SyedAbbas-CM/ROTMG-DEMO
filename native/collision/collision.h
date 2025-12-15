#ifndef ROTMG_COLLISION_H
#define ROTMG_COLLISION_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// AABB (Axis-Aligned Bounding Box) structure
typedef struct {
    float x, y;       // Center position
    float w, h;       // Width and height
} AABB;

// Entity structure for collision detection
typedef struct {
    uint32_t id;      // Entity ID
    AABB bounds;      // Bounding box
    uint8_t layer;    // Collision layer (0=enemy, 1-12=player factions)
} Entity;

// Collision result
typedef struct {
    uint32_t entity_a_id;
    uint32_t entity_b_id;
    float overlap_x;
    float overlap_y;
} Collision;

// Spatial hash grid for fast collision detection
typedef struct SpatialGrid SpatialGrid;

// Create a new spatial grid
// grid_size: size of each grid cell in world units
// world_width, world_height: dimensions of the world
SpatialGrid* spatial_grid_create(float grid_size, float world_width, float world_height);

// Destroy a spatial grid
void spatial_grid_destroy(SpatialGrid* grid);

// Clear all entities from the grid
void spatial_grid_clear(SpatialGrid* grid);

// Insert an entity into the grid
void spatial_grid_insert(SpatialGrid* grid, const Entity* entity);

// Query entities in a region
// Returns number of entities found
int spatial_grid_query(SpatialGrid* grid, const AABB* region,
                       Entity* out_entities, int max_entities);

// Check AABB collision
bool aabb_intersect(const AABB* a, const AABB* b);

// Calculate AABB overlap
void aabb_overlap(const AABB* a, const AABB* b, float* out_x, float* out_y);

// Collision detection between two entity arrays
// Returns number of collisions found
int detect_collisions(const Entity* entities_a, int count_a,
                      const Entity* entities_b, int count_b,
                      Collision* out_collisions, int max_collisions);

// Optimized collision detection using spatial grid
// Returns number of collisions found
int detect_collisions_spatial(SpatialGrid* grid,
                               const Entity* bullets, int bullet_count,
                               const Entity* enemies, int enemy_count,
                               Collision* out_collisions, int max_collisions);

#ifdef __cplusplus
}
#endif

#endif // ROTMG_COLLISION_H
