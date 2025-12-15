#include "collision.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Spatial grid cell structure
typedef struct GridCell {
    Entity* entities;
    int count;
    int capacity;
} GridCell;

// Spatial grid structure
struct SpatialGrid {
    GridCell* cells;
    int grid_width;
    int grid_height;
    float cell_size;
    float world_width;
    float world_height;
};

SpatialGrid* spatial_grid_create(float grid_size, float world_width, float world_height) {
    SpatialGrid* grid = (SpatialGrid*)malloc(sizeof(SpatialGrid));
    if (!grid) return NULL;

    grid->cell_size = grid_size;
    grid->world_width = world_width;
    grid->world_height = world_height;
    grid->grid_width = (int)ceilf(world_width / grid_size);
    grid->grid_height = (int)ceilf(world_height / grid_size);

    int total_cells = grid->grid_width * grid->grid_height;
    grid->cells = (GridCell*)calloc(total_cells, sizeof(GridCell));
    if (!grid->cells) {
        free(grid);
        return NULL;
    }

    return grid;
}

void spatial_grid_destroy(SpatialGrid* grid) {
    if (!grid) return;

    int total_cells = grid->grid_width * grid->grid_height;
    for (int i = 0; i < total_cells; i++) {
        if (grid->cells[i].entities) {
            free(grid->cells[i].entities);
        }
    }
    free(grid->cells);
    free(grid);
}

void spatial_grid_clear(SpatialGrid* grid) {
    if (!grid) return;

    int total_cells = grid->grid_width * grid->grid_height;
    for (int i = 0; i < total_cells; i++) {
        grid->cells[i].count = 0;
    }
}

static inline int grid_index(const SpatialGrid* grid, int x, int y) {
    if (x < 0 || x >= grid->grid_width || y < 0 || y >= grid->grid_height) {
        return -1;
    }
    return y * grid->grid_width + x;
}

void spatial_grid_insert(SpatialGrid* grid, const Entity* entity) {
    if (!grid || !entity) return;

    // Calculate grid bounds for this entity
    float half_w = entity->bounds.w * 0.5f;
    float half_h = entity->bounds.h * 0.5f;

    int min_x = (int)floorf((entity->bounds.x - half_w) / grid->cell_size);
    int max_x = (int)floorf((entity->bounds.x + half_w) / grid->cell_size);
    int min_y = (int)floorf((entity->bounds.y - half_h) / grid->cell_size);
    int max_y = (int)floorf((entity->bounds.y + half_h) / grid->cell_size);

    // Clamp to grid bounds
    if (min_x < 0) min_x = 0;
    if (max_x >= grid->grid_width) max_x = grid->grid_width - 1;
    if (min_y < 0) min_y = 0;
    if (max_y >= grid->grid_height) max_y = grid->grid_height - 1;

    // Insert into all overlapping cells
    for (int y = min_y; y <= max_y; y++) {
        for (int x = min_x; x <= max_x; x++) {
            int idx = grid_index(grid, x, y);
            if (idx < 0) continue;

            GridCell* cell = &grid->cells[idx];

            // Resize if needed
            if (cell->count >= cell->capacity) {
                int new_capacity = cell->capacity == 0 ? 8 : cell->capacity * 2;
                Entity* new_entities = (Entity*)realloc(cell->entities,
                                                        new_capacity * sizeof(Entity));
                if (!new_entities) continue;

                cell->entities = new_entities;
                cell->capacity = new_capacity;
            }

            cell->entities[cell->count++] = *entity;
        }
    }
}

int spatial_grid_query(SpatialGrid* grid, const AABB* region,
                       Entity* out_entities, int max_entities) {
    if (!grid || !region || !out_entities) return 0;

    float half_w = region->w * 0.5f;
    float half_h = region->h * 0.5f;

    int min_x = (int)floorf((region->x - half_w) / grid->cell_size);
    int max_x = (int)floorf((region->x + half_w) / grid->cell_size);
    int min_y = (int)floorf((region->y - half_h) / grid->cell_size);
    int max_y = (int)floorf((region->y + half_h) / grid->cell_size);

    // Clamp to grid bounds
    if (min_x < 0) min_x = 0;
    if (max_x >= grid->grid_width) max_x = grid->grid_width - 1;
    if (min_y < 0) min_y = 0;
    if (max_y >= grid->grid_height) max_y = grid->grid_height - 1;

    int count = 0;
    for (int y = min_y; y <= max_y && count < max_entities; y++) {
        for (int x = min_x; x <= max_x && count < max_entities; x++) {
            int idx = grid_index(grid, x, y);
            if (idx < 0) continue;

            GridCell* cell = &grid->cells[idx];
            for (int i = 0; i < cell->count && count < max_entities; i++) {
                // Check if this entity intersects the query region
                if (aabb_intersect(&cell->entities[i].bounds, region)) {
                    out_entities[count++] = cell->entities[i];
                }
            }
        }
    }

    return count;
}

bool aabb_intersect(const AABB* a, const AABB* b) {
    float a_half_w = a->w * 0.5f;
    float a_half_h = a->h * 0.5f;
    float b_half_w = b->w * 0.5f;
    float b_half_h = b->h * 0.5f;

    return (fabsf(a->x - b->x) < (a_half_w + b_half_w)) &&
           (fabsf(a->y - b->y) < (a_half_h + b_half_h));
}

void aabb_overlap(const AABB* a, const AABB* b, float* out_x, float* out_y) {
    float a_half_w = a->w * 0.5f;
    float a_half_h = a->h * 0.5f;
    float b_half_w = b->w * 0.5f;
    float b_half_h = b->h * 0.5f;

    float dx = fabsf(a->x - b->x);
    float dy = fabsf(a->y - b->y);

    *out_x = (a_half_w + b_half_w) - dx;
    *out_y = (a_half_h + b_half_h) - dy;
}

int detect_collisions(const Entity* entities_a, int count_a,
                      const Entity* entities_b, int count_b,
                      Collision* out_collisions, int max_collisions) {
    int collision_count = 0;

    for (int i = 0; i < count_a && collision_count < max_collisions; i++) {
        for (int j = 0; j < count_b && collision_count < max_collisions; j++) {
            // Skip same-layer collisions (friendly fire)
            if (entities_a[i].layer == entities_b[j].layer) continue;

            if (aabb_intersect(&entities_a[i].bounds, &entities_b[j].bounds)) {
                Collision* col = &out_collisions[collision_count++];
                col->entity_a_id = entities_a[i].id;
                col->entity_b_id = entities_b[j].id;
                aabb_overlap(&entities_a[i].bounds, &entities_b[j].bounds,
                           &col->overlap_x, &col->overlap_y);
            }
        }
    }

    return collision_count;
}

int detect_collisions_spatial(SpatialGrid* grid,
                               const Entity* bullets, int bullet_count,
                               const Entity* enemies, int enemy_count,
                               Collision* out_collisions, int max_collisions) {
    if (!grid) {
        // Fallback to brute force if no grid
        return detect_collisions(bullets, bullet_count, enemies, enemy_count,
                                out_collisions, max_collisions);
    }

    // Clear grid
    spatial_grid_clear(grid);

    // Insert all enemies into grid
    for (int i = 0; i < enemy_count; i++) {
        spatial_grid_insert(grid, &enemies[i]);
    }

    int collision_count = 0;

    // For each bullet, query nearby enemies
    Entity nearby_enemies[64]; // Stack buffer for nearby entities
    for (int i = 0; i < bullet_count && collision_count < max_collisions; i++) {
        // Query enemies near this bullet
        int nearby_count = spatial_grid_query(grid, &bullets[i].bounds,
                                             nearby_enemies, 64);

        // Check collisions with nearby enemies only
        for (int j = 0; j < nearby_count && collision_count < max_collisions; j++) {
            // Skip same-layer collisions
            if (bullets[i].layer == nearby_enemies[j].layer) continue;

            if (aabb_intersect(&bullets[i].bounds, &nearby_enemies[j].bounds)) {
                Collision* col = &out_collisions[collision_count++];
                col->entity_a_id = bullets[i].id;
                col->entity_b_id = nearby_enemies[j].id;
                aabb_overlap(&bullets[i].bounds, &nearby_enemies[j].bounds,
                           &col->overlap_x, &col->overlap_y);
            }
        }
    }

    return collision_count;
}
