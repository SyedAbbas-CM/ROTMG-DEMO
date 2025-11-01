/**
 * BiomeDefinitions.js
 * Defines all biomes using your named tiles from lofi_environment atlas
 */

export const BIOME_DEFINITIONS = {
  // ============================================================================
  // GRASSLAND - Default biome, green grass with occasional trees
  // ============================================================================
  grassland: {
    name: 'Grassland',
    description: 'Rolling fields of green grass with scattered trees',

    // Tile pools (reference by name from lofi_environment.json)
    floorTiles: ['grass', 'grass_dark'],  // NO yellow grass in grassland
    wallTiles: [],  // Grasslands don't have walls
    obstacleTiles: ['tree', 'tree_2', 'tree_3', 'tree_4', 'boulder'],  // Mixed trees
    decorTiles: ['flowers_1', 'flowers_2'],  // Flowers appear in grassland

    // Weighted random selection
    floorWeights: {
      'grass': 85,  // More regular grass
      'grass_dark': 15  // Some darker grass
    },

    obstacleWeights: {
      'tree': 35,
      'tree_2': 25,
      'tree_3': 20,
      'tree_4': 10,
      'boulder': 10
    },

    decorWeights: {
      'flowers_1': 50,
      'flowers_2': 50
    },

    // Generation parameters
    obstacleDensity: 0.03,  // 3% chance per tile - moderate trees
    decorDensity: 0.02,      // 2% chance per tile

    // Noise ranges for selection
    heightRange: [-0.1, 0.2],
    temperatureRange: [-0.3, 0.6],
    moistureRange: [-0.2, 0.5]
  },

  // ============================================================================
  // PLAINS - Open fields with yellow grass, very sparse obstacles
  // ============================================================================
  plains: {
    name: 'Plains',
    description: 'Wide open plains with golden grass',

    floorTiles: ['grass_yellow'],
    wallTiles: [],
    obstacleTiles: ['boulder'],  // NO TREES in plains biome per user requirement
    decorTiles: ['flowers_2', 'flowers_1'],  // Flowers appear in plains

    floorWeights: {
      'grass_yellow': 100
    },

    obstacleWeights: {
      'boulder': 100  // Only boulders, no trees
    },

    decorWeights: {
      'flowers_2': 60,
      'flowers_1': 40
    },

    obstacleDensity: 0.002,  // Extremely sparse - just occasional rocks
    decorDensity: 0.015,  // Moderate flower density

    heightRange: [-0.1, 0.2],
    temperatureRange: [0.1, 0.7],
    moistureRange: [-0.3, 0.2]  // Drier than grassland
  },

  // ============================================================================
  // FOREST - Dense trees, darker grass
  // ============================================================================
  forest: {
    name: 'Forest',
    description: 'Dense woodland with green grass',

    floorTiles: ['grass'],
    wallTiles: [],
    obstacleTiles: ['tree', 'tree_2', 'tree_3', 'tree_4'],  // ONLY trees in forest
    decorTiles: ['flowers_1', 'flowers_2'],  // Flowers appear in forests

    floorWeights: {
      'grass': 100
    },

    obstacleWeights: {
      'tree': 40,
      'tree_2': 25,
      'tree_3': 20,
      'tree_4': 15
    },

    decorWeights: {
      'flowers_1': 60,
      'flowers_2': 40
    },

    obstacleDensity: 0.15,  // Much denser - dense trees
    decorDensity: 0.02,  // Moderate flowers

    heightRange: [0.2, 0.5],
    temperatureRange: [-0.3, 0.4],
    moistureRange: [0.3, 1.0]
  },

  // ============================================================================
  // DESERT - Sand tiles, dead trees, boulders
  // ============================================================================
  desert: {
    name: 'Desert',
    description: 'Hot sandy wasteland',

    floorTiles: ['sand_1', 'sand_2'],
    wallTiles: [],
    obstacleTiles: ['tree_dead', 'tree_burnt', 'Boulder_yellow', 'rocks_3'],
    decorTiles: [],

    floorWeights: {
      'sand_1': 60,
      'sand_2': 40
    },

    obstacleWeights: {
      'tree_dead': 40,
      'tree_burnt': 30,
      'Boulder_yellow': 20,
      'rocks_3': 10
    },

    obstacleDensity: 0.01,  // Very sparse (reduced from 0.02 for cleaner desert feel)
    decorDensity: 0,

    heightRange: [-0.1, 0.2],
    temperatureRange: [0.4, 1.0],
    moistureRange: [-1.0, -0.2]
  },

  // ============================================================================
  // OCEAN - Deep water
  // ============================================================================
  ocean: {
    name: 'Ocean',
    description: 'Deep blue waters',

    floorTiles: ['deep_water', 'deep_water_2'],
    wallTiles: [],
    obstacleTiles: [],
    decorTiles: [],

    floorWeights: {
      'deep_water': 60,
      'deep_water_2': 40
    },

    obstacleDensity: 0,
    decorDensity: 0,

    heightRange: [-1.0, -0.6],
    temperatureRange: [-1.0, 1.0],
    moistureRange: [-1.0, 1.0]
  },

  // ============================================================================
  // COAST - Shallow water near land
  // ============================================================================
  coast: {
    name: 'Coast',
    description: 'Shallow coastal waters',

    floorTiles: ['water_1', 'water_2'],
    wallTiles: [],
    obstacleTiles: ['rocks_1', 'rocks_2'],
    decorTiles: [],

    floorWeights: {
      'water_1': 50,
      'water_2': 50
    },

    obstacleWeights: {
      'rocks_1': 50,
      'rocks_2': 50
    },

    obstacleDensity: 0.05,
    decorDensity: 0,

    heightRange: [-0.6, -0.3],
    temperatureRange: [-1.0, 1.0],
    moistureRange: [-1.0, 1.0]
  },

  // ============================================================================
  // BEACH - Sandy shores
  // ============================================================================
  beach: {
    name: 'Beach',
    description: 'Sandy coastline',

    floorTiles: ['sand_1', 'sand_2'],
    wallTiles: [],
    obstacleTiles: ['rocks_1', 'rocks_2', 'rocks_3'],
    decorTiles: [],

    floorWeights: {
      'sand_1': 60,
      'sand_2': 40
    },

    obstacleWeights: {
      'rocks_1': 40,
      'rocks_2': 40,
      'rocks_3': 20
    },

    obstacleDensity: 0.04,
    decorDensity: 0,

    heightRange: [-0.3, -0.1],
    temperatureRange: [0.3, 1.0],
    moistureRange: [-1.0, 1.0]
  },

  // ============================================================================
  // SWAMP - Dark grass, dead trees, water patches
  // ============================================================================
  swamp: {
    name: 'Swamp',
    description: 'Murky wetlands with dead vegetation',

    floorTiles: ['grass_dark', 'water_1'],
    wallTiles: [],
    obstacleTiles: ['tree_dead', 'tree_burnt', 'rocks_1'],
    decorTiles: [],

    floorWeights: {
      'grass_dark': 60,
      'water_1': 40
    },

    obstacleWeights: {
      'tree_dead': 50,
      'tree_burnt': 30,
      'rocks_1': 20
    },

    obstacleDensity: 0.08,
    decorDensity: 0,

    heightRange: [-0.1, 0.2],
    temperatureRange: [-0.6, 0.0],
    moistureRange: [0.3, 1.0]
  },

  // ============================================================================
  // TUNDRA - Cold plains
  // ============================================================================
  tundra: {
    name: 'Tundra',
    description: 'Cold, barren plains',

    floorTiles: ['grass_dark'],  // Limited palette
    wallTiles: [],
    obstacleTiles: ['rocks_1', 'rocks_2', 'rocks_3', 'boulder'],
    decorTiles: [],

    floorWeights: {
      'grass_dark': 100
    },

    obstacleWeights: {
      'rocks_1': 30,
      'rocks_2': 30,
      'rocks_3': 20,
      'boulder': 20
    },

    obstacleDensity: 0.04,
    decorDensity: 0,

    heightRange: [-0.1, 0.2],
    temperatureRange: [-1.0, -0.3],
    moistureRange: [-1.0, 0.3]
  },

  // ============================================================================
  // JUNGLE - Dense vegetation, bright grass
  // ============================================================================
  jungle: {
    name: 'Jungle',
    description: 'Thick tropical forest',

    floorTiles: ['grass', 'grass_yellow'],
    wallTiles: [],
    obstacleTiles: ['tree', 'tree_yellow', 'rocks_1'],
    decorTiles: ['flowers_1', 'flowers_2'],

    floorWeights: {
      'grass': 50,
      'grass_yellow': 50
    },

    obstacleWeights: {
      'tree': 60,
      'tree_yellow': 30,
      'rocks_1': 10
    },

    obstacleDensity: 0.2,   // Very dense!
    decorDensity: 0.05,

    heightRange: [-0.1, 0.2],
    temperatureRange: [0.2, 1.0],
    moistureRange: [0.5, 1.0]
  },

  // ============================================================================
  // HILLS - Rocky terrain with grass
  // ============================================================================
  hills: {
    name: 'Hills',
    description: 'Rolling rocky hills',

    floorTiles: ['grass', 'grass_dark'],
    wallTiles: [],
    obstacleTiles: ['rocks_1', 'rocks_2', 'rocks_3'],  // Only rocks_1/2/3 for hills
    decorTiles: [],  // NO FLOWERS in hills

    floorWeights: {
      'grass': 60,
      'grass_dark': 40
    },

    obstacleWeights: {
      'rocks_1': 34,  // Even distribution across 3 rock types
      'rocks_2': 33,
      'rocks_3': 33
    },

    obstacleDensity: 0.1,
    decorDensity: 0,

    heightRange: [0.2, 0.5],
    temperatureRange: [-1.0, 1.0],
    moistureRange: [-1.0, 0.3]
  },

  // ============================================================================
  // MOUNTAIN - High elevation, mostly rocks
  // ============================================================================
  mountain: {
    name: 'Mountain',
    description: 'Steep rocky mountains',

    floorTiles: ['cobblestone'],
    wallTiles: [],
    obstacleTiles: ['rocks_1', 'rocks_2', 'rocks_3'],  // Only rocks_1/2/3 for mountains
    decorTiles: [],  // NO FLOWERS

    floorWeights: {
      'cobblestone': 100
    },

    obstacleWeights: {
      'rocks_1': 34,  // Even distribution across 3 rock types
      'rocks_2': 33,
      'rocks_3': 33
    },

    obstacleDensity: 0.15,
    decorDensity: 0,

    heightRange: [0.5, 0.7],
    temperatureRange: [-0.2, 1.0],
    moistureRange: [-1.0, 1.0]
  },

  // ============================================================================
  // SNOW MOUNTAIN - Cold high elevation
  // ============================================================================
  snow_mountain: {
    name: 'Snow Mountain',
    description: 'Frozen peaks',

    floorTiles: ['grass_dark'],  // Could use ice tiles if you add them
    wallTiles: [],
    obstacleTiles: ['rocks_1', 'rocks_2', 'rocks_3'],  // rocks_1/2/3 for snow mountains
    decorTiles: [],  // NO FLOWERS

    floorWeights: {
      'grass_dark': 100
    },

    obstacleWeights: {
      'rocks_1': 34,  // Even distribution
      'rocks_2': 33,
      'rocks_3': 33
    },

    obstacleDensity: 0.12,
    decorDensity: 0,

    heightRange: [0.5, 0.7],
    temperatureRange: [-1.0, -0.2],
    moistureRange: [-1.0, 1.0]
  },

  // ============================================================================
  // MOUNTAIN PEAK - Highest elevations, impassable
  // ============================================================================
  mountain_peak: {
    name: 'Mountain Peak',
    description: 'Towering peaks',

    floorTiles: ['cobblestone'],
    wallTiles: [],
    obstacleTiles: ['rocks_1', 'rocks_2', 'rocks_3'],  // rocks_1/2/3 for mountain peaks
    decorTiles: [],  // NO FLOWERS

    floorWeights: {
      'cobblestone': 100
    },

    obstacleWeights: {
      'rocks_1': 34,  // Even distribution
      'rocks_2': 33,
      'rocks_3': 33
    },

    obstacleDensity: 0.3,  // Nearly impassable - very dense
    decorDensity: 0,

    heightRange: [0.7, 1.0],
    temperatureRange: [-1.0, 0.6],
    moistureRange: [-1.0, 1.0]
  },

  // ============================================================================
  // VOLCANIC - Lava and burnt landscape
  // ============================================================================
  volcanic: {
    name: 'Volcanic',
    description: 'Lava-filled volcanic wasteland',

    floorTiles: ['lava_1', 'lava_2', 'cobblestone'],  // Lava with cobblestone walkable areas
    wallTiles: [],
    obstacleTiles: ['cobblestone'],  // Use cobblestone as obstacles in lava biome
    decorTiles: [],  // NO FLOWERS in lava biome

    floorWeights: {
      'lava_1': 50,
      'lava_2': 40,
      'cobblestone': 10  // Some walkable cobblestone tiles
    },

    obstacleWeights: {
      'cobblestone': 100  // Only cobblestone obstacles
    },

    obstacleDensity: 0.015,  // Very sparse cobblestone rocks
    decorDensity: 0,  // NO flowers

    heightRange: [0.7, 1.0],
    temperatureRange: [0.6, 1.0],
    moistureRange: [-1.0, 1.0]
  }
};
