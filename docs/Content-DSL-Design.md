# Content Definition Language (DSL) - Design Document

## Overview

A human-friendly domain-specific language for defining game content (enemies, dungeons, bosses, items) that compiles to the engine's internal JSON formats. This DSL prioritizes **readability**, **ease of use**, and **rapid iteration** for content creators.

## Design Goals

1. **Minimal Syntax** - Reduce boilerplate, use natural language constructs
2. **Inheritance & Composition** - Reuse definitions via templates and mixins
3. **Clear Hierarchies** - Nest related concepts (states contain behaviors, dungeons contain rooms)
4. **Inline Documentation** - Comments and self-documenting syntax
5. **Compile-Time Validation** - Catch errors before runtime

---

## 1. Enemy DSL

### Basic Enemy Definition

```
enemy Slime {
  sprite: "slime_green"
  hp: 50
  speed: 15
  size: 1x1

  attack {
    damage: 10
    range: 80
    cooldown: 1.5s
    projectile: "slime_shot"
  }

  behavior: aggressive
  xp: 25

  drops {
    common: health_potion (30%)
    rare: slime_essence (5%)
  }
}
```

### State-Based Enemy with Behaviors

```
enemy BossOrc {
  sprite: "orc_king"
  hp: 500
  speed: 20
  size: 2x2

  // Initial state
  state idle {
    on_player_near(distance < 150) -> chase
  }

  state chase {
    behavior: follow_player(speed: 25)

    on_distance(< 80) -> melee_attack
    on_hp(< 50%) -> enrage
  }

  state melee_attack {
    behavior: dash_toward_player(speed: 40)
    attack {
      damage: 30
      range: 40
      type: melee
    }

    on_attack_complete -> chase
  }

  state enrage {
    speed_multiplier: 1.5
    damage_multiplier: 2.0

    behavior: circle_strafe(radius: 100)
    attack {
      damage: 20
      range: 150
      cooldown: 0.5s
      projectiles: 8
      spread: 360deg
      pattern: circle
    }

    on_hp(< 20%) -> desperate
  }

  state desperate {
    behavior: teleport_random(interval: 3s, range: 200)
    attack {
      damage: 15
      range: 200
      cooldown: 0.3s
      projectiles: 3
      spread: 45deg
    }
  }

  drops {
    always: boss_soul
    rare: legendary_axe (10%)
    common: gold (50-100)
  }
}
```

### Enemy Templates & Inheritance

```
template BasicMelee {
  attack {
    type: melee
    range: 50
    cooldown: 1s
  }
  behavior: aggressive
  speed: 20
}

enemy Goblin extends BasicMelee {
  sprite: "goblin"
  hp: 30
  attack.damage: 15
  drops { common: scrap_metal }
}

enemy OrcWarrior extends BasicMelee {
  sprite: "orc_warrior"
  hp: 120
  attack.damage: 35
  attack.range: 60
  speed: 25
  drops { rare: iron_sword (15%) }
}
```

---

## 2. Dungeon DSL

### Room-Based Dungeon Definition

```
dungeon Crypt {
  theme: undead
  difficulty: medium
  size: 30x40
  tile_size: 16

  room entrance at (5, 5) size (8, 8) {
    floor: stone_floor
    walls: stone_wall

    portal at (4, 4) to overworld

    spawn {
      enemy: Skeleton x 3
      positions: random
    }

    exit north to corridor_1
  }

  corridor corridor_1 from entrance.north {
    floor: stone_floor
    walls: stone_wall
    length: 10
    width: 3
    direction: north

    spawn {
      enemy: Zombie x 2
      positions: spread
    }

    exit north to treasure_room
  }

  room treasure_room at (5, 24) size (12, 10) {
    floor: marble_floor
    walls: stone_wall

    decorations {
      torch at corners
      chest at (6, 5) contains {
        gold (50-200)
        potion (50%)
        rare_item (10%)
      }
    }

    spawn {
      enemy: SkeletonArcher x 4
      positions: corners
    }

    exit east to boss_chamber
  }

  room boss_chamber at (18, 24) size (15, 15) {
    floor: blood_stained_floor
    walls: dark_stone_wall

    decorations {
      candles around perimeter
      altar at center
    }

    boss CryptLord at (7, 7) {
      on_death: open_exit
    }

    exit north to loot_room {
      locked: true
      unlock_on: CryptLord.death
    }
  }

  room loot_room at (18, 40) size (10, 8) {
    floor: gold_floor
    walls: stone_wall

    chests {
      legendary_chest at (5, 4) contains {
        legendary_weapon (guaranteed)
        gold (200-500)
        rare_gem (30%)
      }
    }

    portal at (5, 2) to overworld label "Exit"
  }
}
```

### Procedural Dungeon Definition

```
dungeon RandomCave {
  theme: cave
  difficulty: easy
  generation: procedural

  parameters {
    min_rooms: 5
    max_rooms: 10
    room_size: (6x6) to (12x12)
    corridor_width: 3
    branching_factor: 0.3
  }

  tileset {
    floor: [dirt_1, dirt_2, grass_dark]
    walls: [rocks_1, rocks_2, rocks_3]
    obstacles: [boulder, boulder_yellow] (density: 0.1)
  }

  spawn_rules {
    per_room {
      enemies: 2-5
      types: [Bat, Spider, Snake]
      positions: random
    }

    boss_room {
      enemy: CaveQueen
      minions: Spider x 4
    }
  }

  loot_distribution {
    common_chests: 30% of rooms
    rare_chests: 5% of rooms
  }
}
```

---

## 3. Behavior DSL (Embedded)

### Reusable Behavior Definitions

```
behavior aggressive {
  on_see_player: chase
  chase_speed: 1.0x
  attack_when: in_range
}

behavior defensive {
  on_see_player: retreat
  retreat_distance: 100
  attack_when: cornered
}

behavior boss_phase_1 {
  movement: circle_strafe(radius: 150, speed: 20)
  attack_pattern: shotgun(projectiles: 5, spread: 30deg, cooldown: 2s)
}

behavior boss_phase_2 {
  movement: teleport_random(interval: 4s)
  attack_pattern: spiral(projectiles: 12, rotation_speed: 90deg/s)
}
```

---

## 4. Attack Pattern DSL

### Predefined Patterns

```
attack_pattern shotgun {
  projectiles: 5
  spread: 45deg
  damage: 15
  speed: 150
  sprite: "bullet_red"
}

attack_pattern spiral {
  projectiles: 1
  fire_rate: 0.1s
  rotation_speed: 90deg/s
  damage: 10
  speed: 100
  lifetime: 3s
  sprite: "bullet_blue"
}

attack_pattern tracking_missile {
  projectiles: 1
  damage: 25
  speed: 80
  homing: true
  homing_strength: 0.5
  lifetime: 5s
  sprite: "missile"
}
```

---

## 5. Loot Table DSL

```
loot_table goblin_drops {
  always: copper_coin (5-15)

  common {
    health_potion (40%)
    scrap_metal (30%)
  }

  uncommon {
    rusty_dagger (15%)
    leather_armor (10%)
  }

  rare {
    goblin_mask (5%)
    enchanted_ring (2%)
  }
}

loot_table boss_treasure {
  guaranteed {
    boss_soul
    gold (200-500)
  }

  rare {
    legendary_weapon (15%)
    unique_armor (10%)
    skill_tome (8%)
  }

  mythic {
    artifact (2%)
  }
}
```

---

## 6. Syntax Reference

### Data Types

- **Numbers**: `50`, `3.5`, `1.5s` (time), `45deg` (angle)
- **Strings**: `"slime_green"`, `stone_floor`
- **Ranges**: `5-10`, `(100-500)`
- **Percentages**: `30%`, `(5%)`
- **Sizes**: `1x1`, `(8x12)`
- **Positions**: `(x, y)`, `at (5, 10)`
- **Booleans**: `true`, `false`
- **Arrays**: `[item1, item2, item3]`

### Keywords

- **Definitions**: `enemy`, `dungeon`, `room`, `behavior`, `attack_pattern`, `loot_table`
- **Properties**: `sprite`, `hp`, `speed`, `damage`, `range`, `cooldown`
- **Modifiers**: `extends`, `template`, `mixin`, `override`
- **States**: `state`, `on_transition`, `on_event`
- **Triggers**: `on_player_near`, `on_hp`, `on_death`, `on_distance`
- **Spatial**: `at`, `to`, `from`, `size`, `north`, `south`, `east`, `west`
- **Spawning**: `spawn`, `boss`, `portal`, `chest`, `decorations`

---

## 7. Compiler Architecture

```
DSL Source Code (.enemy, .dungeon)
          ↓
    Lexer (Tokenization)
          ↓
    Parser (AST Generation)
          ↓
  Semantic Analysis (Type Checking)
          ↓
    Code Generator (JSON Output)
          ↓
  Engine Runtime (EntityDatabase, MapManager)
```

### File Extensions

- `.enemy` - Enemy definitions
- `.dungeon` - Dungeon/map definitions
- `.behavior` - Reusable behavior definitions
- `.attack` - Attack pattern definitions
- `.loot` - Loot table definitions

### Compiler Output

Compiles to existing JSON formats:
- Enemy DSL → `/src/assets/entities/*.json` (matches `enemySchema.json`)
- Dungeon DSL → `/public/maps/*.json` (matches `TestDungeon.json` format)

---

## 8. Integration with Existing Systems

### Enemy Definitions → EntityDatabase

DSL compiles to JSON consumed by `EntityDatabase.js`:

```javascript
{
  "id": "slime_green",
  "name": "Slime",
  "sprite": "slime_green",
  "hp": 50,
  "speed": 15,
  "width": 1,
  "height": 1,
  "attack": {
    "damage": 10,
    "range": 80,
    "cooldown": 1500,
    "sprite": "slime_shot"
  },
  "ai": { "behavior": "aggressive" },
  "xp": 25,
  "drops": [...]
}
```

### Dungeon Definitions → MapManager

DSL compiles to tilemap JSON consumed by `MapManager.js`:

```javascript
{
  "width": 30,
  "height": 40,
  "tileW": 16,
  "tileH": 16,
  "layers": [...],
  "entities": [...],
  "objects": [...]
}
```

---

## 9. Implementation Phases

### Phase 1: Core Parser (Week 1-2)
- Lexer + Tokenizer
- Basic enemy DSL parser
- Output JSON for simple enemies
- Test with existing EnemyManager

### Phase 2: State Machine Support (Week 3)
- State definitions
- Transition syntax
- Behavior system integration

### Phase 3: Dungeon DSL (Week 4-5)
- Room-based dungeon parser
- Tileset compilation
- Spawn point generation

### Phase 4: Advanced Features (Week 6+)
- Templates & inheritance
- Procedural generation parameters
- Attack pattern library
- Loot table system

---

## 10. Example Workflow

### Content Creator Workflow

1. **Write DSL File** - Create `slime.enemy`:
   ```
   enemy Slime {
     sprite: "slime_green"
     hp: 50
     speed: 15
     attack { damage: 10, range: 80, cooldown: 1.5s }
     behavior: aggressive
   }
   ```

2. **Compile** - Run compiler:
   ```bash
   node tools/compile-content.js enemies/slime.enemy
   ```

3. **Output** - Generates `slime.json` in `/src/assets/entities/`

4. **Test** - Enemy automatically loaded by EntityDatabase on server restart

5. **Iterate** - Edit DSL, recompile, test (< 10 seconds)

---

## 11. Benefits Over Manual JSON

| Aspect | Manual JSON | DSL |
|--------|------------|-----|
| **Boilerplate** | High (quotes, commas, braces) | Minimal (natural syntax) |
| **Readability** | Low (nested structures) | High (hierarchical blocks) |
| **Errors** | Runtime (missing commas, typos) | Compile-time (validation) |
| **Reusability** | Copy-paste | Templates & inheritance |
| **Iteration Speed** | Slow (edit→save→restart→test) | Fast (compile→hot-reload) |
| **Documentation** | External | Inline comments |

---

## 12. Future Extensions

- **Visual Editor** - GUI that generates DSL code
- **Hot Reloading** - Live content updates without server restart
- **Validation Tools** - Lint DSL files for common mistakes
- **AI Assistance** - LLM generates enemy definitions from descriptions
- **Version Control** - Diff-friendly text format (better than binary JSON)

---

## 13. Reference Implementation

See `/src/content/dsl/` for:
- `lexer.js` - Tokenizer
- `parser.js` - AST generator
- `compiler.js` - JSON code generator
- `validator.js` - Schema validation
- `examples/` - Sample `.enemy` and `.dungeon` files

Run tests:
```bash
npm test -- tests/dsl-*.spec.js
```

---

## Conclusion

This DSL provides a **streamlined**, **human-friendly** way to create game content while maintaining compatibility with the existing engine architecture. Content creators can focus on **design** rather than **syntax**, dramatically improving iteration speed and reducing errors.
