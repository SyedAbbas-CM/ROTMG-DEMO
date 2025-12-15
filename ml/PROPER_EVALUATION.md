# Proper VAE Evaluation Framework

## The Real Problem

**What we thought we were building:**
- ML generates diverse bullet patterns
- Patterns have visual variety (size/speed/lifetime)
- Boss uses these for interesting attacks

**What we actually have:**
- ML generates 32x32 grids of numbers
- PatternToBulletAdapter converts to bullets
- All bullets look the same (visual variety not working)
- Only 16 patterns (too few)

---

## Why Current Approach Isn't Working

### Issue 1: Visual Variety Not Applied
```javascript
// PatternToBulletAdapter.js - We ADDED this code:
const sizeMultiplier = 0.5 + (intensity * 1.5);  // Range: 0.5x to 2.0x
const bulletWidth = config.bulletWidth * sizeMultiplier;
const bulletHeight = config.bulletHeight * sizeMultiplier;
```

**But it's not showing in-game!**

Possible causes:
1. Renderer ignores width/height properties
2. Bullets are so small that 2x vs 0.5x isn't noticeable
3. Intensity values are all similar (0.05-0.27) so multipliers barely change
4. Client-side rendering overrides server bullet properties

### Issue 2: Pattern Diversity Not Visible
Even with 16 diverse patterns (similarity=-0.002), they all look like "spikes in 4 directions"

Why?
- 32x32 grid is TOO LOW RESOLUTION
- At low res, everything looks like cross/X/blob patterns
- Need higher resolution OR different representation

### Issue 3: Wrong Integration with Behavior System
ML patterns should be ONE tool in the behavior toolkit, not the ONLY tool.

**What we should have:**
```javascript
// In behavior definition:
{
  type: "PatternAttack",
  pattern: "ml:spiral_dense",      // ← ML-generated
  bulletSpeed: 5,                   // ← Adjustable
  bulletSize: 2,                    // ← Adjustable
  bulletLifetime: 3,                // ← Adjustable
  rotationSpeed: 45,                // ← Adjustable
  repeat: 3                         // ← Adjustable
}
```

**What we have:**
```javascript
// ML pattern is used directly, no control
boss.fireMLPattern(patternId);
```

---

## Proper Testing Methodology

### Test 1: Verify Visual Variety Works

**Goal:** Confirm bullets actually render with different sizes/speeds

**Steps:**
1. Create test pattern with extreme intensity values:
   ```python
   # High intensity blob (top-left)
   pattern[0:8, 0:8] = 1.0  # Should be LARGE SLOW bullets

   # Low intensity blob (bottom-right)
   pattern[24:32, 24:32] = 0.1  # Should be SMALL FAST bullets
   ```

2. Load this test pattern
3. Fire in-game
4. **EXPECTED:** Top-left bullets are visibly larger and slower
5. **IF NOT:** Visual variety code is broken

### Test 2: Verify Pattern Diversity

**Goal:** Confirm different patterns look different

**Steps:**
1. Generate 5 test patterns with OBVIOUS structures:
   ```python
   # Pattern A: All bullets to the right
   pattern_right[:, 16:32] = 1.0
   pattern_right[:, 0:16] = 0.0

   # Pattern B: All bullets upward
   pattern_up[0:16, :] = 1.0
   pattern_up[16:32, :] = 0.0

   # Pattern C: Ring shape
   pattern_ring[radial_distance > 8 and radial_distance < 16] = 1.0

   # Pattern D: Cross shape
   pattern_cross[15:17, :] = 1.0  # Horizontal
   pattern_cross[:, 15:17] = 1.0  # Vertical

   # Pattern E: Spiral (calculated)
   ```

2. Load these 5 patterns
3. Fire each one in-game
4. **EXPECTED:** Each pattern looks visually distinct
5. **IF NOT:** Pattern-to-bullet conversion is broken

### Test 3: Measure Pattern "Interestingness"

**Metrics that actually matter:**

1. **Dodgeability** (can player learn to dodge?)
   - Too easy: Boring
   - Too hard: Frustrating
   - Just right: Fun

2. **Visual clarity** (can player see the pattern?)
   - Bullets form recognizable shapes (Y/N)
   - Pattern telegraphs intent (Y/N)

3. **Variety** (do attacks feel different?)
   - Record 10 consecutive attacks
   - Count how many distinct patterns player recognizes

4. **Aesthetic quality** (does it look designed?)
   - Rate 1-10: Does this look intentional?

### Test 4: Compare to Hand-Coded Patterns

**Gold standard:** Manually designed patterns

Create 5 hand-coded patterns:
```javascript
// behaviors.json
{
  "spiral": {
    bullets: 8,
    radiusStart: 2,
    radiusEnd: 10,
    rotationSpeed: 45,
    // etc
  },
  "burst": { /* ... */ },
  "wave": { /* ... */ },
  "ring": { /* ... */ },
  "shotgun": { /* ... */ }
}
```

**Test:** Play with ML patterns, then play with hand-coded patterns
**Question:** Which feels better? Why?

**If hand-coded is clearly better:**
→ ML is adding complexity without value
→ Should abandon ML approach

**If ML is comparable or better:**
→ ML has value, refine it

---

## Action Plan: Fix Current Implementation

### Fix 1: Make Visual Variety Actually Work

**Check client-side rendering:**
```javascript
// In renderFirstPerson.js or wherever bullets are drawn
// Make sure bullet size is actually used:
ctx.fillRect(
  bullet.x - bullet.width/2,   // ← Uses width
  bullet.y - bullet.height/2,  // ← Uses height
  bullet.width,                // ← Not hardcoded!
  bullet.height
);
```

**Increase intensity variance:**
```python
# In pattern generation, apply contrast enhancement
pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())  # Normalize to 0-1
pattern = pattern ** 0.5  # Gamma correction to increase variance
```

### Fix 2: Generate Larger Pattern Library

**Current:** 16 patterns
**Need:** At least 50-100 patterns

```python
# visualize_patterns.py
# Change from:
num_patterns = 16

# To:
num_patterns = 100

# Also, sample from diverse regions:
latent_vectors = np.random.randn(num_patterns, 32) * 2.0  # Increase variance
```

### Fix 3: Add Pattern Parameters

**Integrate ML with behavior system:**

```javascript
// New approach: ML pattern as BASE, behaviors add parameters
class MLPatternBehavior {
  constructor(config) {
    this.patternId = config.patternId;      // Which ML pattern
    this.speedMultiplier = config.speed;    // ← Adjustable
    this.sizeMultiplier = config.size;      // ← Adjustable
    this.rotation = config.rotation;        // ← Adjustable
    this.spread = config.spread;            // ← Adjustable
  }

  execute() {
    let bullets = this.getMLPattern(this.patternId);

    // Apply behavior parameters
    bullets = bullets.map(b => ({
      ...b,
      vx: b.vx * this.speedMultiplier,
      vy: b.vy * this.speedMultiplier,
      width: b.width * this.sizeMultiplier,
      height: b.height * this.sizeMultiplier
    }));

    // Apply rotation
    if (this.rotation) {
      bullets = this.rotateBullets(bullets, this.rotation);
    }

    return bullets;
  }
}
```

### Fix 4: Hybrid Approach

**Best solution:** Combine ML + Procedural

```javascript
// Use ML for pattern SHAPE
// Use procedural for pattern PARAMETERS

const attack1 = {
  shape: "ml:pattern_42",      // ← ML generates shape
  speed: "fast",               // ← Behavior controls
  size: "large",               // ← Behavior controls
  lifetime: 3,                 // ← Behavior controls
  color: "red"                 // ← Behavior controls
};

const attack2 = {
  shape: "ml:pattern_42",      // ← SAME shape
  speed: "slow",               // ← Different parameters
  size: "small",               // ← Makes it feel different
  lifetime: 10,
  color: "blue"
};
```

This way:
- ML provides variety in shapes
- Behaviors provide control and tunability
- Best of both worlds

---

## Decision Tree: Is VAE Worth It?

### Question 1: Can we make visual variety work?
- **YES** → Continue to Q2
- **NO** → VAE is useless (can't see patterns)

### Question 2: Are ML patterns more interesting than hand-coded?
- **YES** → VAE has value, optimize it
- **NO** → Abandon VAE, use procedural

### Question 3: Can we control ML patterns enough for game design?
- **YES** → Integrate VAE into behavior system
- **NO** → VAE is too unpredictable, use procedural

### Question 4: Is the complexity worth it?
- VAE adds: Model training, pattern generation, debugging complexity
- Benefits: Pattern variety without hand-coding each one
- **Worth it?** Only if ML patterns are significantly better

---

## Recommended Next Steps

### Immediate (Today):

1. **Fix visual variety in renderer**
   - Check if bullet width/height are actually used
   - If not, implement proper size-based rendering
   - Test with extreme values (10x size difference)

2. **Create 5 test patterns with obvious shapes**
   - Hand-craft test patterns to verify the pipeline works
   - If hand-crafted patterns don't work, ML patterns won't either

3. **Compare ML vs Procedural**
   - Implement 3 simple procedural patterns (spiral, burst, wave)
   - Play with both, decide which is better

### This Week:

4. **If continuing with ML:**
   - Generate 100+ pattern library
   - Integrate with behavior system
   - Add parameter controls (speed/size/rotation)

5. **If abandoning ML:**
   - Design 20-30 hand-coded patterns
   - Create pattern combinator system
   - Focus on behavior complexity instead

---

## The Hard Truth

**ML might not be the answer here.**

Reasons to abandon VAE:
- Too complex for the benefit
- Hard to control/tune
- Hand-coded patterns might be just as good
- Behavior system already provides variety

Reasons to keep VAE:
- Can generate thousands of patterns automatically
- Patterns have organic, non-geometric feel
- Can interpolate between patterns smoothly
- Cool tech showcase

**The decision should be based on gameplay, not tech.**

If hand-coded patterns produce better gameplay, use those.
ML should make the game MORE fun, not just MORE complex.
