# ML Model Testing Plan - Systematic Evaluation

## Goal
Determine if the VAE learned meaningful texture patterns and can generate useful bullet patterns.

---

## Phase 1: Model Quality Assessment

### Test 1.1: Reconstruction Quality
**Question:** Can the model reconstruct training images accurately?

**Method:**
- Load 10 random training images
- Encode → Decode (full round trip)
- Measure reconstruction error (MSE, SSIM)
- Visualize original vs reconstructed

**Success Criteria:**
- MSE < 0.05 (good reconstruction)
- Visual similarity recognizable
- Patterns preserve structure

**Script:** `test_reconstruction.py`

---

### Test 1.2: Latent Space Quality
**Question:** Is the 32D latent space structured or random?

**Method:**
- Sample 1000 latent vectors from training data
- Reduce to 2D using t-SNE/PCA
- Visualize clusters
- Check if similar textures cluster together

**Success Criteria:**
- Clear clusters visible
- Similar patterns nearby in latent space
- No "dead zones" with garbage

**Script:** `test_latent_space.py`

---

### Test 1.3: Interpolation Smoothness
**Question:** Do patterns transition smoothly in latent space?

**Method:**
- Take 2 training patterns
- Interpolate between their latent vectors (10 steps)
- Generate intermediate patterns
- Check for smooth visual transition

**Success Criteria:**
- No sudden jumps or noise
- Gradual transformation
- All intermediates look valid

**Script:** `test_interpolation.py`

---

## Phase 2: Pattern Generation Quality

### Test 2.1: Random Sampling Quality
**Question:** What percentage of random samples are "good"?

**Method:**
- Generate 100 random latent vectors
- Decode to patterns
- Measure:
  - % with intensity > 0.3 (not too sparse)
  - % with structure (not pure noise)
  - Visual inspection

**Success Criteria:**
- At least 30% produce usable patterns
- If <30%, random sampling is bad strategy

**Script:** `test_random_sampling.py`

---

### Test 2.2: Latent Space Search
**Question:** Can we find better sampling regions?

**Method:**
- Start from training data latent vectors
- Random walk with small steps
- Score each generated pattern:
  - Density (bullet count)
  - Structure (not random noise)
  - Diversity (unique angles)
- Find "good" regions

**Success Criteria:**
- Identify 5-10 high-quality regions
- Patterns from these regions look coherent
- Better than pure random sampling

**Script:** `find_good_regions.py`

---

### Test 2.3: Style Control
**Question:** Can we control pattern properties?

**Method:**
- Test latent vector scaling:
  - z * 0.5 (sparse patterns)
  - z * 1.0 (normal)
  - z * 1.5 (dense patterns)
- Test dimension manipulation:
  - Vary dimension 0-31 individually
  - Find which dimensions control what
- Measure resulting pattern density

**Success Criteria:**
- Can generate sparse patterns (density < 0.3)
- Can generate dense patterns (density > 0.6)
- Predictable control method found

**Script:** `test_style_control.py`

---

## Phase 3: Generate Production Pattern Library

### Test 3.1: Diverse Pattern Generation
**Question:** Can we generate 100+ diverse, high-quality patterns?

**Method:**
- Use findings from Phase 2
- Sample from good regions
- Apply style controls
- Generate 100 patterns:
  - 30 sparse (low HP phases)
  - 40 medium (normal phases)
  - 30 dense (rage phases)

**Success Criteria:**
- All patterns visually distinct
- Cover range of densities
- No garbage patterns

**Script:** `generate_production_patterns.py`

---

## Phase 4: In-Game Testing

### Test 4.1: Player Difficulty Assessment
**Question:** Are patterns challenging but fair?

**Method:**
- Deploy new pattern library
- Measure:
  - Player death rate
  - Average survival time
  - Pattern variety perception
- Collect player feedback

**Success Criteria:**
- Death rate 20-40% (challenging)
- Patterns feel varied
- No "impossible" patterns

---

## Metrics We'll Track

### Quantitative Metrics:
1. **Reconstruction MSE** (lower is better)
2. **Pattern Density** (% of grid with intensity > 0.3)
3. **Direction Variance** (how diverse bullet angles are)
4. **Pattern Uniqueness** (L2 distance between patterns)
5. **Latent Space Coverage** (how much of 32D space we use)

### Qualitative Metrics:
1. **Visual Coherence** (does it look like a pattern?)
2. **Gameplay Feel** (is it fun to dodge?)
3. **Variety Perception** (do patterns feel different?)

---

## Expected Timeline

**Phase 1** (Model Quality): 30 minutes
- Run 3 test scripts
- Analyze results
- Determine if model is good enough

**Phase 2** (Pattern Generation): 1 hour
- Find good sampling regions
- Test style controls
- Validate diversity

**Phase 3** (Production Library): 30 minutes
- Generate 100+ patterns
- Export to JSON
- Deploy to game

**Phase 4** (In-Game Testing): Ongoing
- Player testing
- Iterate based on feedback

---

## Decision Points

### After Phase 1:
- ✅ Model quality good → Continue to Phase 2
- ❌ Model quality bad → Retrain with better data/architecture

### After Phase 2:
- ✅ Can generate good patterns → Continue to Phase 3
- ❌ Random sampling only → Need conditional VAE or different approach

### After Phase 3:
- ✅ Patterns diverse and usable → Deploy to game
- ❌ Still too random → Consider hand-crafted patterns or different model

---

## Let's Start with Phase 1!

I'll create the test scripts now. Ready to run them?
