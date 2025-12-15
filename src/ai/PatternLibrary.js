/**
 * PatternLibrary - Loads and manages pre-generated bullet patterns
 *
 * Patterns are generated offline using the ML model and stored as JSON.
 * This avoids runtime ML inference overhead.
 */

import fs from 'fs';
import path from 'path';

export class PatternLibrary {
  constructor() {
    this.patterns = [];
    this.patternsByStyle = {};
    this.loaded = false;
  }

  /**
   * Load patterns from JSON file
   * @param {string} jsonPath - Path to pattern_library.json
   */
  loadFromJSON(jsonPath) {
    try {
      const data = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));

      this.patterns = data.patterns.map(p => ({
        id: p.id,
        latentVector: p.latent_vector,
        intensity: p.intensity,
        direction: p.direction,
        stats: p.stats
      }));

      // Categorize by density/style
      this.categorizePatterns();

      this.loaded = true;
      console.log(`[PatternLibrary] Loaded ${this.patterns.length} patterns from ${jsonPath}`);

      return this.patterns.length;
    } catch (error) {
      console.error('[PatternLibrary] Failed to load patterns:', error);
      return 0;
    }
  }

  /**
   * Categorize patterns by their characteristics
   */
  categorizePatterns() {
    this.patternsByStyle = {
      sparse: [],
      medium: [],
      dense: [],
      chaotic: []
    };

    for (const pattern of this.patterns) {
      const density = pattern.stats.density_50; // Percentage of pixels > 0.5 intensity

      if (density < 0.15) {
        this.patternsByStyle.sparse.push(pattern);
      } else if (density < 0.30) {
        this.patternsByStyle.medium.push(pattern);
      } else if (density < 0.50) {
        this.patternsByStyle.dense.push(pattern);
      } else {
        this.patternsByStyle.chaotic.push(pattern);
      }
    }

    console.log('[PatternLibrary] Pattern distribution:');
    console.log(`  Sparse: ${this.patternsByStyle.sparse.length}`);
    console.log(`  Medium: ${this.patternsByStyle.medium.length}`);
    console.log(`  Dense: ${this.patternsByStyle.dense.length}`);
    console.log(`  Chaotic: ${this.patternsByStyle.chaotic.length}`);
  }

  /**
   * Get a random pattern
   * @returns {Object} Pattern object
   */
  getRandomPattern() {
    if (this.patterns.length === 0) {
      console.warn('[PatternLibrary] No patterns loaded');
      return null;
    }

    const index = Math.floor(Math.random() * this.patterns.length);
    return this.patterns[index];
  }

  /**
   * Get a random pattern by style
   * @param {string} style - 'sparse', 'medium', 'dense', 'chaotic'
   * @returns {Object} Pattern object
   */
  getPatternByStyle(style) {
    const stylePatterns = this.patternsByStyle[style];

    if (!stylePatterns || stylePatterns.length === 0) {
      console.warn(`[PatternLibrary] No patterns for style: ${style}`);
      return this.getRandomPattern();
    }

    const index = Math.floor(Math.random() * stylePatterns.length);
    return stylePatterns[index];
  }

  /**
   * Get pattern by ID
   * @param {number} id - Pattern ID
   * @returns {Object} Pattern object
   */
  getPatternById(id) {
    return this.patterns.find(p => p.id === id);
  }

  /**
   * Get a pattern appropriate for boss phase
   * @param {number} phase - Boss phase (1, 2, 3...)
   * @returns {Object} Pattern object
   */
  getPatternForPhase(phase) {
    // Phase 1: Sparse/medium patterns
    // Phase 2: Medium/dense patterns
    // Phase 3: Dense/chaotic patterns

    let style;
    if (phase === 1) {
      style = Math.random() < 0.5 ? 'sparse' : 'medium';
    } else if (phase === 2) {
      style = Math.random() < 0.5 ? 'medium' : 'dense';
    } else {
      style = Math.random() < 0.5 ? 'dense' : 'chaotic';
    }

    return this.getPatternByStyle(style);
  }

  /**
   * Convert pattern to 2D array format expected by adapter
   * @param {Object} pattern - Pattern object from library
   * @returns {Array} [32][32][2] pattern array
   */
  toPatternArray(pattern) {
    const intensity = pattern.intensity;
    const direction = pattern.direction;

    // Convert to [row][col][channel] format
    const patternArray = [];

    for (let row = 0; row < 32; row++) {
      const rowData = [];
      for (let col = 0; col < 32; col++) {
        rowData.push([
          intensity[row][col],
          direction[row][col]
        ]);
      }
      patternArray.push(rowData);
    }

    return patternArray;
  }
}

export default PatternLibrary;
