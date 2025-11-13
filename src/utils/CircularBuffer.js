// File: src/utils/CircularBuffer.js
// Circular buffer for storing position history with timestamps
// Used for lag compensation to rewind player positions

export class CircularBuffer {
  constructor(maxSamples = 10) {
    this.maxSamples = maxSamples;

    // Use typed arrays for memory efficiency
    // Store x, y pairs
    this.positions = new Float32Array(maxSamples * 2);

    // Store timestamps (milliseconds)
    this.timestamps = new Float32Array(maxSamples);

    // Circular buffer state
    this.head = 0;
    this.count = 0;
  }

  /**
   * Add a new position sample to the buffer
   * @param {number} x - X position in tile units
   * @param {number} y - Y position in tile units
   * @param {number} timestamp - Timestamp in milliseconds
   */
  add(x, y, timestamp) {
    const index = this.head;

    // Store position (x, y pair)
    this.positions[index * 2] = x;
    this.positions[index * 2 + 1] = y;

    // Store timestamp
    this.timestamps[index] = timestamp;

    // Advance head (circular)
    this.head = (this.head + 1) % this.maxSamples;

    // Track count until buffer is full
    if (this.count < this.maxSamples) {
      this.count++;
    }
  }

  /**
   * Get position at a specific timestamp using linear interpolation
   * @param {number} targetTimestamp - Timestamp to query (milliseconds)
   * @returns {{x: number, y: number, found: boolean}} - Interpolated position
   */
  getPositionAt(targetTimestamp) {
    if (this.count === 0) {
      return { x: 0, y: 0, found: false };
    }

    // If we only have one sample, return it
    if (this.count === 1) {
      const index = (this.head - 1 + this.maxSamples) % this.maxSamples;
      return {
        x: this.positions[index * 2],
        y: this.positions[index * 2 + 1],
        found: true
      };
    }

    // Find the two samples that bracket the target timestamp
    let beforeIndex = -1;
    let afterIndex = -1;
    let beforeTime = -Infinity;
    let afterTime = Infinity;

    // Search through buffer (circular iteration)
    for (let i = 0; i < this.count; i++) {
      const index = (this.head - 1 - i + this.maxSamples) % this.maxSamples;
      const time = this.timestamps[index];

      if (time <= targetTimestamp && time > beforeTime) {
        beforeIndex = index;
        beforeTime = time;
      }

      if (time >= targetTimestamp && time < afterTime) {
        afterIndex = index;
        afterTime = time;
      }
    }

    // Case 1: Target is before all samples (too old)
    if (beforeIndex === -1) {
      // Return oldest sample we have
      const oldestIndex = (this.head - this.count + this.maxSamples) % this.maxSamples;
      return {
        x: this.positions[oldestIndex * 2],
        y: this.positions[oldestIndex * 2 + 1],
        found: false // Indicate we're extrapolating
      };
    }

    // Case 2: Target is after all samples (too recent)
    if (afterIndex === -1) {
      // Return newest sample we have
      const newestIndex = (this.head - 1 + this.maxSamples) % this.maxSamples;
      return {
        x: this.positions[newestIndex * 2],
        y: this.positions[newestIndex * 2 + 1],
        found: false // Indicate we're extrapolating
      };
    }

    // Case 3: Exact match
    if (beforeIndex === afterIndex) {
      return {
        x: this.positions[beforeIndex * 2],
        y: this.positions[beforeIndex * 2 + 1],
        found: true
      };
    }

    // Case 4: Interpolate between two samples
    const timeDiff = afterTime - beforeTime;
    const t = (targetTimestamp - beforeTime) / timeDiff;

    const x1 = this.positions[beforeIndex * 2];
    const y1 = this.positions[beforeIndex * 2 + 1];
    const x2 = this.positions[afterIndex * 2];
    const y2 = this.positions[afterIndex * 2 + 1];

    return {
      x: x1 + (x2 - x1) * t,
      y: y1 + (y2 - y1) * t,
      found: true
    };
  }

  /**
   * Get the most recent position
   * @returns {{x: number, y: number, timestamp: number, found: boolean}}
   */
  getLatest() {
    if (this.count === 0) {
      return { x: 0, y: 0, timestamp: 0, found: false };
    }

    const index = (this.head - 1 + this.maxSamples) % this.maxSamples;
    return {
      x: this.positions[index * 2],
      y: this.positions[index * 2 + 1],
      timestamp: this.timestamps[index],
      found: true
    };
  }

  /**
   * Get the oldest timestamp in the buffer
   * @returns {number} - Oldest timestamp in milliseconds
   */
  getOldestTimestamp() {
    if (this.count === 0) return 0;

    const oldestIndex = (this.head - this.count + this.maxSamples) % this.maxSamples;
    return this.timestamps[oldestIndex];
  }

  /**
   * Get the newest timestamp in the buffer
   * @returns {number} - Newest timestamp in milliseconds
   */
  getNewestTimestamp() {
    if (this.count === 0) return 0;

    const newestIndex = (this.head - 1 + this.maxSamples) % this.maxSamples;
    return this.timestamps[newestIndex];
  }

  /**
   * Clear all samples from the buffer
   */
  clear() {
    this.head = 0;
    this.count = 0;
  }

  /**
   * Get debug information about the buffer
   * @returns {object} - Debug information
   */
  getDebugInfo() {
    if (this.count === 0) {
      return { count: 0, samples: [] };
    }

    const samples = [];
    for (let i = 0; i < this.count; i++) {
      const index = (this.head - 1 - i + this.maxSamples) % this.maxSamples;
      samples.push({
        x: this.positions[index * 2],
        y: this.positions[index * 2 + 1],
        timestamp: this.timestamps[index]
      });
    }

    return {
      count: this.count,
      maxSamples: this.maxSamples,
      oldestTimestamp: this.getOldestTimestamp(),
      newestTimestamp: this.getNewestTimestamp(),
      samples
    };
  }
}

export default CircularBuffer;
