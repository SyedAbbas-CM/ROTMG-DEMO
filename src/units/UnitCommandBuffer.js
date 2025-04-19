// server/src/units/UnitCommandBuffer.js
const SIZE = 16384;               // circular queue

export default class UnitCommandBuffer {
  constructor() {
    this.buf = new Array(SIZE);
    this.head = 0; this.tail = 0;
  }
  push(cmd) {                      // {unitIdx, kind, tx, ty}
    this.buf[this.head] = cmd;
    this.head = (this.head + 1) & (SIZE-1);
  }
  /** pull all pending commands (called once per tick) */
  flush(fn) {
    while (this.tail !== this.head) {
      fn(this.buf[this.tail]);
      this.tail = (this.tail + 1) & (SIZE-1);
    }
  }
}
