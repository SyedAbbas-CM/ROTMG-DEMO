// src/registry/HotReloadWatcher.js
// Simple file-system watcher for capability directories using Node's fs.watch.

import fs from 'fs';
import path from 'path';
import { EventEmitter } from 'events';

export function watchCapabilities({ baseDir = path.resolve('capabilities') } = {}) {
  const emitter = new EventEmitter();

  if (!fs.existsSync(baseDir)) {
    process.nextTick(() =>
      emitter.emit('error', new Error(`Directory '${baseDir}' does not exist`)),
    );
    return emitter;
  }

  const watcher = fs.watch(baseDir, { recursive: true }, (eventType, filename) => {
    if (!filename) return;
    emitter.emit('change', {
      type: eventType,
      file: path.join(baseDir, filename),
    });
  });

  emitter.close = () => watcher.close();
  return emitter;
} 