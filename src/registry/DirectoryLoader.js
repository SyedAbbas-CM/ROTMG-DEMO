// src/registry/DirectoryLoader.js
// Auto-discovery utility for capability bricks. Scans <baseDir>/<Category>/<Name>/<version>/schema.json + implementation.js
// and returns AJV validators, compile functions, and raw schemas.

import fs from 'fs';
import path from 'path';
import Ajv from 'ajv';
import { pathToFileURL } from 'url';
import { EventEmitter } from 'events';

// Safety overrides injected into every schema (acts as a meta-schema)
const SAFETY_OVERRIDES = {
  properties: {
    projectiles: { maximum: 400 },
    speed:       { maximum: 100 },
    radius:      { maximum: 15 }
  }
};

/**
 * Load all capability schemas & compilers under baseDir.
 * Also writes an aggregated union schema to src/registry/brick-union.schema.json
 */
export async function loadCapabilities(baseDir = path.resolve('src', 'capabilities')) {
  const ajv = new Ajv({ strict: false });
  const validators = {};
  const compilers = {};
  const schemas = {};
  const invokers = {};

  if (!fs.existsSync(baseDir)) {
    console.warn(`[DirectoryLoader] Base directory '${baseDir}' does not exist – returning empty registry`);
    return { validators, compilers, schemas };
  }

  // Depth-first scan for schema.json files
  const walk = async dir => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        await walk(full);
        continue;
      }
      if (entry.isFile() && entry.name === 'schema.json') {
        try {
          const schema = JSON.parse(fs.readFileSync(full, 'utf8'));
          const { $id } = schema;
          if (!$id) {
            console.warn(`[DirectoryLoader] schema at ${full} missing $id, skipping`);
            continue;
          }

          // Skip entirely if a validator for this $id is already present (prevents recursion)
          if (validators[$id]) return;

          // Build a safe schema without duplicating the $id inside allOf – Ajv
          // throws if it encounters the same $id twice during recursive compile.
          const inner = { ...schema };
          delete inner.$id;
          const safeSchema = { ...schema, allOf: [inner, SAFETY_OVERRIDES] };

          // If Ajv already knows this $id (e.g., compiled earlier) skip.
          if (ajv.getSchema($id)) return;

          validators[$id] = ajv.compile(safeSchema);
          schemas[$id]    = safeSchema;

          // ------------------------------------------------------------
          // Eagerly import implementation so its invoke() is registered
          // ------------------------------------------------------------
          const implPath = path.join(path.dirname(full), 'implementation.js');
          try {
            const mod = await import(pathToFileURL(implPath).href);
            if (typeof mod.compile === 'function') {
              compilers[$id] = mod.compile;
            }
            if (typeof mod.invoke === 'function') {
              invokers[$id] = mod.invoke;
            }
          } catch (err) {
            console.warn(`[DirectoryLoader] Missing or invalid implementation for ${$id}`, err.message);
          }
        } catch (err) {
          console.error(`[DirectoryLoader] Failed to load schema at ${full}:`, err);
        }
      }
    }
  };

  await walk(baseDir);

  // -------------------------------------------------------------------
  // Emit union schema (tools & editor typings rely on this)
  // -------------------------------------------------------------------
  try {
    const outPath = path.resolve('src', 'registry', 'brick-union.schema.json');
    const union = {
      $schema: 'http://json-schema.org/draft-07/schema#',
      title: 'Capability Brick Union',
      description: 'Auto-generated. DO NOT EDIT.',
      oneOf: Object.values(schemas)
    };
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    fs.writeFileSync(outPath, JSON.stringify(union, null, 2));
  } catch (err) {
    console.warn('[DirectoryLoader] Failed to write union schema', err.message);
  }

  return { validators, compilers, invokers, schemas };
}

/* =====================================================================
   Hot-reload helper
   ===================================================================== */

export function watchCapabilities(baseDir = path.resolve('src', 'capabilities')) {
  const emitter = new EventEmitter();
  if (!fs.existsSync(baseDir)) return emitter;

  const watcher = fs.watch(baseDir, { recursive: true }, async () => {
    const payload = await loadCapabilities(baseDir);
    emitter.emit('change', payload);
  });

  emitter.on('close', () => watcher.close());
  return emitter;
} 