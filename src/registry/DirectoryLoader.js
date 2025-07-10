// src/registry/DirectoryLoader.js
// Auto-discovery utility for capability bricks. Scans <baseDir>/<Category>/<Name>/<version>/schema.json + implementation.js
// and returns AJV validators, compile functions, and raw schemas.

import fs from 'fs';
import path from 'path';
import Ajv from 'ajv';
import { pathToFileURL } from 'url';

/**
 * Recursively discover capability schemas and implementations.
 * @param {string} [baseDir] - Directory to scan. Defaults to "<repo>/capabilities".
 * @returns {Promise<{ validators: Record<string, Ajv.ValidateFunction>, compilers: Record<string,(brick:any)=>Promise<any>>, schemas: Record<string,any> }>}
 */
export async function loadCapabilities(baseDir = path.resolve('capabilities')) {
  const ajv = new Ajv({ strict: false });
  const validators = {};
  const compilers = {};
  const schemas = {};

  if (!fs.existsSync(baseDir)) {
    console.warn(`[DirectoryLoader] Base directory '${baseDir}' does not exist – returning empty registry`);
    return { validators, compilers, schemas };
  }

  // Depth-first scan for schema.json files
  const walk = dir => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(full);
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
          validators[$id] = ajv.compile(schema);
          schemas[$id] = schema;

          // Resolve implementation.js path in same folder
          const implPath = path.join(path.dirname(full), 'implementation.js');
          let cachedCompiler = null;
          compilers[$id] = async brick => {
            if (!cachedCompiler) {
              const mod = await import(pathToFileURL(implPath).href);
              if (typeof mod.compile !== 'function') {
                throw new Error(`[DirectoryLoader] ${implPath} does not export 'compile'`);
              }
              cachedCompiler = mod.compile;
            }
            return cachedCompiler(brick);
          };
        } catch (err) {
          console.error(`[DirectoryLoader] Failed to load schema at ${full}:`, err);
        }
      }
    }
  };

  walk(baseDir);

  return { validators, compilers, schemas };
} 