#!/usr/bin/env node
// ci-tools/generateUnionSchema.js
// Scans capability folders recursively and creates a union JSON Schema at
// src/registry/brick-union.schema.json.  Intended for local dev + CI.
// Usage: node ci-tools/generateUnionSchema.js

import fs from 'fs';
import path from 'path';

const CAP_DIR = path.resolve('capabilities');
const OUT_PATH = path.resolve('src','registry','brick-union.schema.json');

function findSchemas(dir) {
  const items = fs.readdirSync(dir, { withFileTypes: true });
  const schemas = [];
  for (const ent of items) {
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) {
      schemas.push(...findSchemas(full));
    } else if (ent.isFile() && ent.name === 'schema.json') {
      try {
        const schema = JSON.parse(fs.readFileSync(full,'utf8'));
        schemas.push(schema);
      } catch(err) {
        console.warn('[generateUnionSchema] Failed to parse', full, err.message);
      }
    }
  }
  return schemas;
}

function main() {
  if (!fs.existsSync(CAP_DIR)) {
    console.error(`❌ Capabilities directory '${CAP_DIR}' not found`);
    process.exit(1);
  }

  const schemas = findSchemas(CAP_DIR);
  if (schemas.length === 0) {
    console.error('❌ No capability schemas found');
    process.exit(1);
  }

  const union = {
    $schema: 'http://json-schema.org/draft-07/schema#',
    title: 'Capability Brick Union',
    description: 'Auto-generated union of every capability brick schema',
    oneOf: schemas
  };

  fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
  fs.writeFileSync(OUT_PATH, JSON.stringify(union, null, 2));
  console.log(`✅ Wrote union schema with ${schemas.length} entries → ${OUT_PATH}`);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
} 