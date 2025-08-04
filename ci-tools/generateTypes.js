// test comment
#!/usr/bin/env node
// @ts-nocheck
// ci-tools/generateTypes.js
// Converts the brick union JSON schema into a naive TypeScript declaration.

const fs = require('fs');
const path = require('path');

const SCHEMA_PATH = path.resolve(__dirname, '..', 'src', 'registry', 'brick-union.schema.json');
const OUT_DIR = path.resolve(__dirname, '..', 'types');
const OUT_FILE = path.join(OUT_DIR, 'brick-union.d.ts');

function main() {
  if (!fs.existsSync(SCHEMA_PATH)) {
    console.error('❌ Union schema not found – run generateUnionSchema.js first');
    process.exit(1);
  }

  const schema = JSON.parse(fs.readFileSync(SCHEMA_PATH, 'utf8'));
  const ids = (schema.oneOf || []).map(s => `'${s.$id}'`);

  if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

  const content = `// AUTO-GENERATED – do not edit.
export type BrickType = ${ids.join(' | ') || 'never'};

export interface BrickBase { type: BrickType; /* additional params per brick */ }
`;

  fs.writeFileSync(OUT_FILE, content, 'utf8');
  console.log(`✅ Wrote ${OUT_FILE}`);
}

if (require.main === module) {
  main();
} 