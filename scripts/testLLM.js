// File: scripts/testLLM.js
import path              from 'path';
import { fileURLToPath } from 'url';
import dotenv            from 'dotenv';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.resolve(__dirname, '..', '.env') });

import { createProvider } from '../src/llm/ProviderFactory.js';

(async () => {
  try {
    const provider = createProvider();
    const snapshot = { dummy: true };
    const res      = await provider.generate(snapshot);

    // Fully expand the nested object:
    console.log('Full provider result:\n', JSON.stringify(res, null, 2));
  } catch (err) {
    console.error('LLM test error:', err);
    process.exit(1);
  }
})();
