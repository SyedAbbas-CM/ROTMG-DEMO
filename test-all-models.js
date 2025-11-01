#!/usr/bin/env node
/**
 * Test script to verify all Google AI Studio models
 * Usage: node test-all-models.js
 */

import { config } from 'dotenv';
config();

import { createProvider, ModelPresets } from './src/boss/llm/ProviderFactory.js';

const TEST_SNAPSHOT = {
  boss: { x: 32, y: 32, hp: 500, maxHp: 1000, phase: 0 },
  players: [{ id: 'p1', x: 40, y: 40, hp: 100, distance: 11.3 }],
  bullets: { active: 0 },
  timestamp: Date.now()
};

const MODELS_TO_TEST = [
  { name: 'Gemini 2.5 Flash-Lite (RECOMMENDED TACTICAL)', model: ModelPresets.TACTICAL_CAPACITY },
  { name: 'Gemini 2.0 Flash-Lite (FASTEST)', model: ModelPresets.TACTICAL_FASTEST },
  { name: 'Gemini 2.5 Flash', model: ModelPresets.TACTICAL_BALANCED },
  { name: 'Gemini 2.5 Pro (BEST STRATEGIC)', model: ModelPresets.STRATEGIC_BEST },
  { name: 'Gemini 2.0 Flash', model: ModelPresets.GEMINI_2_0_FLASH },
];

async function testModel(modelConfig) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`Testing: ${modelConfig.name}`);
  console.log(`Model ID: ${modelConfig.model}`);
  console.log('='.repeat(60));

  try {
    const provider = createProvider({
      backend: 'gemini',
      model: modelConfig.model,
      temperature: 0.7,
      maxTokens: 1024
    });

    console.log('✓ Provider created');

    const startTime = Date.now();
    const result = await provider.generate(TEST_SNAPSHOT);
    const duration = Date.now() - startTime;

    console.log('\n✅ SUCCESS!');
    console.log(`   Duration: ${duration}ms`);
    console.log(`   Tokens: ${result.tokens || 'N/A'}`);
    console.log(`   Response: ${JSON.stringify(result.json, null, 2).substring(0, 200)}...`);

    return {
      success: true,
      model: modelConfig.name,
      duration,
      tokens: result.tokens || 0
    };

  } catch (err) {
    console.log('\n❌ FAILED');
    console.log(`   Error: ${err.message}`);

    return {
      success: false,
      model: modelConfig.name,
      error: err.message
    };
  }
}

async function main() {
  console.log('═══════════════════════════════════════════════════════════');
  console.log('  Google AI Studio Model Test Suite');
  console.log('═══════════════════════════════════════════════════════════\n');

  if (!process.env.GOOGLE_API_KEY) {
    console.log('❌ GOOGLE_API_KEY not found in environment!');
    console.log('\nPlease set it in .env file or run:');
    console.log('   GOOGLE_API_KEY=your_key node test-all-models.js\n');
    process.exit(1);
  }

  console.log('✓ API Key found');
  console.log(`✓ Testing ${MODELS_TO_TEST.length} models\n`);

  const results = [];

  for (const modelConfig of MODELS_TO_TEST) {
    const result = await testModel(modelConfig);
    results.push(result);

    // Small delay between tests to respect rate limits
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  // Print summary
  console.log('\n\n' + '═'.repeat(60));
  console.log('  TEST SUMMARY');
  console.log('═'.repeat(60) + '\n');

  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);

  console.log(`Total Tests: ${results.length}`);
  console.log(`✅ Passed: ${successful.length}`);
  console.log(`❌ Failed: ${failed.length}\n`);

  if (successful.length > 0) {
    console.log('Successful Models:');
    successful.forEach(r => {
      console.log(`   ✓ ${r.model} (${r.duration}ms, ${r.tokens} tokens)`);
    });
  }

  if (failed.length > 0) {
    console.log('\nFailed Models:');
    failed.forEach(r => {
      console.log(`   ✗ ${r.model}`);
      console.log(`     Error: ${r.error}`);
    });
  }

  console.log('\n' + '═'.repeat(60));

  if (successful.length > 0) {
    console.log('✅ At least one model is working! You\'re ready to use the Two-Tier LLM Controller.');
  } else {
    console.log('❌ All models failed. Check your API key and network connection.');
    process.exit(1);
  }
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
