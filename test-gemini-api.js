#!/usr/bin/env node
/**
 * Test script to verify Gemini API connectivity
 * Usage: GOOGLE_API_KEY=your_key node test-gemini-api.js
 */

// Load .env file
import { config } from 'dotenv';
config();

import { createProvider } from './src/boss/llm/ProviderFactory.js';

async function testGeminiAPI() {
  console.log('=== Gemini API Test ===\n');

  // Check environment configuration
  console.log('Configuration:');
  console.log('  LLM_BACKEND:', process.env.LLM_BACKEND || 'gemini (default)');
  console.log('  LLM_MODEL:', process.env.LLM_MODEL || 'models/gemini-2.5-flash (default)');
  console.log('  GOOGLE_API_KEY:', process.env.GOOGLE_API_KEY ? '✓ Set' : '✗ Not set');
  console.log();

  const backend = process.env.LLM_BACKEND || 'gemini';

  if (backend === 'gemini' && !process.env.GOOGLE_API_KEY) {
    console.log('❌ GOOGLE_API_KEY is not set!');
    console.log('\nTo test the Gemini API, you need to:');
    console.log('1. Get an API key from https://makersuite.google.com/app/apikey');
    console.log('2. Create a .env file: cp env.example.txt .env');
    console.log('3. Add your key: GOOGLE_API_KEY=your_key_here');
    console.log('\nOr run this script with: GOOGLE_API_KEY=your_key node test-gemini-api.js');
    console.log('\nAlternatively, use mock provider: LLM_BACKEND=mock node test-gemini-api.js');
    process.exit(1);
  }

  if (backend === 'mock') {
    console.log('ℹ️  Using mock provider (no API calls will be made)\n');
  }

  try {
    console.log('Creating provider...');
    const provider = createProvider();
    console.log('✓ Provider created successfully\n');

    console.log('Testing API call with sample boss snapshot...');
    const sampleSnapshot = {
      boss: {
        x: 32,
        y: 32,
        hp: 500,
        maxHp: 1000,
        phase: 0
      },
      players: [
        { id: 'p1', x: 40, y: 40, hp: 100, distance: 11.3 }
      ],
      bullets: { active: 0 },
      timestamp: Date.now()
    };

    const startTime = Date.now();
    const result = await provider.generate(sampleSnapshot);
    const duration = Date.now() - startTime;

    console.log('\n✓ API call successful!');
    console.log('\nResponse:');
    console.log('  Duration:', duration, 'ms');
    console.log('  Tokens:', result.tokens);
    console.log('  Plan:', JSON.stringify(result.json, null, 2));
    console.log('\n✅ Gemini API is working correctly!');

  } catch (err) {
    console.error('\n❌ API test failed:');
    console.error('  Error:', err.message);

    if (err.message.includes('API key')) {
      console.error('\n  Your API key may be invalid or expired.');
      console.error('  Get a new one from: https://makersuite.google.com/app/apikey');
    } else if (err.message.includes('quota')) {
      console.error('\n  API quota exceeded. Check your usage limits.');
    }

    process.exit(1);
  }
}

testGeminiAPI();
