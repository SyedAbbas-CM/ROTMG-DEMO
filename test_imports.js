// Simple test to check if basic imports work
console.log('Testing imports...');

try {
  console.log('Importing express...');
  const express = await import('express');
  console.log('✓ Express imported');

  console.log('Importing MapManager...');
  const { MapManager } = await import('./src/MapManager.js');
  console.log('✓ MapManager imported');

  console.log('Importing ItemManager...');
  const { ItemManager } = await import('./src/ItemManager.js');
  console.log('✓ ItemManager imported');

  console.log('Importing BehaviorSystem...');
  const BehaviorSystem = await import('./src/BehaviorSystem.js');
  console.log('✓ BehaviorSystem imported');

  console.log('All basic imports successful!');
} catch (error) {
  console.error('Import failed:', error);
}