#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { createWriteStream } from 'fs';
import { pipeline } from 'stream';
import { promisify } from 'util';
import archiver from 'archiver';

const pipelineAsync = promisify(pipeline);

/**
 * Script to extract all code files from the ROTMG RTS project
 * Creates a zip file with all source code for external analysis
 */

const PROJECT_ROOT = process.cwd();
const OUTPUT_FILE = 'rotmg-code-export.zip';

// File extensions to include
const CODE_EXTENSIONS = new Set([
  '.js', '.ts', '.jsx', '.tsx',
  '.json', '.md', '.txt',
  '.html', '.css', '.scss',
  '.glsl', '.vert', '.frag',
  '.cpp', '.c', '.h',
  '.wasm'
]);

// Directories to exclude
const EXCLUDE_DIRS = new Set([
  'node_modules',
  '.git',
  '.DS_Store',
  'logs',
  'tmp',
  'temp',
  'build',
  'dist',
  'coverage',
  '.nyc_output',
  'ROTMG-CORE-FILES'
]);

// Files to exclude
const EXCLUDE_FILES = new Set([
  '.env',
  '.env.local',
  '.env.development',
  '.env.production',
  'package-lock.json',
  'yarn.lock',
  '.gitignore',
  '.DS_Store'
]);

/**
 * Check if a file should be included based on extension
 */
function shouldIncludeFile(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  const filename = path.basename(filePath);
  
  // Skip excluded files
  if (EXCLUDE_FILES.has(filename)) {
    return false;
  }
  
  // Include if extension matches
  if (CODE_EXTENSIONS.has(ext)) {
    return true;
  }
  
  // Include specific files without extensions
  const noExtFiles = ['Dockerfile', 'README', 'LICENSE', 'CHANGELOG'];
  if (noExtFiles.some(name => filename.startsWith(name))) {
    return true;
  }
  
  return false;
}

/**
 * Check if directory should be processed
 */
function shouldProcessDirectory(dirName) {
  return !EXCLUDE_DIRS.has(dirName) && !dirName.startsWith('.');
}

/**
 * Recursively collect all code files
 */
function collectFiles(dir, baseDir = dir) {
  const files = [];
  
  try {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      
      if (entry.isDirectory()) {
        if (shouldProcessDirectory(entry.name)) {
          files.push(...collectFiles(fullPath, baseDir));
        }
      } else if (entry.isFile()) {
        if (shouldIncludeFile(fullPath)) {
          const relativePath = path.relative(baseDir, fullPath);
          files.push({
            fullPath,
            relativePath,
            size: fs.statSync(fullPath).size
          });
        }
      }
    }
  } catch (error) {
    console.warn(`Warning: Could not read directory ${dir}: ${error.message}`);
  }
  
  return files;
}

/**
 * Create the zip file
 */
async function createZip() {
  console.log('🔍 Scanning for code files...');
  
  const files = collectFiles(PROJECT_ROOT);
  
  console.log(`📁 Found ${files.length} code files to archive`);
  
  // Calculate total size
  const totalSize = files.reduce((sum, file) => sum + file.size, 0);
  const totalSizeMB = (totalSize / (1024 * 1024)).toFixed(2);
  
  console.log(`📦 Total size: ${totalSizeMB} MB`);
  
  // Create output stream
  const output = createWriteStream(OUTPUT_FILE);
  const archive = archiver('zip', {
    zlib: { level: 9 } // Maximum compression
  });
  
  // Handle archiver events
  output.on('close', () => {
    const finalSize = (archive.pointer() / (1024 * 1024)).toFixed(2);
    console.log(`✅ Archive created: ${OUTPUT_FILE}`);
    console.log(`📏 Compressed size: ${finalSize} MB`);
    console.log(`🗜️  Compression ratio: ${((1 - archive.pointer() / totalSize) * 100).toFixed(1)}%`);
  });
  
  archive.on('error', (err) => {
    throw err;
  });
  
  archive.on('progress', (progress) => {
    const percent = ((progress.entries.processed / files.length) * 100).toFixed(1);
    process.stdout.write(`\r⏳ Progress: ${percent}% (${progress.entries.processed}/${files.length})`);
  });
  
  // Pipe archive data to file
  archive.pipe(output);
  
  // Add files to archive
  console.log('📤 Adding files to archive...');
  
  for (const file of files) {
    try {
      archive.file(file.fullPath, { name: file.relativePath });
    } catch (error) {
      console.warn(`\nWarning: Could not add file ${file.relativePath}: ${error.message}`);
    }
  }
  
  // Finalize the archive
  await archive.finalize();
  
  return new Promise((resolve, reject) => {
    output.on('close', resolve);
    output.on('error', reject);
  });
}

/**
 * Generate a summary file
 */
function generateSummary(files) {
  const summary = {
    timestamp: new Date().toISOString(),
    projectName: 'ROTMG RTS Game',
    totalFiles: files.length,
    filesByExtension: {},
    directoryStructure: {},
    largestFiles: files
      .sort((a, b) => b.size - a.size)
      .slice(0, 10)
      .map(f => ({
        path: f.relativePath,
        size: `${(f.size / 1024).toFixed(1)} KB`
      }))
  };
  
  // Count files by extension
  files.forEach(file => {
    const ext = path.extname(file.relativePath).toLowerCase() || 'no-extension';
    summary.filesByExtension[ext] = (summary.filesByExtension[ext] || 0) + 1;
  });
  
  // Build directory structure
  files.forEach(file => {
    const dir = path.dirname(file.relativePath);
    const parts = dir === '.' ? [] : dir.split(path.sep);
    
    let current = summary.directoryStructure;
    for (const part of parts) {
      if (!current[part]) {
        current[part] = {};
      }
      current = current[part];
    }
  });
  
  return summary;
}

/**
 * Main execution
 */
async function main() {
  try {
    console.log('🚀 ROTMG Code Extraction Tool');
    console.log('==============================');
    
    const files = collectFiles(PROJECT_ROOT);
    
    // Generate summary
    const summary = generateSummary(files);
    
    // Write summary file
    fs.writeFileSync('extraction-summary.json', JSON.stringify(summary, null, 2));
    
    // Create the zip
    await createZip();
    
    console.log('\n\n🎉 Code extraction completed successfully!');
    console.log(`📄 Summary: extraction-summary.json`);
    console.log(`📦 Archive: ${OUTPUT_FILE}`);
    console.log('\nYou can now send this zip file to ChatGPT for analysis.');
    
  } catch (error) {
    console.error('❌ Error during extraction:', error);
    process.exit(1);
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export { collectFiles, createZip, generateSummary };