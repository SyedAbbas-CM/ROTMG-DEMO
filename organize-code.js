import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Get current directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Paths to source directories
const frontendDir = path.join(__dirname, 'public');
const backendDir = path.join(__dirname);

// Output files
const frontendPart1 = path.join(__dirname, 'frontend-part1.txt');
const frontendPart2 = path.join(__dirname, 'frontend-part2.txt');
const frontendPart3 = path.join(__dirname, 'frontend-part3.txt');
const backendPart = path.join(__dirname, 'backend-part.txt');

// Clear existing output files
fs.writeFileSync(frontendPart1, '');
fs.writeFileSync(frontendPart2, '');
fs.writeFileSync(frontendPart3, '');
fs.writeFileSync(backendPart, '');

// Function to check if a path should be ignored
function shouldIgnore(filePath) {
  return filePath.includes('assets') || 
         filePath.includes('node_modules') || 
         filePath.includes('.git') ||
         filePath.endsWith('.md') ||
         filePath.endsWith('.json') ||
         filePath.endsWith('.log') ||
         filePath.endsWith('.DS_Store');
}

// Function to get all files recursively
function getAllFiles(dirPath, arrayOfFiles = []) {
  const files = fs.readdirSync(dirPath);

  files.forEach(file => {
    const filePath = path.join(dirPath, file);
    
    if (shouldIgnore(filePath)) {
      return;
    }
    
    if (fs.statSync(filePath).isDirectory()) {
      arrayOfFiles = getAllFiles(filePath, arrayOfFiles);
    } else {
      arrayOfFiles.push(filePath);
    }
  });

  return arrayOfFiles;
}

// Get all frontend files (excluding assets)
let frontendFiles = getAllFiles(frontendDir).filter(file => !file.includes('assets'));

// Get all backend files
let backendFiles = getAllFiles(backendDir).filter(file => {
  // Exclude frontend files and our script
  return !file.includes('public') && 
         !file.includes('organize-code.js') &&
         !file.includes('frontend-part') &&
         !file.includes('backend-part');
});

// Function to append file content to output file
function appendFileContent(filePath, outputFile) {
  const content = fs.readFileSync(filePath, 'utf8');
  const relativePath = path.relative(__dirname, filePath);
  
  fs.appendFileSync(
    outputFile, 
    `\n\n// =========================================\n` +
    `// FILE: ${relativePath}\n` +
    `// =========================================\n\n` +
    content
  );
}

// Split frontend files into 3 roughly equal parts
const frontendFilesPerPart = Math.ceil(frontendFiles.length / 3);
const frontendParts = [
  frontendFiles.slice(0, frontendFilesPerPart),
  frontendFiles.slice(frontendFilesPerPart, frontendFilesPerPart * 2),
  frontendFiles.slice(frontendFilesPerPart * 2)
];

console.log(`Frontend files: ${frontendFiles.length} (${frontendFilesPerPart} per part)`);
console.log(`Backend files: ${backendFiles.length}`);

// Process frontend files
frontendParts[0].forEach(file => appendFileContent(file, frontendPart1));
frontendParts[1].forEach(file => appendFileContent(file, frontendPart2));
frontendParts[2].forEach(file => appendFileContent(file, frontendPart3));

// Process backend files
backendFiles.forEach(file => appendFileContent(file, backendPart));

console.log('Files have been processed and written to:');
console.log(`- ${path.relative(__dirname, frontendPart1)}`);
console.log(`- ${path.relative(__dirname, frontendPart2)}`);
console.log(`- ${path.relative(__dirname, frontendPart3)}`);
console.log(`- ${path.relative(__dirname, backendPart)}`); 