// extract_systems.js
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Get current directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Define extraction targets
const extractionTargets = [
    // Collision System - Frontend
    {
        source: 'public/src/collision/ClientCollisionManager.js',
        target: 'extraction_output/systems/collision_system/frontend/ClientCollisionManager.js.txt'
    },
    {
        source: 'public/src/collision/collisionSystem.js',
        target: 'extraction_output/systems/collision_system/frontend/collisionSystem.js.txt'
    },
    {
        source: 'public/src/shared/spatialGrid.js',
        target: 'extraction_output/systems/collision_system/frontend/spatialGrid.js.txt'
    },
    {
        source: 'public/src/game/updateCharacter.js',
        target: 'extraction_output/systems/collision_system/frontend/updateCharacter.js.txt'
    },

    // Collision System - Backend
    {
        source: 'src/CollisionManager.js',
        target: 'extraction_output/systems/collision_system/backend/CollisionManager.js.txt'
    },
    {
        source: 'src/networkHandlers/collisionHandler.js',
        target: 'extraction_output/systems/collision_system/backend/collisionHandler.js.txt'
    },

    // Map System - Frontend
    {
        source: 'public/src/map/ClientMapManager.js',
        target: 'extraction_output/systems/map_system/frontend/ClientMapManager.js.txt'
    },
    {
        source: 'public/src/map/tile.js',
        target: 'extraction_output/systems/map_system/frontend/tile.js.txt'
    },
    {
        source: 'public/src/camera.js',
        target: 'extraction_output/systems/map_system/frontend/camera.js.txt'
    },
    {
        source: 'public/src/utils/coordinateUtils.js',
        target: 'extraction_output/systems/map_system/frontend/coordinateUtils.js.txt'
    },

    // Map System - Backend
    {
        source: 'src/MapManager.js',
        target: 'extraction_output/systems/map_system/backend/MapManager.js.txt'
    },
    {
        source: 'src/networkHandlers/chunkRequest.js',
        target: 'extraction_output/systems/map_system/backend/chunkRequest.js.txt'
    },
    {
        source: 'src/world/AdvancedPerlinNoise.js',
        target: 'extraction_output/systems/map_system/backend/AdvancedPerlinNoise.js.txt'
    }
];

// Function to copy a file
function copyFile(source, target) {
    try {
        if (fs.existsSync(source)) {
            const content = fs.readFileSync(source, 'utf8');
            // Ensure target directory exists
            const targetDir = path.dirname(target);
            if (!fs.existsSync(targetDir)) {
                fs.mkdirSync(targetDir, { recursive: true });
            }
            // Write to target file
            fs.writeFileSync(target, content);
            console.log(`✅ Copied: ${source} -> ${target}`);
            return true;
        } else {
            console.log(`❌ Source file not found: ${source}`);
            return false;
        }
    } catch (error) {
        console.error(`❌ Error copying ${source}: ${error.message}`);
        return false;
    }
}

// Create a summary file for each system
function createSummaryFile(system, successfulCopies) {
    const summaryPath = `extraction_output/systems/${system}/${system}_summary.txt`;
    
    let summaryContent = `# ${system.replace('_', ' ').toUpperCase()} SUMMARY\n\n`;
    summaryContent += `Extraction Date: ${new Date().toISOString()}\n\n`;
    
    // Add list of extracted files
    summaryContent += `## Extracted Files:\n\n`;
    
    // Frontend files
    summaryContent += `### Frontend:\n`;
    const frontendFiles = successfulCopies.filter(file => file.includes(`/${system}/frontend/`));
    frontendFiles.forEach(file => {
        const fileName = path.basename(file);
        summaryContent += `- ${fileName}\n`;
    });
    
    // Backend files
    summaryContent += `\n### Backend:\n`;
    const backendFiles = successfulCopies.filter(file => file.includes(`/${system}/backend/`));
    backendFiles.forEach(file => {
        const fileName = path.basename(file);
        summaryContent += `- ${fileName}\n`;
    });
    
    // Add instructions for ChatGPT
    summaryContent += `\n## Usage Instructions for ChatGPT\n\n`;
    summaryContent += `To analyze this system, upload each file individually or as a group. For best results:\n`;
    summaryContent += `1. Ask specific questions about how the ${system.replace('_', ' ')} works\n`;
    summaryContent += `2. Inquire about the relationship between frontend and backend components\n`;
    summaryContent += `3. Focus on understanding the coordinate system and conversion between world and tile coordinates\n`;
    summaryContent += `4. For collision issues, pay attention to the boundary detection logic and collision resolution\n`;
    
    fs.writeFileSync(summaryPath, summaryContent);
    console.log(`✅ Created summary file: ${summaryPath}`);
}

// Create a main index file
function createIndexFile(successfulCopies) {
    const indexPath = 'extraction_output/systems/system_index.txt';
    
    let indexContent = `# GAME SYSTEMS EXTRACTION INDEX\n\n`;
    indexContent += `Extraction Date: ${new Date().toISOString()}\n\n`;
    
    indexContent += `## Systems Extracted:\n\n`;
    
    // Collision System
    indexContent += `### Collision System\n`;
    indexContent += `- Frontend: ${successfulCopies.filter(file => file.includes('/collision_system/frontend/')).length} files\n`;
    indexContent += `- Backend: ${successfulCopies.filter(file => file.includes('/collision_system/backend/')).length} files\n`;
    
    // Map System
    indexContent += `\n### Map System\n`;
    indexContent += `- Frontend: ${successfulCopies.filter(file => file.includes('/map_system/frontend/')).length} files\n`;
    indexContent += `- Backend: ${successfulCopies.filter(file => file.includes('/map_system/backend/')).length} files\n`;
    
    // Add master compilation instructions
    indexContent += `\n## Creating Master Compilations\n\n`;
    indexContent += `To create a single file with all code for each system, run these commands:\n\n`;
    indexContent += `For Collision System:\n`;
    indexContent += "```bash\n";
    indexContent += `cat extraction_output/systems/collision_system/frontend/*.txt > extraction_output/systems/collision_system/frontend_all.txt\n`;
    indexContent += `cat extraction_output/systems/collision_system/backend/*.txt > extraction_output/systems/collision_system/backend_all.txt\n`;
    indexContent += "```\n\n";
    indexContent += `For Map System:\n`;
    indexContent += "```bash\n";
    indexContent += `cat extraction_output/systems/map_system/frontend/*.txt > extraction_output/systems/map_system/frontend_all.txt\n`;
    indexContent += `cat extraction_output/systems/map_system/backend/*.txt > extraction_output/systems/map_system/backend_all.txt\n`;
    indexContent += "```\n\n";
    
    fs.writeFileSync(indexPath, indexContent);
    console.log(`✅ Created index file: ${indexPath}`);
}

// Function to create combined files for direct upload to ChatGPT
function createCombinedFiles(successfulCopies) {
    // Create combined files for each system and section
    const combinations = [
        {
            pattern: '/collision_system/frontend/',
            output: 'extraction_output/systems/collision_system/frontend_all.txt',
            title: 'COLLISION SYSTEM - FRONTEND'
        },
        {
            pattern: '/collision_system/backend/',
            output: 'extraction_output/systems/collision_system/backend_all.txt',
            title: 'COLLISION SYSTEM - BACKEND'
        },
        {
            pattern: '/map_system/frontend/',
            output: 'extraction_output/systems/map_system/frontend_all.txt',
            title: 'MAP SYSTEM - FRONTEND'
        },
        {
            pattern: '/map_system/backend/',
            output: 'extraction_output/systems/map_system/backend_all.txt',
            title: 'MAP SYSTEM - BACKEND'
        }
    ];
    
    combinations.forEach(combo => {
        const files = successfulCopies.filter(file => file.includes(combo.pattern));
        let content = `# ${combo.title}\n\n`;
        content += `Extraction Date: ${new Date().toISOString()}\n\n`;
        
        for (const file of files) {
            if (fs.existsSync(file)) {
                const fileName = path.basename(file);
                content += `\n\n## FILE: ${fileName}\n\n`;
                content += '```javascript\n';
                content += fs.readFileSync(file, 'utf8');
                content += '\n```\n\n';
                content += '---\n';
            }
        }
        
        fs.writeFileSync(combo.output, content);
        console.log(`✅ Created combined file: ${combo.output}`);
    });
    
    // Create super-combined files (one per system)
    const superCombinations = [
        {
            pattern: '/collision_system/',
            output: 'extraction_output/systems/collision_system_all.txt',
            title: 'COMPLETE COLLISION SYSTEM (FRONTEND + BACKEND)'
        },
        {
            pattern: '/map_system/',
            output: 'extraction_output/systems/map_system_all.txt',
            title: 'COMPLETE MAP SYSTEM (FRONTEND + BACKEND)'
        }
    ];
    
    superCombinations.forEach(combo => {
        const files = successfulCopies.filter(file => file.includes(combo.pattern));
        let content = `# ${combo.title}\n\n`;
        content += `Extraction Date: ${new Date().toISOString()}\n\n`;
        content += `This file contains all code related to the ${combo.title.split(' ')[1].toLowerCase()} ${combo.title.split(' ')[2].toLowerCase()} from both frontend and backend.\n\n`;
        
        // First add frontend files
        content += `\n# FRONTEND CODE\n\n`;
        const frontendFiles = files.filter(file => file.includes('/frontend/'));
        for (const file of frontendFiles) {
            if (fs.existsSync(file)) {
                const fileName = path.basename(file);
                const sourceFile = file.replace('extraction_output/systems', '').replace('.txt', '');
                content += `\n\n## FILE: ${sourceFile}\n\n`;
                content += '```javascript\n';
                content += fs.readFileSync(file, 'utf8');
                content += '\n```\n\n';
                content += '---\n';
            }
        }
        
        // Then add backend files
        content += `\n# BACKEND CODE\n\n`;
        const backendFiles = files.filter(file => file.includes('/backend/'));
        for (const file of backendFiles) {
            if (fs.existsSync(file)) {
                const fileName = path.basename(file);
                const sourceFile = file.replace('extraction_output/systems', '').replace('.txt', '');
                content += `\n\n## FILE: ${sourceFile}\n\n`;
                content += '```javascript\n';
                content += fs.readFileSync(file, 'utf8');
                content += '\n```\n\n';
                content += '---\n';
            }
        }
        
        // Add analysis suggestions
        content += `\n# ANALYSIS SUGGESTIONS\n\n`;
        content += `When analyzing this code, focus on:\n\n`;
        content += `1. Coordinate system conversions between world and tile space\n`;
        content += `2. How collision detection is implemented\n`;
        content += `3. Communication between client and server\n`;
        content += `4. Potential areas for bugs or inconsistencies\n`;
        
        fs.writeFileSync(combo.output, content);
        console.log(`✅ Created super-combined file: ${combo.output}`);
    });
    
    // Create a single file with ALL code
    const allFilesOutput = 'extraction_output/systems/all_systems.txt';
    let allContent = `# COMPLETE GAME SYSTEMS CODE\n\n`;
    allContent += `Extraction Date: ${new Date().toISOString()}\n\n`;
    allContent += `This file contains ALL code related to the collision and map systems.\n\n`;
    
    // Add collision system header
    allContent += `\n# COLLISION SYSTEM\n\n`;
    
    // Add collision frontend
    allContent += `\n## COLLISION SYSTEM - FRONTEND\n\n`;
    const collisionFrontendFiles = successfulCopies.filter(file => file.includes('/collision_system/frontend/'));
    for (const file of collisionFrontendFiles) {
        if (fs.existsSync(file)) {
            const sourceFile = file.replace('extraction_output/systems', '').replace('.txt', '');
            allContent += `\n\n### FILE: ${sourceFile}\n\n`;
            allContent += '```javascript\n';
            allContent += fs.readFileSync(file, 'utf8');
            allContent += '\n```\n\n';
            allContent += '---\n';
        }
    }
    
    // Add collision backend
    allContent += `\n## COLLISION SYSTEM - BACKEND\n\n`;
    const collisionBackendFiles = successfulCopies.filter(file => file.includes('/collision_system/backend/'));
    for (const file of collisionBackendFiles) {
        if (fs.existsSync(file)) {
            const sourceFile = file.replace('extraction_output/systems', '').replace('.txt', '');
            allContent += `\n\n### FILE: ${sourceFile}\n\n`;
            allContent += '```javascript\n';
            allContent += fs.readFileSync(file, 'utf8');
            allContent += '\n```\n\n';
            allContent += '---\n';
        }
    }
    
    // Add map system header
    allContent += `\n# MAP SYSTEM\n\n`;
    
    // Add map frontend
    allContent += `\n## MAP SYSTEM - FRONTEND\n\n`;
    const mapFrontendFiles = successfulCopies.filter(file => file.includes('/map_system/frontend/'));
    for (const file of mapFrontendFiles) {
        if (fs.existsSync(file)) {
            const sourceFile = file.replace('extraction_output/systems', '').replace('.txt', '');
            allContent += `\n\n### FILE: ${sourceFile}\n\n`;
            allContent += '```javascript\n';
            allContent += fs.readFileSync(file, 'utf8');
            allContent += '\n```\n\n';
            allContent += '---\n';
        }
    }
    
    // Add map backend
    allContent += `\n## MAP SYSTEM - BACKEND\n\n`;
    const mapBackendFiles = successfulCopies.filter(file => file.includes('/map_system/backend/'));
    for (const file of mapBackendFiles) {
        if (fs.existsSync(file)) {
            const sourceFile = file.replace('extraction_output/systems', '').replace('.txt', '');
            allContent += `\n\n### FILE: ${sourceFile}\n\n`;
            allContent += '```javascript\n';
            allContent += fs.readFileSync(file, 'utf8');
            allContent += '\n```\n\n';
            allContent += '---\n';
        }
    }
    
    // Add analysis hints
    allContent += `\n# COMPLETE SYSTEM ANALYSIS\n\n`;
    allContent += `When analyzing these systems together, focus on:\n\n`;
    allContent += `1. How world coordinates are converted to tile coordinates across both systems\n`;
    allContent += `2. The relationship between collision detection and map boundaries\n`;
    allContent += `3. How client and server coordinate systems might differ\n`;
    allContent += `4. Places where inconsistent tile size usage might cause bugs\n`;
    allContent += `5. Potential fixes for coordinate mismatches reported in logs\n`;
    
    fs.writeFileSync(allFilesOutput, allContent);
    console.log(`✅ Created ALL systems file: ${allFilesOutput}`);
}

// Main execution
console.log('🔍 Starting system extraction...');

// Perform the extractions
const successfulCopies = [];
extractionTargets.forEach(target => {
    if (copyFile(target.source, target.target)) {
        successfulCopies.push(target.target);
    }
});

// Create summary files
createSummaryFile('collision_system', successfulCopies);
createSummaryFile('map_system', successfulCopies);

// Create index file
createIndexFile(successfulCopies);

// Create combined files
createCombinedFiles(successfulCopies);

console.log(`\n✨ Extraction complete! ${successfulCopies.length}/${extractionTargets.length} files processed.`);
console.log('📁 Check the extraction_output/systems directory for the extracted files.');
console.log('📄 For direct ChatGPT uploads, use the *_all.txt files in each system directory.'); 