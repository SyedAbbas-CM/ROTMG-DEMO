// extract_systems.js
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Get current directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Define system names and base paths
const systems = {
    collision_system: { frontend: 'public/src/collision', backend: 'src' },
    map_system:       { frontend: 'public/src/map', backend: 'src' },
    networking:       { frontend: 'public/src/network', backend: 'src' },
    rendering_ui:     { frontend: 'public/src' }, // More dispersed frontend system
    entity_management:{ frontend: 'public/src', backend: 'src' },
    game_logic:       { backend: 'src' },
    inventory_items:  { backend: 'src' },
    persistence:      { backend: 'src' },
    shared_utils:     { frontend: 'public/src', backend: 'src' },
    wasm_integration: { frontend: 'public/src/wasm', backend: 'src' },
};

// Define extraction targets
// Note: Some paths might need refinement based on actual file contents/usage
const extractionTargets = [
    // --- Collision System ---
    // Frontend
    { system: 'collision_system', type: 'frontend', source: 'public/src/collision/ClientCollisionManager.js' },
    { system: 'collision_system', type: 'frontend', source: 'public/src/collision/collisionSystem.js' },
    { system: 'collision_system', type: 'frontend', source: 'public/src/shared/spatialGrid.js' }, // Shared but relevant
    { system: 'collision_system', type: 'frontend', source: 'public/src/game/updateCharacter.js' }, // Assumed relevance
    // Backend
    { system: 'collision_system', type: 'backend', source: 'src/CollisionManager.js' },
    //{ system: 'collision_system', type: 'backend', source: 'src/networkHandlers/collisionHandler.js' }, // Verify path

    // --- Map System ---
    // Frontend
    { system: 'map_system', type: 'frontend', source: 'public/src/map/ClientMapManager.js' },
    { system: 'map_system', type: 'frontend', source: 'public/src/map/tile.js' },
    { system: 'map_system', type: 'frontend', source: 'public/src/camera.js' }, // Related
    { system: 'map_system', type: 'frontend', source: 'public/src/utils/coordinateUtils.js' }, // Likely related
    // Backend
    { system: 'map_system', type: 'backend', source: 'src/MapManager.js' },
    //{ system: 'map_system', type: 'backend', source: 'src/networkHandlers/chunkRequest.js' }, // Verify path
    { system: 'map_system', type: 'backend', source: 'src/world/AdvancedPerlinNoise.js' },

    // --- Networking ---
    // Frontend
    //{ system: 'networking', type: 'frontend', source: 'public/src/network/' }, // Need specific files or pattern
    { system: 'networking', type: 'frontend', source: 'public/src/networking.js' },
    // Backend
    { system: 'networking', type: 'backend', source: 'src/NetworkManager.js' },
    //{ system: 'networking', type: 'backend', source: 'src/networkHandlers/' }, // Need specific files or pattern
    //{ system: 'networking', type: 'backend', source: 'src/routes/' }, // Need specific files or pattern

    // --- Rendering/UI ---
    // Frontend
    //{ system: 'rendering_ui', type: 'frontend', source: 'public/src/render/' }, // Need specific files or pattern
    //{ system: 'rendering_ui', type: 'frontend', source: 'public/src/ui/' }, // Need specific files or pattern
    //{ system: 'rendering_ui', type: 'frontend', source: 'public/src/screens/' }, // Need specific files or pattern
    { system: 'rendering_ui', type: 'frontend', source: 'public/src/sprite/spritesheet.js' }, // Example, add more
    { system: 'rendering_ui', type: 'frontend', source: 'public/src/camera.js' }, // Also relevant here

    // --- Entity Management ---
    // Frontend
    //{ system: 'entity_management', type: 'frontend', source: 'public/src/entities/' }, // Need specific files or pattern
    { system: 'entity_management', type: 'frontend', source: 'public/src/units/ClientUnitManager.js' }, // Example, add more
    //{ system: 'entity_management', type: 'frontend', source: 'public/src/game/' }, // Need specific files or pattern
    // Backend
    //{ system: 'entity_management', type: 'backend', source: 'src/units/' }, // Need specific files or pattern
    { system: 'entity_management', type: 'backend', source: 'src/EnemyManager.js' },
    { system: 'entity_management', type: 'backend', source: 'src/BulletManager.js' },
    { system: 'entity_management', type: 'backend', source: 'src/MapObjectManager.js' },

    // --- Game Logic/Behavior ---
    // Backend
    { system: 'game_logic', type: 'backend', source: 'src/BehaviorSystem.js' },
    { system: 'game_logic', type: 'backend', source: 'src/BehaviorState.js' },
    { system: 'game_logic', type: 'backend', source: 'src/Behaviors.js' },
    //{ system: 'game_logic', type: 'backend', source: 'src/Behaviours/' }, // Need specific files or pattern
    { system: 'game_logic', type: 'backend', source: 'src/Transitions.js' },

    // --- Inventory/Items ---
    // Backend
    { system: 'inventory_items', type: 'backend', source: 'src/InventoryManager.js' },
    { system: 'inventory_items', type: 'backend', source: 'src/ItemManager.js' },

    // --- Persistence ---
    // Backend
    { system: 'persistence', type: 'backend', source: 'src/database.js' },

    // --- Shared Utilities ---
    // Frontend
    //{ system: 'shared_utils', type: 'frontend', source: 'public/src/utils/' }, // Need specific files or pattern
    { system: 'shared_utils', type: 'frontend', source: 'public/src/shared/spatialGrid.js' }, // Example
    //{ system: 'shared_utils', type: 'frontend', source: 'public/src/constants/' }, // Need specific files or pattern
    // Backend
    //{ system: 'shared_utils', type: 'backend', source: 'src/utils/' }, // Need specific files or pattern
    //{ system: 'shared_utils', type: 'backend', source: 'src/shared/' }, // Need specific files or pattern

    // --- WASM Integration ---
    // Frontend
    //{ system: 'wasm_integration', type: 'frontend', source: 'public/src/wasm/' }, // Need specific files or pattern
    // Backend
    //{ system: 'wasm_integration', type: 'backend', source: 'src/wasm/' }, // Need specific files or pattern
    { system: 'wasm_integration', type: 'backend', source: 'src/wasmLoader.js' },

].map(target => ({
    ...target,
    // Generate target path dynamically
    target: `extraction_output/systems/${target.system}/${target.type}/${path.basename(target.source)}.txt`
}));

// Function to copy a file
function copyFile(source, target) {
    try {
        const absoluteSource = path.resolve(__dirname, '..', source); // Go up one level from script location
        if (fs.existsSync(absoluteSource)) {
            const content = fs.readFileSync(absoluteSource, 'utf8');
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
            console.log(`❌ Source file not found: ${absoluteSource} (Searched for relative: ${source})`);
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
    
    let summaryContent = `# ${system.replace(/_/g, ' ').toUpperCase()} SUMMARY\n\n`;
    summaryContent += `Extraction Date: ${new Date().toISOString()}\n\n`;
    
    // Add list of extracted files
    summaryContent += `## Extracted Files:\n\n`;
    
    const systemFiles = successfulCopies.filter(file => file.includes(`/${system}/`));

    // Frontend files
    const frontendFiles = systemFiles.filter(file => file.includes(`/${system}/frontend/`));
    if (frontendFiles.length > 0) {
        summaryContent += `### Frontend:\n`;
        frontendFiles.forEach(file => {
            const fileName = path.basename(file);
            summaryContent += `- ${fileName}\n`;
        });
    }
    
    // Backend files
    const backendFiles = systemFiles.filter(file => file.includes(`/${system}/backend/`));
    if (backendFiles.length > 0) {
        summaryContent += `\n### Backend:\n`;
        backendFiles.forEach(file => {
            const fileName = path.basename(file);
            summaryContent += `- ${fileName}\n`;
        });
    }
    
    // Add instructions for ChatGPT
    summaryContent += `\n## Usage Instructions for LLMs (ChatGPT, Claude, etc.)\n\n`;
    summaryContent += `To analyze this system, upload the relevant \`*_all.txt\` files or individual \`.txt\` files. For best results:\n`;
    summaryContent += `1. Ask specific questions about how the ${system.replace(/_/g, ' ')} works.\n`;
    summaryContent += `2. Inquire about the relationship between frontend and backend components (if applicable).\n`;
    summaryContent += `3. Request explanations of specific functions or logic blocks.\n`;
    summaryContent += `4. Ask for potential areas of improvement or bug identification.\n`;
    
    fs.writeFileSync(summaryPath, summaryContent);
    console.log(`✅ Created summary file: ${summaryPath}`);
}

// Create a main index file
function createIndexFile(successfulCopies) {
    const indexPath = 'extraction_output/systems/system_index.txt';
    
    let indexContent = `# GAME SYSTEMS EXTRACTION INDEX\n\n`;
    indexContent += `Extraction Date: ${new Date().toISOString()}\n\n`;
    
    indexContent += `## Systems Extracted:\n\n`;
    
    const extractedSystems = [...new Set(successfulCopies.map(file => file.split('/')[2]))]; // Get unique system names

    extractedSystems.sort().forEach(system => {
        const systemNameFormatted = system.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()); // Format name nicely
        indexContent += `### ${systemNameFormatted}\n`;
        const frontendCount = successfulCopies.filter(file => file.includes(`/${system}/frontend/`)).length;
        const backendCount = successfulCopies.filter(file => file.includes(`/${system}/backend/`)).length;
        if (frontendCount > 0) indexContent += `- Frontend: ${frontendCount} files\n`;
        if (backendCount > 0) indexContent += `- Backend: ${backendCount} files\n`;
        indexContent += '\n';
    });

    // Add master compilation instructions
    indexContent += `## Creating Master Compilations\n\n`;
    indexContent += `To create a single file with all code for each system part (frontend/backend), run these commands from the root directory:\n\n`;

    extractedSystems.forEach(system => {
         const systemNameFormatted = system.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
         indexContent += `For ${systemNameFormatted}:\n`;
         indexContent += "```bash\n";
         const frontendFiles = successfulCopies.filter(file => file.includes(`/${system}/frontend/`));
         const backendFiles = successfulCopies.filter(file => file.includes(`/${system}/backend/`));

         if (frontendFiles.length > 0) {
             indexContent += `cat extraction_output/systems/${system}/frontend/*.txt > extraction_output/systems/${system}/frontend_all.txt\n`;
         }
         if (backendFiles.length > 0) {
             indexContent += `cat extraction_output/systems/${system}/backend/*.txt > extraction_output/systems/${system}/backend_all.txt\n`;
         }
         indexContent += "```\n\n";
    });
    
    fs.writeFileSync(indexPath, indexContent);
    console.log(`✅ Created index file: ${indexPath}`);
}

// Function to create combined files for direct upload to ChatGPT
function createCombinedFiles(successfulCopies) {
    const extractedSystems = [...new Set(successfulCopies.map(file => file.split('/')[2]))]; // Get unique system names

    const combinations = [];
    const superCombinations = [];

    extractedSystems.forEach(system => {
        const systemNameFormatted = system.replace(/_/g, ' ').toUpperCase();
        const hasFrontend = successfulCopies.some(file => file.includes(`/${system}/frontend/`));
        const hasBackend = successfulCopies.some(file => file.includes(`/${system}/backend/`));

        if (hasFrontend) {
            combinations.push({
                pattern: `/${system}/frontend/`,
                output: `extraction_output/systems/${system}/frontend_all.txt`,
                title: `${systemNameFormatted} - FRONTEND`
            });
        }
        if (hasBackend) {
             combinations.push({
                pattern: `/${system}/backend/`,
                output: `extraction_output/systems/${system}/backend_all.txt`,
                title: `${systemNameFormatted} - BACKEND`
            });
        }
        if (hasFrontend || hasBackend) {
            superCombinations.push({
                system: system,
                pattern: `/${system}/`,
                output: `extraction_output/systems/${system}_all.txt`,
                title: `COMPLETE ${systemNameFormatted} (${[hasFrontend && 'FRONTEND', hasBackend && 'BACKEND'].filter(Boolean).join(' + ')})`
            });
        }
    });
    
    combinations.forEach(combo => {
        const files = successfulCopies.filter(file => file.includes(combo.pattern));
        if (files.length === 0) return; // Skip if no files match

        let content = `# ${combo.title}\n\n`;
        content += `Extraction Date: ${new Date().toISOString()}\n\n`;
        
        for (const file of files) {
            if (fs.existsSync(file)) {
                const fileName = path.basename(file, '.txt'); // Get original JS filename
                const sourceFile = extractionTargets.find(t => t.target === file)?.source || fileName; // Find original source path
                content += `\n\n## FILE: ${sourceFile}\n\n`;
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
    superCombinations.forEach(combo => {
        const files = successfulCopies.filter(file => file.includes(combo.pattern));
         if (files.length === 0) return; // Skip if no files match

        let content = `# ${combo.title}\n\n`;
        content += `Extraction Date: ${new Date().toISOString()}\n\n`;
        content += `This file contains all extracted code related to the ${combo.system.replace(/_/g, ' ')} system.\n\n`;

        // First add frontend files
        const frontendFiles = files.filter(file => file.includes('/frontend/'));
        if (frontendFiles.length > 0) {
            content += `\n# FRONTEND CODE\n\n`;
            for (const file of frontendFiles) {
                if (fs.existsSync(file)) {
                     const fileName = path.basename(file, '.txt');
                     const sourceFile = extractionTargets.find(t => t.target === file)?.source || fileName;
                     content += `\n\n## FILE: ${sourceFile}\n\n`;
                    content += '```javascript\n';
                    content += fs.readFileSync(file, 'utf8');
                    content += '\n```\n\n';
                    content += '---\n';
                }
            }
        }

        // Then add backend files
        const backendFiles = files.filter(file => file.includes('/backend/'));
        if (backendFiles.length > 0) {
            content += `\n# BACKEND CODE\n\n`;
            for (const file of backendFiles) {
                if (fs.existsSync(file)) {
                    const fileName = path.basename(file, '.txt');
                    const sourceFile = extractionTargets.find(t => t.target === file)?.source || fileName;
                    content += `\n\n## FILE: ${sourceFile}\n\n`;
                    content += '```javascript\n';
                    content += fs.readFileSync(file, 'utf8');
                    content += '\n```\n\n';
                    content += '---\n';
                }
            }
        }
        
        // Add analysis suggestions
        content += `\n# ANALYSIS SUGGESTIONS\n\n`;
        content += `When analyzing this code, focus on:\n\n`;
        content += `1. The overall purpose and responsibilities of the ${combo.system.replace(/_/g, ' ')} system.\n`;
        content += `2. Key algorithms or data structures used.\n`;
        content += `3. How it interacts with other systems (if discernible).\n`;
        if (combo.title.includes('FRONTEND') && combo.title.includes('BACKEND')) {
            content += `4. The communication flow between the client and server parts.\n`;
        }
        content += `5. Potential areas for bugs, improvements, or refactoring.\n`;

        fs.writeFileSync(combo.output, content);
        console.log(`✅ Created super-combined file: ${combo.output}`);
    });

    // Create a single file with ALL code
    const allFilesOutput = 'extraction_output/systems/all_systems.txt';
    let allContent = `# COMPLETE JAVASCRIPT GAME SYSTEMS CODE\n\n`; // Updated title
    allContent += `Extraction Date: ${new Date().toISOString()}\n\n`;
    allContent += `This file contains ALL extracted JavaScript code related to the identified game systems.\n\n`;

    // Loop through each system and add its sections
    extractedSystems.sort().forEach(system => {
        const systemNameFormatted = system.replace(/_/g, ' ').toUpperCase();
        const systemFiles = successfulCopies.filter(file => file.includes(`/${system}/`));
        const frontendFiles = systemFiles.filter(file => file.includes('/frontend/'));
        const backendFiles = systemFiles.filter(file => file.includes('/backend/'));

         if (systemFiles.length === 0) return; // Skip empty systems

        allContent += `\n# ${systemNameFormatted} SYSTEM\n\n`;

        // Add frontend
        if (frontendFiles.length > 0) {
            allContent += `\n## ${systemNameFormatted} - FRONTEND\n\n`;
            for (const file of frontendFiles) {
                if (fs.existsSync(file)) {
                    const fileName = path.basename(file, '.txt');
                    const sourceFile = extractionTargets.find(t => t.target === file)?.source || fileName;
                    allContent += `\n\n### FILE: ${sourceFile}\n\n`;
                    allContent += '```javascript\n';
                    allContent += fs.readFileSync(file, 'utf8');
                    allContent += '\n```\n\n';
                    allContent += '---\n';
                }
            }
        }

        // Add backend
        if (backendFiles.length > 0) {
            allContent += `\n## ${systemNameFormatted} - BACKEND\n\n`;
            for (const file of backendFiles) {
                if (fs.existsSync(file)) {
                    const fileName = path.basename(file, '.txt');
                    const sourceFile = extractionTargets.find(t => t.target === file)?.source || fileName;
                    allContent += `\n\n### FILE: ${sourceFile}\n\n`;
                    allContent += '```javascript\n';
                    allContent += fs.readFileSync(file, 'utf8');
                    allContent += '\n```\n\n';
                    allContent += '---\n';
                }
            }
        }
    });

    // Add analysis hints
    allContent += `\n# COMPLETE SYSTEM ANALYSIS\n\n`;
    allContent += `When analyzing these systems together, focus on:\n\n`;
    allContent += `1. Interactions and dependencies between different systems.\n`;
    allContent += `2. Consistent coding patterns and potential inconsistencies.\n`;
    allContent += `3. Overall architecture and data flow.\n`;
    allContent += `4. Identifying shared components and potential duplication.\n`;
    allContent += `5. Potential performance bottlenecks or areas for optimization.\n`;

    fs.writeFileSync(allFilesOutput, allContent);
    console.log(`✅ Created ALL systems file: ${allFilesOutput}`);
}

// Main execution
console.log('🚀 Preparing system extraction script...');

// Get all unique system names from targets
const allSystemNames = [...new Set(extractionTargets.map(t => t.system))];

// Perform the extractions (Wrapped in a function to prevent execution for now)
function runExtraction() {
    console.log('🔍 Starting system extraction...');
    const successfulCopies = [];
    extractionTargets.forEach(target => {
        // Resolve source path relative to the project root (assuming script is in a subfolder)
        const absoluteSourcePath = path.resolve(__dirname, '..', target.source);
         // Check if source is a directory - skip for now, requires directory walking logic
         if (fs.existsSync(absoluteSourcePath) && fs.lstatSync(absoluteSourcePath).isDirectory()) {
            console.warn(`⚠️ Skipping directory source: ${target.source}. Add specific files or implement directory copying.`);
            return; // Skip directories for now
        }
        if (copyFile(target.source, target.target)) {
            successfulCopies.push(target.target);
        }
    });

    // Create summary files for each system
    allSystemNames.forEach(system => {
        createSummaryFile(system, successfulCopies);
    });

    // Create index file
    createIndexFile(successfulCopies);

    // Create combined files
    createCombinedFiles(successfulCopies);

    console.log(`\n✨ Extraction complete! ${successfulCopies.length}/${extractionTargets.filter(t => {
        const absPath = path.resolve(__dirname, '..', t.source);
        return !fs.existsSync(absPath) || !fs.lstatSync(absPath).isDirectory(); // Count only files attempted
    }).length} files processed.`);
    console.log('📁 Check the extraction_output/systems directory for the extracted files.');
    console.log('📄 For direct LLM uploads, use the *_all.txt files in each system directory or the main all_systems.txt.');
}

// --- SCRIPT EXECUTION CONTROL ---
// To run the extraction, uncomment the following line:
// runExtraction();

// To just log the targets and planned structure (without copying):
console.log('\n--- Planned Extraction Targets ---');
extractionTargets.forEach(t => console.log(`System: ${t.system}, Type: ${t.type}, Source: ${t.source} -> Target: ${t.target}`));
console.log('\n--- Systems Identified ---');
allSystemNames.forEach(s => console.log(s));
console.log('\nScript updated. Run manually by uncommenting `runExtraction();` at the end of the file.');
// ------------------------------- 