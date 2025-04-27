#!/bin/bash

# extract_game_systems.sh
# A script to extract game system files for analysis

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}== Game System Extraction Tool ==${NC}"
echo "This script will compile your game systems for sharing with ChatGPT"
echo ""

# Make sure the extraction output directory exists
mkdir -p extraction_output

# Run the extraction script
echo -e "${BLUE}Running extraction script...${NC}"
node extract_systems.js

# Check if extraction was successful
if [ $? -ne 0 ]; then
  echo -e "${RED}Extraction failed. Check errors above.${NC}"
  exit 1
fi

# Copy the extraction files to a location that's easier to access
echo -e "${BLUE}Creating easy access copies...${NC}"

# Create a simple access directory
mkdir -p easy_access

# Copy the compiled files - UPDATED WITH NEW CONCATENATED FILES
cp extraction_output/systems/collision_system_all.txt easy_access/collision_system_all.txt
cp extraction_output/systems/map_system_all.txt easy_access/map_system_all.txt
cp extraction_output/systems/all_systems.txt easy_access/all_systems.txt

# Keep the old files for backwards compatibility
cp extraction_output/systems/collision_system/frontend_all.txt easy_access/collision_system_frontend.txt
cp extraction_output/systems/collision_system/backend_all.txt easy_access/collision_system_backend.txt
cp extraction_output/systems/map_system/frontend_all.txt easy_access/map_system_frontend.txt
cp extraction_output/systems/map_system/backend_all.txt easy_access/map_system_backend.txt
cp extraction_output/systems/system_index.txt easy_access/system_index.txt

echo -e "${GREEN}Extraction complete!${NC}"
echo "You can find the extracted files in these locations:"
echo ""
echo "Detailed extraction:"
echo "  extraction_output/systems/..."
echo ""
echo -e "${GREEN}NEW CONCATENATED FILES:${NC}"
echo "  easy_access/collision_system_all.txt  - All collision system code (frontend + backend)"
echo "  easy_access/map_system_all.txt        - All map system code (frontend + backend)"
echo "  easy_access/all_systems.txt           - Everything in one file!"
echo ""
echo "Individual component files:"
echo "  easy_access/collision_system_frontend.txt"
echo "  easy_access/collision_system_backend.txt"
echo "  easy_access/map_system_frontend.txt"
echo "  easy_access/map_system_backend.txt"
echo ""
echo -e "${BLUE}Instructions:${NC}"
echo "1. Upload the files to ChatGPT for analysis:" 
echo "   - Use all_systems.txt for a complete analysis"
echo "   - Use collision_system_all.txt or map_system_all.txt for system-specific analysis"
echo "2. Ask specific questions about the coordinate system and collision detection"
echo "3. Focus on understanding the relationship between world and tile coordinates"
echo ""
echo -e "${GREEN}Happy debugging!${NC}" 