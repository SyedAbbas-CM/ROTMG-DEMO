// Managers/world/mapLoader.js

/**
 * MapLoader class to handle loading of fixed maps from JSON files.
 */
export class MapLoader {
    /**
     * Loads a map from a given URL.
     * @param {string} url - The URL of the map JSON file.
     * @returns {Promise<Object>} - A promise that resolves to the map data.
     */
    static async loadMapFromFile(url) {
      try {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Failed to load map from ${url}: ${response.statusText}`);
        }
        const mapData = await response.json();
        return mapData;
      } catch (error) {
        console.error('Error loading map:', error);
        throw error;
      }
    }
  }
  