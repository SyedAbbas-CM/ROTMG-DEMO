/**
 * ClientUnitRenderer.js - Handles rendering of military units on the client
 */

import { spriteManager } from '../assets/spriteManager.js';
import { TILE_SIZE, SCALE } from '../constants/constants.js';

export class ClientUnitRenderer {
    constructor() {
        this.renderScale = 1.0; // Base rendering scale
        this.teamColors = {
            blue: { primary: '#4A90E2', secondary: '#2E5B9A' },
            red: { primary: '#E24A4A', secondary: '#9A2E2E' },
            green: { primary: '#4AE24A', secondary: '#2E9A2E' },
            yellow: { primary: '#E2E24A', secondary: '#9A9A2E' },
            neutral: { primary: '#808080', secondary: '#404040' }
        };
    }
    
    /**
     * Render a single unit
     */
    renderUnit(ctx, unit, cameraPosition, viewType = 'topdown') {
        if (!unit || !ctx) return;
        
        // Calculate screen position
        const screenWidth = ctx.canvas.width;
        const screenHeight = ctx.canvas.height;
        const scaleFactor = viewType === 'strategic' ? 0.5 : 1.0;
        
        const screenX = (unit.x - cameraPosition.x) * TILE_SIZE * scaleFactor + screenWidth / 2;
        const screenY = (unit.y - cameraPosition.y) * TILE_SIZE * scaleFactor + screenHeight / 2;
        
        // Skip if off-screen
        const margin = 50;
        if (screenX < -margin || screenX > screenWidth + margin || 
            screenY < -margin || screenY > screenHeight + margin) {
            return;
        }
        
        // Get sprite info
        const sprite = unit.sprite;
        let renderSuccess = false;
        
        if (sprite && spriteManager) {
            try {
                const spriteSheet = spriteManager.getSpriteSheet(sprite.sheet);
                if (spriteSheet) {
                    const spriteData = spriteManager.getSpriteData(sprite.sheet, sprite.name);
                    if (spriteData) {
                        const size = (sprite.scale || 0.5) * SCALE * scaleFactor;
                        
                        // Save context for team coloring
                        ctx.save();
                        
                        // Apply team coloring filter
                        const teamColor = this.teamColors[unit.team] || this.teamColors.neutral;
                        if (unit.team && unit.team !== 'neutral') {
                            ctx.filter = `hue-rotate(${this.getTeamHueRotation(unit.team)}deg) saturate(1.2)`;
                        }
                        
                        // Draw the unit sprite
                        spriteManager.drawSprite(
                            ctx,
                            sprite.sheet,
                            spriteData.x || 0,
                            spriteData.y || 0, 
                            screenX - size/2,
                            screenY - size/2,
                            size,
                            size
                        );
                        
                        ctx.restore();
                        renderSuccess = true;
                    }
                }
            } catch (error) {
                console.warn(`[ClientUnitRenderer] Failed to render sprite for unit ${unit.id}:`, error);
            }
        }
        
        // Fallback rendering if sprite fails
        if (!renderSuccess) {
            this.renderFallbackUnit(ctx, unit, screenX, screenY, scaleFactor);
        }
        
        // Render unit info overlay
        this.renderUnitOverlay(ctx, unit, screenX, screenY, scaleFactor, viewType);
    }
    
    /**
     * Fallback rendering when sprites aren't available
     */
    renderFallbackUnit(ctx, unit, screenX, screenY, scaleFactor) {
        const size = 16 * scaleFactor;
        const teamColor = this.teamColors[unit.team] || this.teamColors.neutral;
        
        ctx.save();
        
        // Main unit body
        ctx.fillStyle = teamColor.primary;
        ctx.fillRect(screenX - size/2, screenY - size/2, size, size);
        
        // Unit type indicator
        ctx.fillStyle = teamColor.secondary;
        const indicatorSize = size * 0.3;
        
        switch (unit.category) {
            case 'infantry':
                // Draw a square
                ctx.fillRect(screenX - indicatorSize/2, screenY - indicatorSize/2, indicatorSize, indicatorSize);
                break;
            case 'cavalry':
                // Draw a triangle
                ctx.beginPath();
                ctx.moveTo(screenX, screenY - indicatorSize/2);
                ctx.lineTo(screenX - indicatorSize/2, screenY + indicatorSize/2);
                ctx.lineTo(screenX + indicatorSize/2, screenY + indicatorSize/2);
                ctx.closePath();
                ctx.fill();
                break;
            case 'ranged':
                // Draw a circle
                ctx.beginPath();
                ctx.arc(screenX, screenY, indicatorSize/2, 0, Math.PI * 2);
                ctx.fill();
                break;
            default:
                // Default diamond
                ctx.beginPath();
                ctx.moveTo(screenX, screenY - indicatorSize/2);
                ctx.lineTo(screenX + indicatorSize/2, screenY);
                ctx.lineTo(screenX, screenY + indicatorSize/2);
                ctx.lineTo(screenX - indicatorSize/2, screenY);
                ctx.closePath();
                ctx.fill();
        }
        
        ctx.restore();
    }
    
    /**
     * Render unit overlay (health, name, etc.)
     */
    renderUnitOverlay(ctx, unit, screenX, screenY, scaleFactor, viewType) {
        const size = 16 * scaleFactor;
        
        ctx.save();
        
        // Health bar
        if (unit.health !== undefined && unit.maxHealth !== undefined) {
            const healthPercent = Math.max(0, unit.health / unit.maxHealth);
            const barWidth = size * 1.2;
            const barHeight = 3 * scaleFactor;
            const barY = screenY + size/2 + 2;
            
            // Background
            ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
            ctx.fillRect(screenX - barWidth/2, barY, barWidth, barHeight);
            
            // Health
            let healthColor = '#4CAF50'; // Green
            if (healthPercent < 0.6) healthColor = '#FFC107'; // Yellow
            if (healthPercent < 0.3) healthColor = '#F44336'; // Red
            
            ctx.fillStyle = healthColor;
            ctx.fillRect(screenX - barWidth/2, barY, barWidth * healthPercent, barHeight);
        }
        
        // Unit name (only in detailed view)
        if (viewType !== 'strategic' && scaleFactor > 0.7) {
            const displayName = unit.displayName || unit.typeName || 'Unit';
            ctx.fillStyle = '#FFFFFF';
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 2;
            ctx.font = `${Math.max(8, 10 * scaleFactor)}px Arial`;
            ctx.textAlign = 'center';
            
            const textY = screenY - size/2 - 5;
            ctx.strokeText(displayName, screenX, textY);
            ctx.fillText(displayName, screenX, textY);
        }
        
        // Level indicator (for experienced units)
        if (unit.level && unit.level > 1) {
            const levelSize = 8 * scaleFactor;
            ctx.fillStyle = '#FFD700'; // Gold
            ctx.strokeStyle = '#B8860B'; // Dark gold
            ctx.lineWidth = 1;
            
            ctx.beginPath();
            ctx.arc(screenX + size/2 - 2, screenY - size/2 + 2, levelSize/2, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            
            ctx.fillStyle = '#000000';
            ctx.font = `${Math.max(6, 8 * scaleFactor)}px Arial`;
            ctx.textAlign = 'center';
            ctx.fillText(unit.level.toString(), screenX + size/2 - 2, screenY - size/2 + 6);
        }
        
        // Morale indicator (color coded)
        if (unit.morale !== undefined && viewType !== 'strategic') {
            const moraleColor = this.getMoraleColor(unit.morale);
            ctx.fillStyle = moraleColor;
            const moraleSize = 3 * scaleFactor;
            ctx.fillRect(screenX - size/2, screenY - size/2, moraleSize, moraleSize);
        }
        
        ctx.restore();
    }
    
    /**
     * Render multiple units efficiently
     */
    renderUnits(ctx, units, cameraPosition, viewType = 'topdown') {
        if (!units || !Array.isArray(units)) return;
        
        // Sort by Y position for proper layering
        const sortedUnits = [...units].sort((a, b) => a.y - b.y);
        
        sortedUnits.forEach(unit => {
            this.renderUnit(ctx, unit, cameraPosition, viewType);
        });
    }
    
    /**
     * Get team-specific hue rotation
     */
    getTeamHueRotation(team) {
        const rotations = {
            blue: 0,
            red: 180,
            green: 120,
            yellow: 60,
            purple: 270,
            orange: 30
        };
        return rotations[team] || 0;
    }
    
    /**
     * Get morale-based color
     */
    getMoraleColor(morale) {
        if (morale > 80) return '#4CAF50'; // High morale - green
        if (morale > 60) return '#8BC34A'; // Good morale - light green
        if (morale > 40) return '#FFC107'; // Medium morale - yellow
        if (morale > 20) return '#FF9800'; // Low morale - orange
        return '#F44336'; // Very low morale - red
    }
    
    /**
     * Debug rendering - show unit details
     */
    renderUnitDebugInfo(ctx, unit, screenX, screenY) {
        if (!unit) return;
        
        ctx.save();
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(screenX + 20, screenY - 40, 150, 80);
        
        ctx.fillStyle = '#FFFFFF';
        ctx.font = '10px monospace';
        ctx.textAlign = 'left';
        
        const lines = [
            `ID: ${unit.id}`,
            `Type: ${unit.displayName || unit.typeName}`,
            `Team: ${unit.team}`,
            `Health: ${Math.floor(unit.health)}/${unit.maxHealth}`,
            `Morale: ${Math.floor(unit.morale || 0)}`,
            `State: ${unit.state}`,
            `Level: ${unit.level || 1}`
        ];
        
        lines.forEach((line, index) => {
            ctx.fillText(line, screenX + 25, screenY - 30 + index * 12);
        });
        
        ctx.restore();
    }
}

// Create a singleton instance for use across the app
export const unitRenderer = new ClientUnitRenderer();