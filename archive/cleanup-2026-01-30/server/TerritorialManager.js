/**
 * TerritorialManager - Risk-like territorial control system
 * Manages ownership, conflicts, and strategic resources across the galaxy
 */

export class TerritorialManager {
    constructor(galaxyManager, options = {}) {
        this.galaxyManager = galaxyManager;
        
        // Territorial data
        this.territories = new Map();        // territoryId -> TerritoryData
        this.factions = new Map();           // factionId -> FactionData  
        this.borders = new Map();            // territoryId -> neighboring territories
        this.conflicts = new Map();          // conflictId -> ConflictData
        this.supply_lines = new Map();       // Routes between territories
        
        // Game balance parameters
        this.TERRITORY_TYPES = {
            PLAINS: { defenseDifficulty: 1.0, resourceMultiplier: 1.0 },
            FORTRESS: { defenseDifficulty: 2.5, resourceMultiplier: 0.8 },
            RESOURCE: { defenseDifficulty: 0.7, resourceMultiplier: 2.0 },
            CROSSING: { defenseDifficulty: 1.5, resourceMultiplier: 1.2 },
            WASTELAND: { defenseDifficulty: 0.5, resourceMultiplier: 0.3 },
            CAPITAL: { defenseDifficulty: 3.0, resourceMultiplier: 1.8 },
            PORTAL: { defenseDifficulty: 1.2, resourceMultiplier: 1.5 }
        };

        // Conflict resolution settings
        this.conflictResolution = {
            minDuration: 60000,          // 1 minute minimum battle
            maxDuration: 1800000,        // 30 minutes maximum 
            moraleThreshold: 0.2,        // Route at 20% morale
            reinforcementRate: 0.1       // 10% of reserves per minute
        };

        console.log('[TerritorialManager] Initialized territorial control system');
    }

    /**
     * Initialize a faction (player or AI)
     */
    createFaction(factionData) {
        const faction = {
            id: factionData.id,
            name: factionData.name,
            color: factionData.color,
            type: factionData.type, // 'player', 'ai', 'neutral'
            
            // Strategic resources
            resources: {
                gold: factionData.startingGold || 1000,
                supply: factionData.startingSupply || 500, 
                morale: factionData.startingMorale || 100,
                recruitment: factionData.startingRecruitment || 50
            },
            
            // Military capabilities
            military: {
                totalUnits: 0,
                reserves: 0,
                veterancy: factionData.veterancy || 1.0,
                doctrine: factionData.doctrine || 'balanced' // 'aggressive', 'defensive', 'mobile'
            },
            
            // Territory holdings
            territories: new Set(),
            homeTerritory: null,
            
            // Diplomatic status  
            relations: new Map(), // factionId -> relation value (-100 to +100)
            
            // AI behavior (for non-player factions)
            ai: {
                aggressiveness: factionData.aggressiveness || 0.5,
                expansionism: factionData.expansionism || 0.5,
                diplomatic: factionData.diplomatic || 0.5
            }
        };

        this.factions.set(faction.id, faction);
        console.log(`[Territorial] Created faction: ${faction.name}`);
        return faction;
    }

    /**
     * Claim a territory for a faction
     */
    claimTerritory(territoryId, factionId, method = 'conquest') {
        const territory = this.territories.get(territoryId);
        const faction = this.factions.get(factionId);
        
        if (!territory || !faction) {
            console.warn(`[Territorial] Invalid claim: ${territoryId} by ${factionId}`);
            return false;
        }

        // Remove from previous owner
        if (territory.owner) {
            const previousOwner = this.factions.get(territory.owner);
            if (previousOwner) {
                previousOwner.territories.delete(territoryId);
            }
        }

        // Assign to new owner
        territory.owner = factionId;
        territory.claimedAt = Date.now();
        territory.claimMethod = method;
        
        faction.territories.add(territoryId);
        
        // Set as home territory if first claim
        if (!faction.homeTerritory) {
            faction.homeTerritory = territoryId;
            console.log(`[Territorial] ${faction.name} established capital at ${territoryId}`);
        }

        // Update resource income
        this.recalculateFactionResources(factionId);
        
        console.log(`[Territorial] ${faction.name} claimed ${territoryId} via ${method}`);
        return true;
    }

    /**
     * Start a territorial conflict between factions
     */
    initiateConflict(attackerFactionId, defenderFactionId, territoryId) {
        const attacker = this.factions.get(attackerFactionId);
        const defender = this.factions.get(defenderFactionId);
        const territory = this.territories.get(territoryId);

        if (!attacker || !defender || !territory) {
            console.warn('[Territorial] Invalid conflict initiation');
            return null;
        }

        const conflictId = `${attackerFactionId}_vs_${defenderFactionId}_${territoryId}_${Date.now()}`;
        
        const conflict = {
            id: conflictId,
            attacker: attackerFactionId,
            defender: defenderFactionId,
            territory: territoryId,
            
            // Battle state
            status: 'active', // 'active', 'resolved', 'stalemate'
            startedAt: Date.now(),
            duration: 0,
            
            // Military forces
            attackerForces: this.calculateAvailableForces(attackerFactionId, territoryId),
            defenderForces: this.calculateAvailableForces(defenderFactionId, territoryId),
            
            // Battle dynamics
            attackerMorale: 100,
            defenderMorale: 100 + (territory.defenseBonus || 0),
            
            // Reinforcements
            attackerReinforcements: 0,
            defenderReinforcements: 0,
            
            // Resolution data
            resolvedAt: null,
            winner: null,
            casualties: { attacker: 0, defender: 0 }
        };

        this.conflicts.set(conflictId, conflict);
        
        console.log(`[Territorial] Conflict initiated: ${attacker.name} attacks ${defender.name} for ${territoryId}`);
        
        // Notify interested systems (battle system, UI, etc.)
        this.onConflictStarted(conflict);
        
        return conflict;
    }

    /**
     * Calculate available military forces for a conflict
     */
    calculateAvailableForces(factionId, territoryId) {
        const faction = this.factions.get(factionId);
        if (!faction) return 0;

        // Base forces from reserves
        let availableForces = Math.floor(faction.military.reserves * 0.5);
        
        // Distance penalty (simplified - could use pathfinding)
        const homeTerritory = this.territories.get(faction.homeTerritory);
        const targetTerritory = this.territories.get(territoryId);
        
        if (homeTerritory && targetTerritory) {
            const distance = this.calculateTerritoryDistance(homeTerritory, targetTerritory);
            const distancePenalty = Math.max(0.3, 1.0 - (distance * 0.1));
            availableForces = Math.floor(availableForces * distancePenalty);
        }

        // Doctrine modifiers
        switch (faction.military.doctrine) {
            case 'aggressive':
                availableForces = Math.floor(availableForces * 1.3);
                break;
            case 'defensive':
                if (targetTerritory.owner === factionId) {
                    availableForces = Math.floor(availableForces * 1.5);
                }
                break;
            case 'mobile':
                // Reduced distance penalty
                availableForces = Math.floor(availableForces * 1.1);
                break;
        }

        return Math.max(1, availableForces);
    }

    /**
     * Update ongoing conflicts (called each game tick)
     */
    updateConflicts(deltaTime) {
        for (const [conflictId, conflict] of this.conflicts.entries()) {
            if (conflict.status !== 'active') continue;

            conflict.duration += deltaTime;
            
            // Simulate battle progression
            this.simulateConflictTick(conflict, deltaTime);
            
            // Check for resolution conditions
            if (this.shouldResolveConflict(conflict)) {
                this.resolveConflict(conflict);
            }
        }
    }

    /**
     * Simulate a single tick of conflict
     */
    simulateConflictTick(conflict, deltaTime) {
        const territory = this.territories.get(conflict.territory);
        const territoryType = this.TERRITORY_TYPES[territory.type.toUpperCase()] || this.TERRITORY_TYPES.PLAINS;
        
        // Morale degradation over time
        const baseMoraleDecay = 0.01 * (deltaTime / 1000); // 1% per second baseline
        
        // Defender advantage from territory
        const defenseMultiplier = territoryType.defenseDifficulty;
        
        conflict.attackerMorale -= baseMoraleDecay * 1.2; // Attackers lose morale faster
        conflict.defenderMorale -= baseMoraleDecay * (1.0 / defenseMultiplier);
        
        // Force casualties (simplified)
        if (Math.random() < 0.1) { // 10% chance per tick
            const attackerCasualties = Math.floor(Math.random() * 5);
            const defenderCasualties = Math.floor(Math.random() * 5);
            
            conflict.casualties.attacker += attackerCasualties;
            conflict.casualties.defender += defenderCasualties;
            
            // Morale impact from casualties
            conflict.attackerMorale -= attackerCasualties * 2;
            conflict.defenderMorale -= defenderCasualties * 2;
        }
        
        // Clamp morale values
        conflict.attackerMorale = Math.max(0, conflict.attackerMorale);
        conflict.defenderMorale = Math.max(0, conflict.defenderMorale);
    }

    /**
     * Check if conflict should be resolved
     */
    shouldResolveConflict(conflict) {
        // Morale collapse
        if (conflict.attackerMorale <= this.conflictResolution.moraleThreshold * 100) return true;
        if (conflict.defenderMorale <= this.conflictResolution.moraleThreshold * 100) return true;
        
        // Maximum duration reached
        if (conflict.duration >= this.conflictResolution.maxDuration) return true;
        
        // Minimum duration not yet reached
        if (conflict.duration < this.conflictResolution.minDuration) return false;
        
        return false;
    }

    /**
     * Resolve a conflict and determine winner
     */
    resolveConflict(conflict) {
        conflict.status = 'resolved';
        conflict.resolvedAt = Date.now();
        
        // Determine winner based on remaining morale
        if (conflict.attackerMorale > conflict.defenderMorale) {
            conflict.winner = conflict.attacker;
            // Attacker takes territory
            this.claimTerritory(conflict.territory, conflict.attacker, 'conquest');
        } else if (conflict.defenderMorale > conflict.attackerMorale) {
            conflict.winner = conflict.defender;
            // Defender retains territory
        } else {
            // Stalemate
            conflict.status = 'stalemate';
            conflict.winner = null;
        }

        // Apply casualties to factions
        const attacker = this.factions.get(conflict.attacker);
        const defender = this.factions.get(conflict.defender);
        
        if (attacker) {
            attacker.military.totalUnits -= conflict.casualties.attacker;
            attacker.military.reserves = Math.max(0, attacker.military.reserves - conflict.casualties.attacker);
        }
        
        if (defender) {
            defender.military.totalUnits -= conflict.casualties.defender;
            defender.military.reserves = Math.max(0, defender.military.reserves - conflict.casualties.defender);
        }

        console.log(`[Territorial] Conflict resolved: ${conflict.winner ? this.factions.get(conflict.winner).name + ' wins' : 'stalemate'} at ${conflict.territory}`);
        
        // Notify systems
        this.onConflictResolved(conflict);
    }

    /**
     * Calculate resource income for a faction
     */
    recalculateFactionResources(factionId) {
        const faction = this.factions.get(factionId);
        if (!faction) return;

        let totalIncome = { gold: 0, supply: 0, morale: 0, recruitment: 0 };
        
        for (const territoryId of faction.territories) {
            const territory = this.territories.get(territoryId);
            if (!territory) continue;

            const territoryType = this.TERRITORY_TYPES[territory.type.toUpperCase()] || this.TERRITORY_TYPES.PLAINS;
            const multiplier = territoryType.resourceMultiplier;
            
            totalIncome.gold += (territory.resources?.gold || 0) * multiplier;
            totalIncome.supply += (territory.resources?.supply || 0) * multiplier;
            totalIncome.morale += (territory.resources?.morale || 0) * multiplier;
            totalIncome.recruitment += (territory.resources?.recruitment || 0) * multiplier;
        }

        faction.resourceIncome = totalIncome;
    }

    /**
     * Get territorial overview for strategic display
     */
    getTerritorialOverview() {
        const overview = {
            territories: Array.from(this.territories.values()),
            factions: Array.from(this.factions.values()),
            activeConflicts: Array.from(this.conflicts.values()).filter(c => c.status === 'active'),
            totalTerritories: this.territories.size
        };

        return overview;
    }

    /**
     * Calculate distance between territories (simplified)
     */
    calculateTerritoryDistance(territory1, territory2) {
        const dx = territory1.x - territory2.x;
        const dy = territory1.y - territory2.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Event handlers for other systems to hook into
     */
    onConflictStarted(conflict) {
        // Override in integrations
    }

    onConflictResolved(conflict) {
        // Override in integrations
    }
}