/**
 * CommandSystem.js - In-game command system for unit management
 */

import { BinaryPacket, MessageType } from '../common/protocol.js';

export class CommandSystem {
    constructor(serverInstance) {
        this.server = serverInstance; // Reference to main server
        this.commands = new Map();
        
        // Register available commands
        this.registerCommands();
        
        console.log(`[CommandSystem] Initialized with ${this.commands.size} commands`);
    }
    
    registerCommands() {
        // Unit spawning commands
        this.commands.set('spawn', {
            description: 'Spawn a unit: /spawn <type> [team] [count]',
            handler: this.handleSpawnCommand.bind(this),
            adminOnly: false
        });
        
        this.commands.set('spawnunit', {
            description: 'Spawn specific unit: /spawnunit <unitType>',
            handler: this.handleSpawnUnitCommand.bind(this),
            adminOnly: false
        });
        
        // Unit control commands
        this.commands.set('move', {
            description: 'Move units: /move <team> <x> <y>',
            handler: this.handleMoveCommand.bind(this),
            adminOnly: false
        });
        
        this.commands.set('attack', {
            description: 'Order attack: /attack <team> <x> <y>',
            handler: this.handleAttackCommand.bind(this),
            adminOnly: false
        });
        
        this.commands.set('formation', {
            description: 'Set formation: /formation <team> <type>',
            handler: this.handleFormationCommand.bind(this),
            adminOnly: false
        });
        
        // Utility commands
        this.commands.set('clear', {
            description: 'Clear all units: /clear [team]',
            handler: this.handleClearCommand.bind(this),
            adminOnly: false
        });
        
        this.commands.set('status', {
            description: 'Show unit status: /status',
            handler: this.handleStatusCommand.bind(this),
            adminOnly: false
        });
        
        this.commands.set('help', {
            description: 'Show available commands: /help',
            handler: this.handleHelpCommand.bind(this),
            adminOnly: false
        });
        
        // Unit types help
        this.commands.set('units', {
            description: 'List available unit types: /units',
            handler: this.handleUnitsCommand.bind(this),
            adminOnly: false
        });
    }
    
    /**
     * Process a chat message and check if it's a command
     */
    processMessage(clientId, message, playerData) {
        if (!message || typeof message !== 'string') return false;
        
        const trimmed = message.trim();
        if (!trimmed.startsWith('/')) {
            // Not a command, handle as regular chat
            this.broadcastChat(playerData.name || `Player ${clientId}`, trimmed);
            return true;
        }
        
        // Parse command
        const parts = trimmed.slice(1).split(/\s+/);
        const commandName = parts[0].toLowerCase();
        const args = parts.slice(1);
        
        const command = this.commands.get(commandName);
        if (!command) {
            this.sendMessageToClient(clientId, `Unknown command: /${commandName}. Type /help for available commands.`);
            return true;
        }
        
        // Execute command
        try {
            command.handler(clientId, args, playerData);
        } catch (error) {
            console.error(`[CommandSystem] Error executing command /${commandName}:`, error);
            this.sendMessageToClient(clientId, `Error executing command: ${error.message}`);
        }
        
        return true;
    }
    
    /**
     * Send a message to a specific client
     */
    sendMessageToClient(clientId, message) {
        const client = this.server.clients?.get(clientId);
        if (client && client.socket && client.socket.readyState === 1) { // 1 = OPEN
            try {
                const packet = BinaryPacket.encode(MessageType.CHAT_MESSAGE, {
                    sender: 'Server',
                    message: message,
                    color: '#FF6B6B'
                });
                client.socket.send(packet);
            } catch (err) {
                console.error(`[CommandSystem] Failed to send message to client ${clientId}:`, err);
            }
        }
    }
    
    /**
     * Broadcast a chat message to all clients
     */
    broadcastChat(sender, message) {
        const packet = BinaryPacket.encode(MessageType.CHAT_MESSAGE, {
            sender: sender,
            message: message,
            color: '#FFFFFF'
        });

        if (this.server.clients) {
            this.server.clients.forEach(client => {
                if (client.socket && client.socket.readyState === 1) {
                    try {
                        client.socket.send(packet);
                    } catch (err) {
                        console.error('[CommandSystem] Failed to broadcast message:', err);
                    }
                }
            });
        }
    }
    
    // Command handlers
    handleSpawnCommand(clientId, args, playerData) {
        if (args.length < 1) {
            this.sendMessageToClient(clientId, 'Usage: /spawn <type> [team] [count]');
            this.sendMessageToClient(clientId, 'Types: 0=Light Infantry, 1=Heavy Infantry, 2=Light Cavalry, 3=Heavy Cavalry, 4=Archer, 5=Crossbowman');
            return;
        }
        
        const type = parseInt(args[0]);
        const team = args[1] || 'player';
        const count = Math.min(parseInt(args[2]) || 1, 10); // Max 10 units per command
        
        if (isNaN(type) || type < 0 || type > 5) {
            this.sendMessageToClient(clientId, 'Invalid unit type. Use 0-5.');
            return;
        }
        
        // Get player position for spawning
        const client = this.server.clients?.get(clientId);
        if (!client || !client.player) {
            this.sendMessageToClient(clientId, 'Error: Could not determine your position.');
            return;
        }
        
        const playerX = client.player.x;
        const playerY = client.player.y;
        const mapId = client.mapId || this.server.gameState?.mapId || 'map_1';
        
        // Get world context
        const worldCtx = this.server.getWorldCtx ? this.server.getWorldCtx(mapId) : null;
        if (!worldCtx || !worldCtx.soldierMgr) {
            this.sendMessageToClient(clientId, 'Error: Could not access unit manager.');
            return;
        }
        
        // Spawn units in a small formation around player
        let spawned = 0;
        for (let i = 0; i < count; i++) {
            const offsetX = (Math.random() - 0.5) * 6; // Spread units around
            const offsetY = (Math.random() - 0.5) * 6;
            const spawnX = playerX + offsetX;
            const spawnY = playerY + offsetY;
            
            const unitId = worldCtx.soldierMgr.spawn(type, spawnX, spawnY, { team, owner: clientId });
            if (unitId) {
                spawned++;
            }
        }
        
        const unitNames = ['Light Infantry', 'Heavy Infantry', 'Light Cavalry', 'Heavy Cavalry', 'Archer', 'Crossbowman'];
        this.sendMessageToClient(clientId, `Spawned ${spawned} ${unitNames[type]}(s) for team ${team}`);
    }
    
    handleSpawnUnitCommand(clientId, args, playerData) {
        // Alias for spawn command with better names
        const unitMap = {
            'infantry': 0, 'light': 0, 'lightinfantry': 0,
            'heavy': 1, 'heavyinfantry': 1, 'tank': 1,
            'cavalry': 2, 'lightcav': 2, 'lightcavalry': 2,
            'heavycav': 3, 'heavycavalry': 3, 'knight': 3,
            'archer': 4, 'bow': 4, 'ranged': 4,
            'crossbow': 5, 'crossbowman': 5, 'xbow': 5
        };
        
        if (args.length < 1) {
            this.sendMessageToClient(clientId, 'Usage: /spawnunit <unitType>');
            this.sendMessageToClient(clientId, 'Unit types: ' + Object.keys(unitMap).join(', '));
            return;
        }
        
        const unitName = args[0].toLowerCase();
        const type = unitMap[unitName];
        
        if (type === undefined) {
            this.sendMessageToClient(clientId, 'Unknown unit type: ' + unitName);
            this.sendMessageToClient(clientId, 'Available types: ' + Object.keys(unitMap).join(', '));
            return;
        }
        
        // Call the regular spawn command
        this.handleSpawnCommand(clientId, [type.toString(), 'player', '1'], playerData);
    }
    
    handleMoveCommand(clientId, args, playerData) {
        if (args.length < 3) {
            this.sendMessageToClient(clientId, 'Usage: /move <team> <x> <y>');
            return;
        }
        
        const team = args[0];
        const targetX = parseFloat(args[1]);
        const targetY = parseFloat(args[2]);
        
        if (isNaN(targetX) || isNaN(targetY)) {
            this.sendMessageToClient(clientId, 'Invalid coordinates.');
            return;
        }
        
        const mapId = this.server.clients?.get(clientId)?.mapId || this.server.gameState?.mapId || 'map_1';
        const worldCtx = this.server.getWorldCtx ? this.server.getWorldCtx(mapId) : null;
        
        if (!worldCtx || !worldCtx.soldierMgr || !worldCtx.unitSystems) {
            this.sendMessageToClient(clientId, 'Error: Could not access unit systems.');
            return;
        }
        
        // Issue move commands to all units of the specified team
        let commandedCount = 0;
        for (let i = 0; i < worldCtx.soldierMgr.count; i++) {
            if (worldCtx.soldierMgr.owner && worldCtx.soldierMgr.owner[i] === team) {
                worldCtx.soldierMgr.cmdKind[i] = 1; // Move command
                worldCtx.soldierMgr.cmdTX[i] = targetX;
                worldCtx.soldierMgr.cmdTY[i] = targetY;
                commandedCount++;
            }
        }
        
        this.sendMessageToClient(clientId, `Ordered ${commandedCount} units from team ${team} to move to (${targetX}, ${targetY})`);
    }
    
    handleAttackCommand(clientId, args, playerData) {
        if (args.length < 3) {
            this.sendMessageToClient(clientId, 'Usage: /attack <team> <x> <y>');
            return;
        }
        
        const team = args[0];
        const targetX = parseFloat(args[1]);
        const targetY = parseFloat(args[2]);
        
        if (isNaN(targetX) || isNaN(targetY)) {
            this.sendMessageToClient(clientId, 'Invalid coordinates.');
            return;
        }
        
        const mapId = this.server.clients?.get(clientId)?.mapId || this.server.gameState?.mapId || 'map_1';
        const worldCtx = this.server.getWorldCtx ? this.server.getWorldCtx(mapId) : null;
        
        if (!worldCtx || !worldCtx.soldierMgr) {
            this.sendMessageToClient(clientId, 'Error: Could not access unit systems.');
            return;
        }
        
        let commandedCount = 0;
        for (let i = 0; i < worldCtx.soldierMgr.count; i++) {
            if (worldCtx.soldierMgr.owner && worldCtx.soldierMgr.owner[i] === team) {
                worldCtx.soldierMgr.cmdKind[i] = 2; // Attack-move command
                worldCtx.soldierMgr.cmdTX[i] = targetX;
                worldCtx.soldierMgr.cmdTY[i] = targetY;
                commandedCount++;
            }
        }
        
        this.sendMessageToClient(clientId, `Ordered ${commandedCount} units from team ${team} to attack-move to (${targetX}, ${targetY})`);
    }
    
    handleFormationCommand(clientId, args, playerData) {
        if (args.length < 2) {
            this.sendMessageToClient(clientId, 'Usage: /formation <team> <type>');
            this.sendMessageToClient(clientId, 'Formation types: line, wedge, column');
            return;
        }
        
        const team = args[0];
        const formationType = args[1].toLowerCase();
        
        if (!['line', 'wedge', 'column'].includes(formationType)) {
            this.sendMessageToClient(clientId, 'Invalid formation type. Use: line, wedge, or column');
            return;
        }
        
        this.sendMessageToClient(clientId, `Set formation ${formationType} for team ${team} (formation system in development)`);
    }
    
    handleClearCommand(clientId, args, playerData) {
        const team = args[0] || null; // Clear specific team or all
        
        const mapId = this.server.clients?.get(clientId)?.mapId || this.server.gameState?.mapId || 'map_1';
        const worldCtx = this.server.getWorldCtx ? this.server.getWorldCtx(mapId) : null;
        
        if (!worldCtx || !worldCtx.soldierMgr) {
            this.sendMessageToClient(clientId, 'Error: Could not access unit systems.');
            return;
        }
        
        let removedCount = 0;
        if (team) {
            // Remove units of specific team
            for (let i = worldCtx.soldierMgr.count - 1; i >= 0; i--) {
                if (worldCtx.soldierMgr.owner && worldCtx.soldierMgr.owner[i] === team) {
                    worldCtx.soldierMgr._removeAt(i);
                    removedCount++;
                }
            }
            this.sendMessageToClient(clientId, `Removed ${removedCount} units from team ${team}`);
        } else {
            // Clear all units
            removedCount = worldCtx.soldierMgr.count;
            worldCtx.soldierMgr.cleanup();
            this.sendMessageToClient(clientId, `Removed all ${removedCount} units`);
        }
    }
    
    handleStatusCommand(clientId, args, playerData) {
        const mapId = this.server.clients?.get(clientId)?.mapId || this.server.gameState?.mapId || 'map_1';
        const worldCtx = this.server.getWorldCtx ? this.server.getWorldCtx(mapId) : null;
        
        if (!worldCtx || !worldCtx.soldierMgr) {
            this.sendMessageToClient(clientId, 'Error: Could not access unit systems.');
            return;
        }
        
        const soldierMgr = worldCtx.soldierMgr;
        
        // Count units by type and team
        const stats = {
            total: soldierMgr.count,
            byType: {},
            byTeam: {}
        };
        
        const unitNames = ['Light Infantry', 'Heavy Infantry', 'Light Cavalry', 'Heavy Cavalry', 'Archer', 'Crossbowman'];
        
        for (let i = 0; i < soldierMgr.count; i++) {
            const type = soldierMgr.type ? soldierMgr.type[i] : (soldierMgr.typeIdx ? soldierMgr.typeIdx[i] : 0);
            const team = soldierMgr.owner ? soldierMgr.owner[i] : 'unknown';
            
            const typeName = unitNames[type] || `Type ${type}`;
            stats.byType[typeName] = (stats.byType[typeName] || 0) + 1;
            stats.byTeam[team] = (stats.byTeam[team] || 0) + 1;
        }
        
        let statusMsg = `Unit Status - Total: ${stats.total}\n`;

        if (Object.keys(stats.byType).length > 0) {
            statusMsg += 'By Type: ' + Object.entries(stats.byType)
                .map(([type, count]) => `${type}: ${count}`)
                .join(', ') + '\n';
        }
        
        if (Object.keys(stats.byTeam).length > 0) {
            statusMsg += 'By Team: ' + Object.entries(stats.byTeam)
                .map(([team, count]) => `${team}: ${count}`)
                .join(', ');
        }
        
        this.sendMessageToClient(clientId, statusMsg);
    }
    
    handleHelpCommand(clientId, args, playerData) {
        const commandList = Array.from(this.commands.entries())
            .map(([name, cmd]) => `/${name} - ${cmd.description}`)
            .join('\n');

        this.sendMessageToClient(clientId, 'Available Commands:\n' + commandList);
    }
    
    handleUnitsCommand(clientId, args, playerData) {
        const unitInfo = [
            '0: Light Infantry - Fast moving foot soldiers',
            '1: Heavy Infantry - Armored warriors with shields',
            '2: Light Cavalry - Fast mounted flankers',
            '3: Heavy Cavalry - Heavily armored knights',
            '4: Archer - Skilled bowmen with long range',
            '5: Crossbowman - Elite marksmen with high accuracy'
        ].join('\n');

        this.sendMessageToClient(clientId, 'Available Unit Types:\n' + unitInfo);
    }
}