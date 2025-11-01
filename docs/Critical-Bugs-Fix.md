# Critical Gameplay Bugs - Chat & Shooting

## Bug #1: Shooting Not Working 🔫

### Root Cause
**Client sends shooting:** `MessageType.BULLET_CREATE` (ID: 30)
**Server handles:** ❌ NOTHING - No handler exists!

### Evidence
**Client code** (`public/src/network/ClientNetworkManager.js:1125-1128`):
```javascript
sendShoot(bulletData) {
    console.log(`Sending shoot event...`);
    return this.send(MessageType.BULLET_CREATE, bulletData);
}
```

**Server code** (`Server.js:984-1000`):
```javascript
const packet = BinaryPacket.decode(buffer);
if(packet.type === MessageType.MOVE_ITEM){
  processMoveItem(clientId, packet.data);
} else if(packet.type === MessageType.PICKUP_ITEM){
  processPickupMessage(clientId, packet.data);
} else if(packet.type === MessageType.PLAYER_TEXT){
  // Handle chat
} else if(packet.type === MessageType.CHUNK_REQUEST){
  handleChunkRequest(clientId, packet.data);
} else {
  handleClientMessage(clientId, message); // Falls through to "unhandled"
}
```

**No BULLET_CREATE handler!**

### Fix Required
Add handler in `Server.js` around line 997:

```javascript
} else if(packet.type === MessageType.CHUNK_REQUEST){
  handleChunkRequest(clientId, packet.data);
} else if(packet.type === MessageType.BULLET_CREATE){
  handlePlayerShoot(clientId, packet.data);
} else {
  handleClientMessage(clientId, message);
}
```

And implement the function:

```javascript
function handlePlayerShoot(clientId, bulletData) {
  const client = clients.get(clientId);
  if (!client) return;

  const { x, y, angle, speed, damage } = bulletData;
  const mapId = client.mapId;

  // Get world context
  const ctx = getWorldCtx(mapId);
  if (!ctx || !ctx.bulletMgr) {
    console.error(`[SHOOT] No bullet manager for map ${mapId}`);
    return;
  }

  // Create bullet owned by player
  const bullet = {
    id: `bullet_${Date.now()}_${clientId}_${Math.random()}`,
    x: x || client.player.x,
    y: y || client.player.y,
    vx: Math.cos(angle) * speed,
    vy: Math.sin(angle) * speed,
    damage: damage || 10,
    owner: clientId,
    ownerType: 'player',
    worldId: mapId,
    createdAt: Date.now(),
    lifetime: 5000 // 5 seconds
  };

  // Add to bullet manager
  ctx.bulletMgr.addBullet(bullet);

  if (DEBUG.bulletEvents) {
    console.log(`[SHOOT] Player ${clientId} fired bullet at angle ${angle.toFixed(2)}`);
  }
}
```

---

## Bug #2: Chat Not Working 💬

### Root Cause
**Chat messages go to command system only**, not broadcasted to other players.

### Evidence
**Server code** (`Server.js:989-995`):
```javascript
} else if(packet.type === MessageType.PLAYER_TEXT){
  // Initialize command system if not already done
  const cmdSystem = initializeCommandSystem();
  const client = clients.get(clientId);
  if (client && packet.data && packet.data.text) {
    cmdSystem.processMessage(clientId, packet.data.text, client.player);
    // ❌ Does NOT broadcast to other players!
  }
}
```

### Fix Required
Modify to broadcast non-command messages:

```javascript
} else if(packet.type === MessageType.PLAYER_TEXT){
  const client = clients.get(clientId);
  if (!client || !packet.data || !packet.data.text) return;

  const text = packet.data.text;

  // Check if it's a command (starts with /)
  if (text.startsWith('/')) {
    const cmdSystem = initializeCommandSystem();
    cmdSystem.processMessage(clientId, text, client.player);
  } else {
    // Regular chat message - broadcast to all players in same world
    const chatMessage = {
      sender: client.player.name || `Player ${clientId}`,
      text: text,
      senderId: clientId,
      timestamp: Date.now()
    };

    // Broadcast to all players in same map
    clients.forEach((c, id) => {
      if (c.mapId === client.mapId) {
        sendToClient(c.socket, MessageType.CHAT_MESSAGE, chatMessage);
      }
    });

    if (DEBUG.chat) {
      console.log(`[CHAT] ${chatMessage.sender}: ${text}`);
    }
  }
}
```

---

## Implementation Steps

1. Add `handlePlayerShoot()` function to Server.js
2. Add BULLET_CREATE handler in message switch
3. Update PLAYER_TEXT handler to broadcast chat
4. Restart server
5. Test shooting and chat

---

## Data Format Validation

### Bullet Data Expected
Client sends:
```javascript
{
  x: 100.5,
  y: 200.3,
  angle: 1.57, // radians
  speed: 10,
  damage: 10
}
```

Server needs to create bullet with `vx, vy` from angle+speed.

### Chat Data Expected
Client sends:
```javascript
{
  text: "Hello world"
}
```

Server broadcasts:
```javascript
{
  sender: "PlayerName",
  text: "Hello world",
  senderId: 123,
  timestamp: 1234567890
}
```

---

## Additional Issues to Check

### 3. Player Movement
Check if `MessageType.PLAYER_UPDATE` is handled on server.

### 4. Collision Detection
Check if bullets actually hit enemies (collision system).

### 5. World Updates
Check if `WORLD_UPDATE` includes bullets so client can render them.
