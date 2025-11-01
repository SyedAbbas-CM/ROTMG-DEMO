/**
 * UnitNetworkAdapter – bridges UnitManager ↔ client WebSockets
 * Drop‐in usage inside server.js:
 *
 *    import UnitNetwork from './network/UnitNetworkAdapter.js';
 *    const unitNet = new UnitNetwork(wss, unitManager, unitSystems);
 *
 * inside game‑loop:  unitNet.broadcastUpdates();
 * on ws msg:         unitNet.handleClientPacket(clientId, packet);
 */

import { BinaryPacket, MessageType } from '../../common/protocol.js';

export default class UnitNetworkAdapter {
  constructor(wss, unitManager, unitSystems) {
    this.wss = wss;
    this.um  = unitManager;
    this.sys = unitSystems;

    // store unsent diffs per tick – here we simply resend a full SoA diff
    this.lastBroadcastTime = 0;
  }

  /** Called inside ws.on('message') before generic switch‑case. */
  handleClientPacket(clientId, { type, data }) {
    if (type !== MessageType.UNIT_COMMAND) return false;

    // { cmd:'move'|'attack', unitIds:[], tx, ty }
    const { cmd, unitIds, tx, ty } = data;
    for (const uid of unitIds) {
      const idx = this.um.indexFromId(uid);
      if (idx === -1) continue;
      switch (cmd) {
        case 'move':   this.sys.issueMove(idx, tx, ty);   break;
        case 'attack': this.sys.issueAttack(idx, tx, ty); break;
      }
    }
    return true;
  }

  /** Broadcast full unit snapshot every 100 ms (10 Hz) */
  broadcastUpdates() {
    const now = Date.now();
    if (now - this.lastBroadcastTime < 100) return;

    const payload = this.um.getSnapshot();
    this.broadcast(MessageType.UNIT_UPDATE, payload);
    this.lastBroadcastTime = now;
  }

  /** helper */
  broadcast(type, data) {
    const pkt = BinaryPacket.encode(type, data);
    this.wss.clients.forEach(c => c.readyState === 1 && c.send(pkt));
  }
}
