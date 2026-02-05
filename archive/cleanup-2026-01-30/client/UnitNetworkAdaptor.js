/**
 *  Front‑end glue: plugs client UnitManager into OptimizedNetworkManager.
 *  Usage in bootstrap:
 *      import initUnitNet from './net/unitNetAdapter.js';
 *      initUnitNet(net, gameState.units);
 */

import ClientUnitManager from '../units/ClientUnitManager.js';
import { MessageType } from '../../Managers/NetworkManager.js';

export default function initUnitNet(net, unitMgr) {
  // unit create (initial sync)
  net.on(MessageType.UNIT_CREATE, arr => unitMgr.spawnMany(arr));

  // incremental updates
  net.on(MessageType.UNIT_UPDATE, arr => {
    arr.forEach(u => unitMgr.applyUpdate(u));
  });

  // removals
  net.on(MessageType.UNIT_REMOVE, ids => unitMgr.remove(ids));
}

/** Helper for the UI: send a command */
export function sendUnitCommand(net, cmd, unitIds, tx, ty) {
  net.send(MessageType.UNIT_COMMAND, { cmd, unitIds, tx, ty });
}
