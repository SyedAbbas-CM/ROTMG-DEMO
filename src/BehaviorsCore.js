// BehaviorsCore.js – legacy shim
// This file now simply re-exports the full RotMG-style behaviour implementations
// defined in src/Behaviors.js.  Keeping this shim means we don’t have to touch
// existing import paths (BehaviourSystem, BehaviourTree, etc.).

export * from './Behaviors.js';
export * from './CompositeBehaviors.js'; 