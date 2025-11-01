// tests/dsl-interpreter.spec.js
import DslInterpreter from '../src/boss/DslInterpreter.js';
import { registry } from '../src/registry/index.js';

describe('DslInterpreter smoke test', () => {
  it('emits mutator nodes for demo script', () => {
    const demo = {
      nodes: [
        { type: 'Emitter:RadialBurst@1.0.0', projectiles: 6 },
        { type: 'Core:Wait@1.0.0', t: 0.05 },
        { parallel: [
            { type: 'Movement:Dash@1.0.0', dx: 3, duration: 0.3 },
            { sequence: [
                { type: 'Emitter:RadialBurst@1.0.0', projectiles: 12 },
                { type: 'Core:Wait@1.0.0', t: 0.02 }
            ]}
        ]}
      ]
    };
    const interp = new DslInterpreter();
    interp.load(demo.nodes);
    const out1 = interp.tick(0.016);
    expect(out1[0].args.projectiles).toBe(6);
    // advance beyond wait
    interp.tick(0.06);
    const out2 = interp.tick(0.1);
    // should include at least one compiled node (dash or radial burst)
    const types = out2.map(n=>n._capType);
    expect(types.length).toBeGreaterThan(0);
  });
}); 