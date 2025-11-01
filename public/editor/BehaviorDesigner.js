/**
 * Visual Behavior Designer - Build enemy behaviors with visual blocks
 *
 * Features:
 * - State-based visual editor
 * - Drag-and-drop behavior blocks
 * - Real-time DSL code generation
 * - Parameter adjustment UI
 * - Pattern library browser
 */

const { useState, useEffect, useRef } = React;

// ============= BUILDING BLOCKS LIBRARY =============
const MOVEMENT_BLOCKS = [
  { id: 'MoveToward', name: 'Move Toward', params: { target: 'ToPlayer', speed: 1.0 }, color: '#4CAF50' },
  { id: 'MoveAway', name: 'Move Away', params: { target: 'ToPlayer', distance: 300 }, color: '#4CAF50' },
  { id: 'Orbit', name: 'Orbit', params: { target: 'ToPlayer', radius: 200, speed: 1.0, clockwise: true }, color: '#4CAF50' },
  { id: 'ZigZag', name: 'Zig Zag', params: { target: 'ToPlayer', amplitude: 100, frequency: 2 }, color: '#4CAF50' },
  { id: 'Circle', name: 'Circle', params: { radius: 150, speed: 1.0 }, color: '#4CAF50' },
  { id: 'Charge', name: 'Charge', params: { target: 'ToPlayer', speed: 2.0, duration: 1000 }, color: '#4CAF50' },
  { id: 'Retreat', name: 'Retreat', params: { target: 'ToPlayer', distance: 400, speed: 1.5 }, color: '#4CAF50' },
  { id: 'Wander', name: 'Wander', params: { changeInterval: 1000 }, color: '#4CAF50' },
  { id: 'Stop', name: 'Stop', params: {}, color: '#4CAF50' }
];

const SHOOTING_BLOCKS = [
  { id: 'Shoot', name: 'Shoot', params: { angle: 'GetAngle()', speed: 25, bulletType: 'arrow' }, color: '#F44336' },
  { id: 'ShootPattern', name: 'Shoot Pattern', params: { pattern: 'triple_forward', angle: 'GetAngle()', speed: 25 }, color: '#F44336' },
  { id: 'ShootSpread', name: 'Shoot Spread', params: { angles: [-30, 0, 30], speed: 25 }, color: '#F44336' },
  { id: 'ShootSpiral', name: 'Shoot Spiral', params: { count: 8, rotationSpeed: 90, bulletSpeed: 25 }, color: '#F44336' },
  { id: 'ShootRing', name: 'Shoot Ring', params: { count: 12, speed: 25, bulletType: 'arrow' }, color: '#F44336' }
];

const STATE_BLOCKS = [
  { id: 'SetSpeed', name: 'Set Speed', params: { value: 1.0 }, color: '#2196F3' },
  { id: 'SetAngle', name: 'Set Angle', params: { value: 'ToPlayer' }, color: '#2196F3' },
  { id: 'SetVar', name: 'Set Variable', params: { name: 'myVar', value: 0 }, color: '#2196F3' }
];

const PATTERNS = [
  { id: 'triple_forward', name: 'Triple Forward', preview: '↑↑↑' },
  { id: 'five_spread', name: 'Five Spread', preview: '↗↑↖' },
  { id: 'ring', name: 'Ring', preview: '⊕' },
  { id: 'spiral', name: 'Spiral', preview: '⚪' },
  { id: 'cross', name: 'Cross', preview: '+' },
  { id: 'x_pattern', name: 'X Pattern', preview: '×' },
  { id: 'aimed', name: 'Aimed', preview: '→' }
];

// ============= BLOCK PALETTE =============
function BlockPalette({ onAddBlock }) {
  const [category, setCategory] = useState('movement');

  const blocks = category === 'movement' ? MOVEMENT_BLOCKS :
                 category === 'shooting' ? SHOOTING_BLOCKS : STATE_BLOCKS;

  return (
    <div style={{ padding: '10px', borderBottom: '1px solid #ccc' }}>
      <h4 style={{ margin: '0 0 10px 0' }}>Building Blocks</h4>

      <div style={{ marginBottom: '10px' }}>
        <button
          onClick={() => setCategory('movement')}
          style={{
            padding: '5px 10px',
            marginRight: '5px',
            background: category === 'movement' ? '#4CAF50' : '#ddd',
            color: category === 'movement' ? 'white' : 'black',
            border: 'none',
            cursor: 'pointer'
          }}>
          Movement
        </button>
        <button
          onClick={() => setCategory('shooting')}
          style={{
            padding: '5px 10px',
            marginRight: '5px',
            background: category === 'shooting' ? '#F44336' : '#ddd',
            color: category === 'shooting' ? 'white' : 'black',
            border: 'none',
            cursor: 'pointer'
          }}>
          Shooting
        </button>
        <button
          onClick={() => setCategory('state')}
          style={{
            padding: '5px 10px',
            background: category === 'state' ? '#2196F3' : '#ddd',
            color: category === 'state' ? 'white' : 'black',
            border: 'none',
            cursor: 'pointer'
          }}>
          State
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px' }}>
        {blocks.map(block => (
          <div
            key={block.id}
            onClick={() => onAddBlock(block)}
            style={{
              padding: '10px',
              background: block.color,
              color: 'white',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '12px',
              textAlign: 'center',
              fontWeight: 'bold'
            }}>
            {block.name}
          </div>
        ))}
      </div>
    </div>
  );
}

// ============= PARAMETER EDITOR =============
function ParameterEditor({ block, onChange, onClose }) {
  const [params, setParams] = useState(block.params);

  const handleChange = (key, value) => {
    const newParams = { ...params, [key]: value };
    setParams(newParams);
  };

  const handleSave = () => {
    onChange(newParams);
    onClose();
  };

  return (
    <div style={{
      position: 'fixed',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      background: 'white',
      border: '2px solid #333',
      borderRadius: '8px',
      padding: '20px',
      zIndex: 1000,
      minWidth: '300px',
      maxHeight: '80vh',
      overflowY: 'auto'
    }}>
      <h3 style={{ margin: '0 0 15px 0' }}>{block.name} Parameters</h3>

      {Object.entries(params).map(([key, value]) => (
        <div key={key} style={{ marginBottom: '12px' }}>
          <label style={{ display: 'block', fontSize: '11px', marginBottom: '4px', fontWeight: 'bold' }}>
            {key}:
          </label>

          {typeof value === 'boolean' ? (
            <input
              type="checkbox"
              checked={value}
              onChange={(e) => handleChange(key, e.target.checked)}
            />
          ) : typeof value === 'number' ? (
            <input
              type="number"
              value={value}
              onChange={(e) => handleChange(key, parseFloat(e.target.value))}
              style={{ width: '100%', padding: '5px' }}
            />
          ) : Array.isArray(value) ? (
            <input
              type="text"
              value={JSON.stringify(value)}
              onChange={(e) => {
                try {
                  handleChange(key, JSON.parse(e.target.value));
                } catch {}
              }}
              style={{ width: '100%', padding: '5px' }}
            />
          ) : (
            <input
              type="text"
              value={value}
              onChange={(e) => handleChange(key, e.target.value)}
              style={{ width: '100%', padding: '5px' }}
            />
          )}
        </div>
      ))}

      <div style={{ display: 'flex', gap: '10px', marginTop: '20px' }}>
        <button
          onClick={handleSave}
          style={{
            flex: 1,
            padding: '10px',
            background: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}>
          Save
        </button>
        <button
          onClick={onClose}
          style={{
            flex: 1,
            padding: '10px',
            background: '#f44336',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}>
          Cancel
        </button>
      </div>
    </div>
  );
}

// ============= STATE EDITOR =============
function StateEditor({ state, onUpdate, onDelete, selected, onSelect }) {
  const [editingBlock, setEditingBlock] = useState(null);

  const addBlock = (block, phase) => {
    const newState = { ...state };
    const newBlock = { ...block, id: `${block.id}_${Date.now()}`, params: { ...block.params } };

    if (phase === 'onEnter') newState.onEnter.push(newBlock);
    else if (phase === 'onTick') newState.onTick.push(newBlock);
    else if (phase === 'onExit') newState.onExit.push(newBlock);

    onUpdate(newState);
  };

  const removeBlock = (blockId, phase) => {
    const newState = { ...state };
    if (phase === 'onEnter') newState.onEnter = newState.onEnter.filter(b => b.id !== blockId);
    else if (phase === 'onTick') newState.onTick = newState.onTick.filter(b => b.id !== blockId);
    else if (phase === 'onExit') newState.onExit = newState.onExit.filter(b => b.id !== blockId);
    onUpdate(newState);
  };

  const updateBlock = (blockId, phase, newParams) => {
    const newState = { ...state };
    let blocks = phase === 'onEnter' ? newState.onEnter :
                 phase === 'onTick' ? newState.onTick : newState.onExit;

    const blockIndex = blocks.findIndex(b => b.id === blockId);
    if (blockIndex >= 0) {
      blocks[blockIndex] = { ...blocks[blockIndex], params: newParams };
      onUpdate(newState);
    }
  };

  const renderBlockList = (blocks, phase) => (
    <div style={{ marginBottom: '10px' }}>
      {blocks.map(block => (
        <div
          key={block.id}
          style={{
            padding: '8px',
            margin: '4px 0',
            background: block.color,
            color: 'white',
            borderRadius: '4px',
            fontSize: '11px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
          <span style={{ fontWeight: 'bold' }}>{block.name}</span>
          <div>
            <button
              onClick={() => setEditingBlock({ block, phase })}
              style={{
                padding: '2px 6px',
                marginRight: '4px',
                background: 'rgba(255,255,255,0.3)',
                border: 'none',
                borderRadius: '3px',
                cursor: 'pointer',
                color: 'white'
              }}>
              ⚙
            </button>
            <button
              onClick={() => removeBlock(block.id, phase)}
              style={{
                padding: '2px 6px',
                background: 'rgba(255,255,255,0.3)',
                border: 'none',
                borderRadius: '3px',
                cursor: 'pointer',
                color: 'white'
              }}>
              ×
            </button>
          </div>
        </div>
      ))}
    </div>
  );

  return (
    <div
      onClick={() => onSelect(state)}
      style={{
        padding: '15px',
        margin: '10px',
        border: selected ? '3px solid #2196F3' : '2px solid #ccc',
        borderRadius: '8px',
        background: selected ? '#E3F2FD' : 'white'
      }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
        <input
          type="text"
          value={state.name}
          onChange={(e) => onUpdate({ ...state, name: e.target.value })}
          style={{
            fontSize: '16px',
            fontWeight: 'bold',
            border: 'none',
            background: 'transparent',
            flex: 1
          }}
        />
        <button
          onClick={() => onDelete(state.id)}
          style={{
            padding: '5px 10px',
            background: '#f44336',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}>
          Delete State
        </button>
      </div>

      <div style={{ marginBottom: '10px' }}>
        <label style={{ fontSize: '11px', fontWeight: 'bold' }}>Duration (ms):</label>
        <input
          type="number"
          value={state.duration || ''}
          onChange={(e) => onUpdate({ ...state, duration: parseInt(e.target.value) || null })}
          placeholder="No limit"
          style={{ width: '100%', padding: '5px', marginTop: '4px' }}
        />
      </div>

      <div style={{ marginBottom: '10px' }}>
        <label style={{ fontSize: '11px', fontWeight: 'bold' }}>Next State:</label>
        <input
          type="text"
          value={state.next || ''}
          onChange={(e) => onUpdate({ ...state, next: e.target.value })}
          placeholder="State name"
          style={{ width: '100%', padding: '5px', marginTop: '4px' }}
        />
      </div>

      <div style={{ marginTop: '15px' }}>
        <h5 style={{ margin: '5px 0', fontSize: '12px' }}>ON ENTER (runs once when entering state)</h5>
        {renderBlockList(state.onEnter, 'onEnter')}
        <BlockPalette onAddBlock={(block) => addBlock(block, 'onEnter')} />
      </div>

      <div style={{ marginTop: '15px' }}>
        <h5 style={{ margin: '5px 0', fontSize: '12px' }}>TICK (runs every frame)</h5>
        {renderBlockList(state.onTick, 'onTick')}
        <BlockPalette onAddBlock={(block) => addBlock(block, 'onTick')} />
      </div>

      <div style={{ marginTop: '15px' }}>
        <h5 style={{ margin: '5px 0', fontSize: '12px' }}>ON EXIT (runs once when leaving state)</h5>
        {renderBlockList(state.onExit, 'onExit')}
        <BlockPalette onAddBlock={(block) => addBlock(block, 'onExit')} />
      </div>

      {editingBlock && (
        <>
          <div
            onClick={() => setEditingBlock(null)}
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'rgba(0,0,0,0.5)',
              zIndex: 999
            }}
          />
          <ParameterEditor
            block={editingBlock.block}
            onChange={(params) => updateBlock(editingBlock.block.id, editingBlock.phase, params)}
            onClose={() => setEditingBlock(null)}
          />
        </>
      )}
    </div>
  );
}

// ============= DSL CODE VIEWER =============
function DSLCodeViewer({ states }) {
  const generateDSL = () => {
    let code = '// Generated Behavior DSL\n\n';

    for (const state of states) {
      code += `STATE ${state.name} {\n`;

      // ON_ENTER
      if (state.onEnter.length > 0) {
        for (const block of state.onEnter) {
          code += `  ON_ENTER: ${generateBlockCall(block)}\n`;
        }
      }

      // TICK
      if (state.onTick.length > 0) {
        for (const block of state.onTick) {
          code += `  TICK: ${generateBlockCall(block)}\n`;
        }
      }

      // ON_EXIT
      if (state.onExit.length > 0) {
        for (const block of state.onExit) {
          code += `  ON_EXIT: ${generateBlockCall(block)}\n`;
        }
      }

      // DURATION
      if (state.duration) {
        code += `  DURATION: ${state.duration}ms\n`;
      }

      // NEXT
      if (state.next) {
        code += `  NEXT: ${state.next}\n`;
      }

      code += `}\n\n`;
    }

    return code;
  };

  const generateBlockCall = (block) => {
    const params = Object.entries(block.params)
      .map(([key, value]) => {
        if (typeof value === 'string') {
          return `${key}="${value}"`;
        }
        return `${key}=${JSON.stringify(value)}`;
      })
      .join(', ');

    return `${block.id.split('_')[0]}(${params})`;
  };

  return (
    <div style={{ padding: '15px', background: '#1e1e1e', color: '#d4d4d4', borderRadius: '8px', fontFamily: 'monospace', fontSize: '12px', whiteSpace: 'pre-wrap', maxHeight: '400px', overflowY: 'auto' }}>
      {generateDSL()}
    </div>
  );
}

// ============= MAIN BEHAVIOR DESIGNER =============
function BehaviorDesigner() {
  const [states, setStates] = useState([]);
  const [selectedState, setSelectedState] = useState(null);
  const [behaviorName, setBehaviorName] = useState('MyBehavior');
  const [showCode, setShowCode] = useState(false);
  const [showTestArena, setShowTestArena] = useState(true);

  const addState = () => {
    const newState = {
      id: `state_${Date.now()}`,
      name: `state_${states.length + 1}`,
      onEnter: [],
      onTick: [],
      onExit: [],
      duration: null,
      next: null
    };
    setStates([...states, newState]);
    setSelectedState(newState);
  };

  const updateState = (updatedState) => {
    setStates(states.map(s => s.id === updatedState.id ? updatedState : s));
    setSelectedState(updatedState);
  };

  const deleteState = (stateId) => {
    setStates(states.filter(s => s.id !== stateId));
    if (selectedState && selectedState.id === stateId) {
      setSelectedState(null);
    }
  };

  const loadBehavior = async (name) => {
    try {
      const response = await fetch(`/api/behavior-designer/load/${name}`);
      const result = await response.json();

      if (result.success) {
        setBehaviorName(result.behavior.name);
        setStates(result.behavior.states);
        setSelectedState(null);
        console.log('[BehaviorDesigner] Loaded behavior:', name);
      } else {
        alert('Failed to load: ' + result.error);
      }
    } catch (err) {
      alert('Failed to load: ' + err.message);
    }
  };

  const saveBehavior = async () => {
    const dsl = generateDSLFromStates();

    const behaviorData = {
      name: behaviorName,
      dsl,
      states
    };

    try {
      const response = await fetch('/api/behavior-designer/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(behaviorData)
      });

      const result = await response.json();
      if (result.success) {
        alert(`Behavior "${behaviorName}" saved!`);
      } else {
        alert('Failed to save: ' + result.error);
      }
    } catch (err) {
      alert('Failed to save: ' + err.message);
    }
  };

  const generateDSLFromStates = () => {
    let code = '';

    for (const state of states) {
      code += `STATE ${state.name} {\n`;

      for (const block of state.onEnter) {
        code += `  ON_ENTER: ${generateBlockCall(block)}\n`;
      }

      for (const block of state.onTick) {
        code += `  TICK: ${generateBlockCall(block)}\n`;
      }

      for (const block of state.onExit) {
        code += `  ON_EXIT: ${generateBlockCall(block)}\n`;
      }

      if (state.duration) {
        code += `  DURATION: ${state.duration}ms\n`;
      }

      if (state.next) {
        code += `  NEXT: ${state.next}\n`;
      }

      code += `}\n\n`;
    }

    return code;
  };

  const generateBlockCall = (block) => {
    const params = Object.entries(block.params)
      .map(([key, value]) => `${key}=${typeof value === 'string' ? `"${value}"` : JSON.stringify(value)}`)
      .join(', ');
    return `${block.id.split('_')[0]}(${params})`;
  };

  return (
    <div style={{ width: '100%', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <div style={{ padding: '15px', background: '#333', color: 'white', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <h2 style={{ margin: 0 }}>Behavior Designer</h2>
          <input
            type="text"
            value={behaviorName}
            onChange={(e) => setBehaviorName(e.target.value)}
            style={{ padding: '8px', fontSize: '14px', borderRadius: '4px', border: 'none' }}
            placeholder="Behavior name"
          />
        </div>

        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            onClick={() => loadBehavior('ChargingShooter')}
            style={{
              padding: '10px 20px',
              background: '#4CAF50',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }}>
            Load ChargingShooter
          </button>

          <button
            onClick={() => setShowTestArena(!showTestArena)}
            style={{
              padding: '10px 20px',
              background: '#9C27B0',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }}>
            {showTestArena ? 'Hide' : 'Show'} Test Arena
          </button>

          <button
            onClick={() => setShowCode(!showCode)}
            style={{
              padding: '10px 20px',
              background: '#2196F3',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }}>
            {showCode ? 'Hide' : 'Show'} Code
          </button>

          <button
            onClick={saveBehavior}
            style={{
              padding: '10px 20px',
              background: '#FF9800',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }}>
            Save Behavior
          </button>
        </div>
      </div>

      {/* Main content - Unity-style layout */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* LEFT SIDEBAR - Inspector (Unity-style) */}
        <div style={{ width: '350px', background: '#2a2a2a', borderRight: '1px solid #444', overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
          {/* State Inspector */}
          {selectedState ? (
            <div style={{ padding: '15px', color: 'white' }}>
              <h3 style={{ margin: '0 0 15px 0', color: '#64B5F6', borderBottom: '2px solid #64B5F6', paddingBottom: '8px' }}>
                Inspector
              </h3>

              <div style={{ marginBottom: '20px', background: '#333', padding: '12px', borderRadius: '6px' }}>
                <label style={{ display: 'block', fontSize: '11px', fontWeight: 'bold', marginBottom: '6px', color: '#aaa', textTransform: 'uppercase' }}>
                  State Name
                </label>
                <input
                  type="text"
                  value={selectedState.name}
                  onChange={(e) => updateState({ ...selectedState, name: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '8px',
                    fontSize: '14px',
                    borderRadius: '4px',
                    border: '1px solid #555',
                    background: '#1a1a1a',
                    color: 'white'
                  }}
                />
              </div>

              <div style={{ marginBottom: '20px', background: '#333', padding: '12px', borderRadius: '6px' }}>
                <label style={{ display: 'block', fontSize: '11px', fontWeight: 'bold', marginBottom: '6px', color: '#aaa', textTransform: 'uppercase' }}>
                  Duration (ms)
                </label>
                <input
                  type="number"
                  value={selectedState.duration || ''}
                  onChange={(e) => updateState({ ...selectedState, duration: parseInt(e.target.value) || null })}
                  placeholder="Auto transition time"
                  style={{
                    width: '100%',
                    padding: '8px',
                    fontSize: '14px',
                    borderRadius: '4px',
                    border: '1px solid #555',
                    background: '#1a1a1a',
                    color: 'white'
                  }}
                />
              </div>

              <div style={{ marginBottom: '20px', background: '#333', padding: '12px', borderRadius: '6px' }}>
                <label style={{ display: 'block', fontSize: '11px', fontWeight: 'bold', marginBottom: '6px', color: '#aaa', textTransform: 'uppercase' }}>
                  Next State
                </label>
                <select
                  value={selectedState.next || ''}
                  onChange={(e) => updateState({ ...selectedState, next: e.target.value || null })}
                  style={{
                    width: '100%',
                    padding: '8px',
                    fontSize: '14px',
                    borderRadius: '4px',
                    border: '1px solid #555',
                    background: '#1a1a1a',
                    color: 'white'
                  }}>
                  <option value="">None</option>
                  {states.filter(s => s.id !== selectedState.id).map(s => (
                    <option key={s.id} value={s.name}>{s.name}</option>
                  ))}
                </select>
              </div>

              <div style={{ borderTop: '1px solid #444', paddingTop: '15px' }}>
                <StateEditor
                  state={selectedState}
                  onUpdate={updateState}
                  onDelete={deleteState}
                  selected={true}
                  onSelect={setSelectedState}
                />
              </div>
            </div>
          ) : (
            <div style={{ padding: '20px', color: '#888', textAlign: 'center' }}>
              <h3 style={{ color: '#64B5F6', marginBottom: '10px' }}>Inspector</h3>
              <p>Select a state to edit its properties</p>
              <div style={{ marginTop: '20px', padding: '15px', background: '#333', borderRadius: '6px', textAlign: 'left', fontSize: '13px' }}>
                <strong style={{ color: '#4CAF50' }}>Quick Start:</strong>
                <ul style={{ marginTop: '10px', paddingLeft: '20px', lineHeight: '1.8' }}>
                  <li>Double-click canvas to add states</li>
                  <li>Click & drag to move states</li>
                  <li>Shift+click to connect states</li>
                  <li>Click state to edit behaviors</li>
                </ul>
              </div>
            </div>
          )}
        </div>

        {/* CENTER/RIGHT - Visual State Machine Graph */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '10px', background: '#1a1a1a', display: 'flex', flexDirection: 'column' }}>
          {window.VisualStateMachine ? (
            <window.VisualStateMachine
              states={states}
              onStatesChange={setStates}
              selectedState={selectedState}
              onSelectState={setSelectedState}
            />
          ) : (
            <div style={{ padding: '20px', color: '#999' }}>Loading visual editor...</div>
          )}
        </div>

        {/* RIGHT PANELS - Test Arena & Code */}
        {(showTestArena || showCode) && (
          <div style={{ width: '450px', background: '#2d2d2d', borderLeft: '1px solid #444', display: 'flex', flexDirection: 'column' }}>
            {showTestArena && window.BehaviorTestArena && (
              <div style={{ flex: showCode ? '1' : 'auto', padding: '15px', overflowY: 'auto', borderBottom: showCode ? '1px solid #444' : 'none' }}>
                <window.BehaviorTestArena states={states} behaviorName={behaviorName} />
              </div>
            )}

            {showCode && (
              <div style={{ flex: showTestArena ? '1' : 'auto', padding: '15px', overflowY: 'auto' }}>
                <h3 style={{ color: 'white', marginTop: 0 }}>Generated DSL Code</h3>
                <DSLCodeViewer states={states} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// Export for use in other files
window.BehaviorDesigner = BehaviorDesigner;

// Render if this is the main page
if (document.getElementById('behavior-designer-root')) {
  ReactDOM.createRoot(document.getElementById('behavior-designer-root')).render(<BehaviorDesigner />);
}
