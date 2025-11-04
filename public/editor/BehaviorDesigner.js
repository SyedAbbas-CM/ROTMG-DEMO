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
  const [category, setCategory] = React.useState('movement');

  const blocks = category === 'movement' ? MOVEMENT_BLOCKS :
                 category === 'shooting' ? SHOOTING_BLOCKS : STATE_BLOCKS;

  return React.createElement('div', { style: { padding: '10px', borderBottom: '1px solid #ccc' } },
    React.createElement('h4', { style: { margin: '0 0 10px 0' } }, 'Building Blocks'),
    React.createElement('div', { style: { marginBottom: '10px' } },
      React.createElement('button', {
        onClick: () => setCategory('movement'),
        style: {
          padding: '5px 10px',
          marginRight: '5px',
          background: category === 'movement' ? '#4CAF50' : '#ddd',
          color: category === 'movement' ? 'white' : 'black',
          border: 'none',
          cursor: 'pointer'
        }
      }, 'Movement'),
      React.createElement('button', {
        onClick: () => setCategory('shooting'),
        style: {
          padding: '5px 10px',
          marginRight: '5px',
          background: category === 'shooting' ? '#F44336' : '#ddd',
          color: category === 'shooting' ? 'white' : 'black',
          border: 'none',
          cursor: 'pointer'
        }
      }, 'Shooting'),
      React.createElement('button', {
        onClick: () => setCategory('state'),
        style: {
          padding: '5px 10px',
          background: category === 'state' ? '#2196F3' : '#ddd',
          color: category === 'state' ? 'white' : 'black',
          border: 'none',
          cursor: 'pointer'
        }
      }, 'State')
    ),
    React.createElement('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px' } },
      blocks.map(block =>
        React.createElement('div', {
          key: block.id,
          onClick: () => onAddBlock(block),
          style: {
            padding: '10px',
            background: block.color,
            color: 'white',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '12px',
            textAlign: 'center',
            fontWeight: 'bold'
          }
        }, block.name)
      )
    )
  );
}

// ============= PARAMETER EDITOR =============
function ParameterEditor({ block, onChange, onClose }) {
  const [params, setParams] = React.useState(block.params);

  const handleChange = (key, value) => {
    const newParams = { ...params, [key]: value };
    setParams(newParams);
  };

  const handleSave = () => {
    onChange(newParams);
    onClose();
  };

  return React.createElement('div', {
    style: {
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
    }
  },
    React.createElement('h3', { style: { margin: '0 0 15px 0' } }, block.name + ' Parameters'),
    Object.entries(params).map(([key, value]) =>
      React.createElement('div', { key: key, style: { marginBottom: '12px' } },
        React.createElement('label', { style: { display: 'block', fontSize: '11px', marginBottom: '4px', fontWeight: 'bold' } }, key + ':'),
        typeof value === 'boolean' ?
          React.createElement('input', {
            type: 'checkbox',
            checked: value,
            onChange: (e) => handleChange(key, e.target.checked)
          })
        : typeof value === 'number' ?
          React.createElement('input', {
            type: 'number',
            value: value,
            onChange: (e) => handleChange(key, parseFloat(e.target.value)),
            style: { width: '100%', padding: '5px' }
          })
        : Array.isArray(value) ?
          React.createElement('input', {
            type: 'text',
            value: JSON.stringify(value),
            onChange: (e) => {
              try {
                handleChange(key, JSON.parse(e.target.value));
              } catch {}
            },
            style: { width: '100%', padding: '5px' }
          })
        :
          React.createElement('input', {
            type: 'text',
            value: value,
            onChange: (e) => handleChange(key, e.target.value),
            style: { width: '100%', padding: '5px' }
          })
      )
    ),
    React.createElement('div', { style: { display: 'flex', gap: '10px', marginTop: '20px' } },
      React.createElement('button', {
        onClick: handleSave,
        style: {
          flex: 1,
          padding: '10px',
          background: '#4CAF50',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
          fontWeight: 'bold'
        }
      }, 'Save'),
      React.createElement('button', {
        onClick: onClose,
        style: {
          flex: 1,
          padding: '10px',
          background: '#f44336',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
          fontWeight: 'bold'
        }
      }, 'Cancel')
    )
  );
}

// ============= STATE EDITOR =============
function StateEditor({ state, onUpdate, onDelete, selected, onSelect }) {
  const [editingBlock, setEditingBlock] = React.useState(null);

  const addBlock = (block, phase) => {
    const newState = { ...state };
    const newBlock = { ...block, id: `${block.id}_${Date.now()}`, params: { ...block.params } };

    // Ensure arrays exist (for backward compatibility)
    if (phase === 'movement') {
      if (!newState.movement) newState.movement = [];
      newState.movement.push(newBlock);
    }
    else if (phase === 'attack') {
      if (!newState.attack) newState.attack = [];
      newState.attack.push(newBlock);
    }
    else if (phase === 'utility') {
      if (!newState.utility) newState.utility = [];
      newState.utility.push(newBlock);
    }
    else if (phase === 'onEnter') {
      if (!newState.onEnter) newState.onEnter = [];
      newState.onEnter.push(newBlock);
    }
    else if (phase === 'onExit') {
      if (!newState.onExit) newState.onExit = [];
      newState.onExit.push(newBlock);
    }

    onUpdate(newState);
  };

  const removeBlock = (blockId, phase) => {
    const newState = { ...state };
    if (phase === 'movement' && newState.movement) newState.movement = newState.movement.filter(b => b.id !== blockId);
    else if (phase === 'attack' && newState.attack) newState.attack = newState.attack.filter(b => b.id !== blockId);
    else if (phase === 'utility' && newState.utility) newState.utility = newState.utility.filter(b => b.id !== blockId);
    else if (phase === 'onEnter' && newState.onEnter) newState.onEnter = newState.onEnter.filter(b => b.id !== blockId);
    else if (phase === 'onExit' && newState.onExit) newState.onExit = newState.onExit.filter(b => b.id !== blockId);
    onUpdate(newState);
  };

  const updateBlock = (blockId, phase, newParams) => {
    const newState = { ...state };
    let blocks = phase === 'movement' ? (newState.movement || []) :
                 phase === 'attack' ? (newState.attack || []) :
                 phase === 'utility' ? (newState.utility || []) :
                 phase === 'onEnter' ? (newState.onEnter || []) : (newState.onExit || []);

    const blockIndex = blocks.findIndex(b => b.id === blockId);
    if (blockIndex >= 0) {
      blocks[blockIndex] = { ...blocks[blockIndex], params: newParams };
      onUpdate(newState);
    }
  };

  const renderBlockList = (blocks, phase) =>
    React.createElement('div', { style: { marginBottom: '10px' } },
      blocks.map(block =>
        React.createElement('div', {
          key: block.id,
          style: {
            padding: '8px',
            margin: '4px 0',
            background: block.color,
            color: 'white',
            borderRadius: '4px',
            fontSize: '11px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }
        },
          React.createElement('span', { style: { fontWeight: 'bold' } }, block.name),
          React.createElement('div', null,
            React.createElement('button', {
              onClick: () => setEditingBlock({ block, phase }),
              style: {
                padding: '2px 6px',
                marginRight: '4px',
                background: 'rgba(255,255,255,0.3)',
                border: 'none',
                borderRadius: '3px',
                cursor: 'pointer',
                color: 'white'
              }
            }, '\u2699'),
            React.createElement('button', {
              onClick: () => removeBlock(block.id, phase),
              style: {
                padding: '2px 6px',
                background: 'rgba(255,255,255,0.3)',
                border: 'none',
                borderRadius: '3px',
                cursor: 'pointer',
                color: 'white'
              }
            }, '\u00D7')
          )
        )
      )
    );

  return React.createElement('div', {
    onClick: () => onSelect(state),
    style: {
      padding: '15px',
      margin: '10px',
      border: selected ? '3px solid #2196F3' : '2px solid #ccc',
      borderRadius: '8px',
      background: selected ? '#E3F2FD' : 'white'
    }
  },
    React.createElement('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' } },
      React.createElement('input', {
        type: 'text',
        value: state.name,
        onChange: (e) => onUpdate({ ...state, name: e.target.value }),
        style: {
          fontSize: '16px',
          fontWeight: 'bold',
          border: 'none',
          background: 'transparent',
          flex: 1
        }
      }),
      React.createElement('button', {
        onClick: () => onDelete(state.id),
        style: {
          padding: '5px 10px',
          background: '#f44336',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
          fontWeight: 'bold'
        }
      }, 'Delete State')
    ),
    React.createElement('div', { style: { marginBottom: '10px' } },
      React.createElement('label', { style: { fontSize: '11px', fontWeight: 'bold' } }, 'Duration (ms):'),
      React.createElement('input', {
        type: 'number',
        value: state.duration || '',
        onChange: (e) => onUpdate({ ...state, duration: parseInt(e.target.value) || null }),
        placeholder: 'No limit',
        style: { width: '100%', padding: '5px', marginTop: '4px' }
      })
    ),
    React.createElement('div', { style: { marginBottom: '10px' } },
      React.createElement('label', { style: { fontSize: '11px', fontWeight: 'bold' } }, 'Next State:'),
      React.createElement('input', {
        type: 'text',
        value: state.next || '',
        onChange: (e) => onUpdate({ ...state, next: e.target.value }),
        placeholder: 'State name',
        style: { width: '100%', padding: '5px', marginTop: '4px' }
      })
    ),
    React.createElement('div', { style: { marginTop: '15px', background: '#1b5e20', padding: '10px', borderRadius: '6px', border: '2px solid #4CAF50' } },
      React.createElement('h5', { style: { margin: '5px 0', fontSize: '12px', color: '#81C784', fontWeight: 'bold' } }, '\u25CF MOVEMENT (runs continuously)'),
      renderBlockList(state.movement || [], 'movement'),
      React.createElement(BlockPalette, { onAddBlock: (block) => addBlock(block, 'movement') })
    ),
    React.createElement('div', { style: { marginTop: '15px', background: '#b71c1c', padding: '10px', borderRadius: '6px', border: '2px solid #F44336' } },
      React.createElement('h5', { style: { margin: '5px 0', fontSize: '12px', color: '#EF9A9A', fontWeight: 'bold' } }, '\u25CF ATTACK (runs continuously)'),
      renderBlockList(state.attack || [], 'attack'),
      React.createElement(BlockPalette, { onAddBlock: (block) => addBlock(block, 'attack') })
    ),
    React.createElement('div', { style: { marginTop: '15px', background: '#0d47a1', padding: '10px', borderRadius: '6px', border: '2px solid #2196F3' } },
      React.createElement('h5', { style: { margin: '5px 0', fontSize: '12px', color: '#90CAF9', fontWeight: 'bold' } }, '\u25CF UTILITY (runs continuously)'),
      renderBlockList(state.utility || [], 'utility'),
      React.createElement(BlockPalette, { onAddBlock: (block) => addBlock(block, 'utility') })
    ),
    React.createElement('div', { style: { marginTop: '15px', background: '#2a2a2a', padding: '10px', borderRadius: '6px', border: '1px solid #555' } },
      React.createElement('h5', { style: { margin: '5px 0', fontSize: '12px', color: '#999' } }, 'ON ENTER (runs once when entering state)'),
      renderBlockList(state.onEnter || [], 'onEnter'),
      React.createElement(BlockPalette, { onAddBlock: (block) => addBlock(block, 'onEnter') })
    ),
    React.createElement('div', { style: { marginTop: '15px', background: '#2a2a2a', padding: '10px', borderRadius: '6px', border: '1px solid #555' } },
      React.createElement('h5', { style: { margin: '5px 0', fontSize: '12px', color: '#999' } }, 'ON EXIT (runs once when leaving state)'),
      renderBlockList(state.onExit || [], 'onExit'),
      React.createElement(BlockPalette, { onAddBlock: (block) => addBlock(block, 'onExit') })
    ),
    editingBlock && React.createElement(React.Fragment, null,
      React.createElement('div', {
        onClick: () => setEditingBlock(null),
        style: {
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0,0,0,0.5)',
          zIndex: 999
        }
      }),
      React.createElement(ParameterEditor, {
        block: editingBlock.block,
        onChange: (params) => updateBlock(editingBlock.block.id, editingBlock.phase, params),
        onClose: () => setEditingBlock(null)
      })
    )
  );
}

// ============= DSL CODE VIEWER =============
function DSLCodeViewer({ states }) {
  const generateDSL = () => {
    let code = '// Generated Behavior DSL\n\n';

    for (const state of states) {
      code += `STATE ${state.name} {\n`;

      // MOVEMENT (continuous)
      if (state.movement && state.movement.length > 0) {
        code += `  // Movement behaviors (run continuously)\n`;
        for (const block of state.movement) {
          code += `  MOVEMENT: ${generateBlockCall(block)}\n`;
        }
      }

      // ATTACK (continuous)
      if (state.attack && state.attack.length > 0) {
        code += `  // Attack behaviors (run continuously)\n`;
        for (const block of state.attack) {
          code += `  ATTACK: ${generateBlockCall(block)}\n`;
        }
      }

      // UTILITY (continuous)
      if (state.utility && state.utility.length > 0) {
        code += `  // Utility behaviors (run continuously)\n`;
        for (const block of state.utility) {
          code += `  UTILITY: ${generateBlockCall(block)}\n`;
        }
      }

      // ON_ENTER (once)
      if (state.onEnter && state.onEnter.length > 0) {
        code += `  // Entry behaviors (run once)\n`;
        for (const block of state.onEnter) {
          code += `  ON_ENTER: ${generateBlockCall(block)}\n`;
        }
      }

      // ON_EXIT (once)
      if (state.onExit && state.onExit.length > 0) {
        code += `  // Exit behaviors (run once)\n`;
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

  return React.createElement('div', {
    style: { padding: '15px', background: '#1e1e1e', color: '#d4d4d4', borderRadius: '8px', fontFamily: 'monospace', fontSize: '12px', whiteSpace: 'pre-wrap', maxHeight: '400px', overflowY: 'auto' }
  }, generateDSL());
}

// ============= SPRITE CANVAS RENDERER =============
function SpriteCanvas({ spriteName, size = 64 }) {
  const canvasRef = React.useRef(null);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !window.spriteDB) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, size, size);

    // Extract sprite name from "atlas:name" format if needed
    const nameOnly = spriteName.includes(':') ? spriteName.split(':')[1] : spriteName;

    // Get sprite data
    const sprite = window.spriteDB.getSprite(nameOnly);
    if (sprite && sprite.image) {
      // Draw sprite centered and scaled to fit
      const scale = Math.min(size / sprite.width, size / sprite.height) * 0.9;
      const drawW = sprite.width * scale;
      const drawH = sprite.height * scale;
      const drawX = (size - drawW) / 2;
      const drawY = (size - drawH) / 2;

      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(
        sprite.image,
        sprite.x, sprite.y, sprite.width, sprite.height,
        drawX, drawY, drawW, drawH
      );
    } else {
      // Fallback: draw placeholder
      ctx.fillStyle = '#666';
      ctx.font = `${size * 0.6}px Arial`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('?', size / 2, size / 2);
    }
  }, [spriteName, size]);

  return React.createElement('canvas', {
    ref: canvasRef,
    width: size,
    height: size,
    style: {
      width: `${size}px`,
      height: `${size}px`,
      imageRendering: 'pixelated'
    }
  });
}

// ============= SPRITE PICKER =============
function SpritePicker({ selectedSprite, onSelectSprite, onClose }) {
  const [filter, setFilter] = React.useState('enemies');
  const [searchTerm, setSearchTerm] = React.useState('');

  // Get available sprites from sprite database
  const getAvailableSprites = () => {
    if (!window.spriteDB || !window.getSpritesByGroup) return [];

    let sprites = [];
    if (filter === 'all') {
      sprites = window.getAllSprites ? window.getAllSprites() : [];
    } else {
      sprites = window.getSpritesByGroup(filter);
    }

    // Filter by search term
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      sprites = sprites.filter(s => s.toLowerCase().includes(term));
    }

    return sprites.sort();
  };

  const availableSprites = getAvailableSprites();

  return React.createElement('div', {
    style: {
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0,0,0,0.8)',
      zIndex: 9999,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px'
    },
    onClick: onClose
  },
    React.createElement('div', {
      style: {
        background: '#1a1a1a',
        borderRadius: '12px',
        border: '2px solid #4CAF50',
        maxWidth: '800px',
        maxHeight: '80vh',
        width: '100%',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column'
      },
      onClick: (e) => e.stopPropagation()
    },
      React.createElement('div', {
        style: {
          padding: '20px',
          borderBottom: '2px solid #4CAF50'
        }
      },
        React.createElement('div', {
          style: {
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '15px'
          }
        },
          React.createElement('h3', {
            style: { margin: 0, color: '#4CAF50', fontSize: '18px' }
          }, 'Choose Sprite'),
          React.createElement('button', {
            onClick: onClose,
            style: {
              background: '#FF5722',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              padding: '8px 16px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }
          }, 'Close')
        ),
        React.createElement('input', {
          type: 'text',
          value: searchTerm,
          onChange: (e) => setSearchTerm(e.target.value),
          placeholder: 'Search sprites...',
          style: {
            width: '100%',
            padding: '8px',
            fontSize: '14px',
            borderRadius: '4px',
            border: '1px solid #444',
            background: '#0d0d0d',
            color: 'white',
            marginBottom: '10px'
          }
        }),
        React.createElement('div', {
          style: {
            display: 'flex',
            gap: '8px',
            flexWrap: 'wrap'
          }
        },
          ['enemies', 'objects', 'tiles', 'misc', 'all'].map(category =>
            React.createElement('button', {
              key: category,
              onClick: () => {
                setFilter(category);
                setSearchTerm('');
              },
              style: {
                padding: '6px 12px',
                background: filter === category ? '#4CAF50' : '#2a2a2a',
                color: 'white',
                border: filter === category ? '2px solid #81C784' : '2px solid #444',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '12px',
                fontWeight: filter === category ? 'bold' : 'normal',
                textTransform: 'capitalize'
              }
            }, category)
          )
        )
      ),
      React.createElement('div', {
        style: {
          padding: '10px 20px',
          background: '#2a2a2a',
          borderBottom: '1px solid #444',
          fontSize: '12px',
          color: '#aaa'
        }
      }, `${availableSprites.length} sprites found`),
      React.createElement('div', {
        style: {
          padding: '20px',
          overflowY: 'auto',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))',
          gap: '12px'
        }
      },
        availableSprites.map(spriteName =>
          React.createElement('div', {
            key: spriteName,
            onClick: () => {
              // Get the sprite to find its atlas
              const sprite = window.spriteDB.getSprite(spriteName);
              const fullName = sprite ? `${sprite.atlasName}:${spriteName}` : spriteName;
              onSelectSprite(fullName);
              onClose();
            },
            style: {
              padding: '12px',
              background: selectedSprite && selectedSprite.includes(spriteName) ? '#4CAF50' : '#2a2a2a',
              border: `2px solid ${selectedSprite && selectedSprite.includes(spriteName) ? '#81C784' : '#444'}`,
              borderRadius: '8px',
              cursor: 'pointer',
              textAlign: 'center',
              transition: 'all 0.2s',
              ':hover': {
                background: '#333',
                borderColor: '#4CAF50'
              }
            },
            onMouseEnter: (e) => {
              e.currentTarget.style.background = '#333';
              e.currentTarget.style.borderColor = '#4CAF50';
            },
            onMouseLeave: (e) => {
              const isSelected = selectedSprite && selectedSprite.includes(spriteName);
              e.currentTarget.style.background = isSelected ? '#4CAF50' : '#2a2a2a';
              e.currentTarget.style.borderColor = isSelected ? '#81C784' : '#444';
            }
          },
            React.createElement('div', {
              style: {
                width: '64px',
                height: '64px',
                margin: '0 auto 8px',
                background: '#0d0d0d',
                borderRadius: '4px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }
            }, React.createElement(SpriteCanvas, { spriteName, size: 60 })),
            React.createElement('div', {
              style: {
                fontSize: '11px',
                color: 'white',
                wordBreak: 'break-word',
                fontWeight: selectedSprite === `chars:${spriteName}` ? 'bold' : 'normal'
              }
            }, spriteName)
          )
        )
      )
    )
  );
}

// ============= ENEMY STATS PANEL =============
function EnemyStatsPanel({ enemy, onUpdate, onSave }) {
  const [showSpritePicker, setShowSpritePicker] = React.useState(false);

  if (!enemy) return null;

  const updateField = (field, value) => {
    onUpdate({ ...enemy, [field]: value });
  };

  const updateAttackField = (field, value) => {
    onUpdate({
      ...enemy,
      attack: { ...enemy.attack, [field]: value }
    });
  };

  return React.createElement('div', {
    style: {
      width: '350px',
      background: '#2a2a2a',
      borderRight: '1px solid #444',
      overflowY: 'auto',
      padding: '15px',
      color: 'white'
    }
  },
    React.createElement('h3', {
      style: { margin: '0 0 15px 0', color: '#FF5722', borderBottom: '2px solid #FF5722', paddingBottom: '8px' }
    }, 'Enemy Stats & Attacks'),
    React.createElement('div', {
      style: { marginBottom: '20px', background: '#333', padding: '12px', borderRadius: '6px' }
    },
      React.createElement('label', {
        style: { display: 'block', fontSize: '11px', fontWeight: 'bold', marginBottom: '6px', color: '#aaa', textTransform: 'uppercase' }
      }, 'Enemy Sprite'),
      React.createElement('div', {
        style: { padding: '12px', background: '#1a1a1a', borderRadius: '4px', textAlign: 'center' }
      },
        React.createElement('div', {
          style: {
            width: '96px',
            height: '96px',
            margin: '0 auto 12px',
            background: '#0d0d0d',
            borderRadius: '8px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            border: '2px solid #444'
          }
        }, enemy.sprite ? React.createElement(SpriteCanvas, {
          spriteName: enemy.sprite,
          size: 88
        }) : React.createElement('div', { style: { fontSize: '48px', color: '#666' } }, '?')),
        React.createElement('div', {
          style: { fontSize: '13px', color: '#4CAF50', marginBottom: '12px', fontWeight: 'bold' }
        }, enemy.sprite || 'No sprite selected'),
        React.createElement('button', {
          onClick: () => setShowSpritePicker(true),
          style: {
            width: '100%',
            padding: '10px',
            background: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold',
            fontSize: '13px',
            marginBottom: '8px'
          }
        }, '🎨 Choose Sprite'),
        React.createElement('input', {
          type: 'text',
          value: enemy.sprite || '',
          onChange: (e) => updateField('sprite', e.target.value),
          placeholder: 'Or type: chars:name',
          style: {
            width: '100%',
            padding: '6px',
            fontSize: '11px',
            borderRadius: '4px',
            border: '1px solid #555',
            background: '#0d0d0d',
            color: '#888',
            textAlign: 'center'
          }
        })
      )
    ),
    showSpritePicker && React.createElement(SpritePicker, {
      selectedSprite: enemy.sprite,
      onSelectSprite: (sprite) => updateField('sprite', sprite),
      onClose: () => setShowSpritePicker(false)
    }),
    React.createElement('div', {
      style: { marginBottom: '20px', background: '#333', padding: '12px', borderRadius: '6px' }
    },
      React.createElement('h4', {
        style: { margin: '0 0 10px 0', fontSize: '13px', color: '#FF9800' }
      }, 'Basic Info'),
      React.createElement('label', {
        style: { display: 'block', fontSize: '11px', fontWeight: 'bold', marginBottom: '4px', color: '#aaa' }
      }, 'Enemy ID (unique identifier)'),
      React.createElement('input', {
        type: 'text',
        value: enemy.id || '',
        onChange: (e) => updateField('id', e.target.value),
        placeholder: 'my_custom_boss',
        style: {
          width: '100%',
          padding: '6px',
          fontSize: '13px',
          borderRadius: '4px',
          border: '1px solid #555',
          background: '#1a1a1a',
          color: 'white',
          marginBottom: '8px'
        }
      }),
      React.createElement('div', {
        style: { fontSize: '10px', color: '#888', marginBottom: '12px' }
      }, 'Use lowercase, underscores only. This is what you use in spawn config!'),
      React.createElement('label', {
        style: { display: 'block', fontSize: '11px', fontWeight: 'bold', marginBottom: '4px', color: '#aaa' }
      }, 'Display Name'),
      React.createElement('input', {
        type: 'text',
        value: enemy.name || '',
        onChange: (e) => updateField('name', e.target.value),
        placeholder: 'My Custom Boss',
        style: {
          width: '100%',
          padding: '6px',
          fontSize: '13px',
          borderRadius: '4px',
          border: '1px solid #555',
          background: '#1a1a1a',
          color: 'white',
          marginBottom: '8px'
        }
      }),
      React.createElement('label', {
        style: { display: 'block', fontSize: '11px', fontWeight: 'bold', marginBottom: '4px', color: '#aaa' }
      }, 'HP'),
      React.createElement('input', {
        type: 'number',
        value: enemy.hp || 0,
        onChange: (e) => updateField('hp', parseInt(e.target.value) || 0),
        style: {
          width: '100%',
          padding: '6px',
          fontSize: '13px',
          borderRadius: '4px',
          border: '1px solid #555',
          background: '#1a1a1a',
          color: 'white',
          marginBottom: '8px'
        }
      }),
      React.createElement('label', {
        style: { display: 'block', fontSize: '11px', fontWeight: 'bold', marginBottom: '4px', color: '#aaa' }
      }, 'Speed'),
      React.createElement('input', {
        type: 'number',
        step: '0.1',
        value: enemy.speed || 0,
        onChange: (e) => updateField('speed', parseFloat(e.target.value) || 0),
        style: {
          width: '100%',
          padding: '6px',
          fontSize: '13px',
          borderRadius: '4px',
          border: '1px solid #555',
          background: '#1a1a1a',
          color: 'white'
        }
      })
    ),
    enemy.attack && React.createElement('div', {
      style: { marginBottom: '20px', background: '#333', padding: '12px', borderRadius: '6px' }
    },
      React.createElement('h4', {
        style: { margin: '0 0 10px 0', fontSize: '13px', color: '#FF5722' }
      }, 'Attack Configuration'),
      React.createElement('label', {
        style: { display: 'block', fontSize: '11px', fontWeight: 'bold', marginBottom: '4px', color: '#aaa' }
      }, 'Bullet ID'),
      React.createElement('input', {
        type: 'text',
        value: enemy.attack.bulletId || '',
        onChange: (e) => updateAttackField('bulletId', e.target.value),
        style: {
          width: '100%',
          padding: '6px',
          fontSize: '13px',
          borderRadius: '4px',
          border: '1px solid #555',
          background: '#1a1a1a',
          color: 'white',
          marginBottom: '8px'
        }
      }),
      React.createElement('label', {
        style: { display: 'block', fontSize: '11px', fontWeight: 'bold', marginBottom: '4px', color: '#aaa' }
      }, 'Cooldown (ms)'),
      React.createElement('input', {
        type: 'number',
        value: enemy.attack.cooldown || 0,
        onChange: (e) => updateAttackField('cooldown', parseInt(e.target.value) || 0),
        style: {
          width: '100%',
          padding: '6px',
          fontSize: '13px',
          borderRadius: '4px',
          border: '1px solid #555',
          background: '#1a1a1a',
          color: 'white',
          marginBottom: '8px'
        }
      }),
      React.createElement('label', {
        style: { display: 'block', fontSize: '11px', fontWeight: 'bold', marginBottom: '4px', color: '#aaa' }
      }, 'Speed'),
      React.createElement('input', {
        type: 'number',
        step: '0.1',
        value: enemy.attack.speed || 0,
        onChange: (e) => updateAttackField('speed', parseFloat(e.target.value) || 0),
        style: {
          width: '100%',
          padding: '6px',
          fontSize: '13px',
          borderRadius: '4px',
          border: '1px solid #555',
          background: '#1a1a1a',
          color: 'white',
          marginBottom: '8px'
        }
      }),
      React.createElement('label', {
        style: { display: 'block', fontSize: '11px', fontWeight: 'bold', marginBottom: '4px', color: '#aaa' }
      }, 'Lifetime (ms)'),
      React.createElement('input', {
        type: 'number',
        value: enemy.attack.lifetime || 0,
        onChange: (e) => updateAttackField('lifetime', parseInt(e.target.value) || 0),
        style: {
          width: '100%',
          padding: '6px',
          fontSize: '13px',
          borderRadius: '4px',
          border: '1px solid #555',
          background: '#1a1a1a',
          color: 'white',
          marginBottom: '8px'
        }
      }),
      React.createElement('label', {
        style: { display: 'block', fontSize: '11px', fontWeight: 'bold', marginBottom: '4px', color: '#aaa' }
      }, 'Count'),
      React.createElement('input', {
        type: 'number',
        value: enemy.attack.count || 1,
        onChange: (e) => updateAttackField('count', parseInt(e.target.value) || 1),
        style: {
          width: '100%',
          padding: '6px',
          fontSize: '13px',
          borderRadius: '4px',
          border: '1px solid #555',
          background: '#1a1a1a',
          color: 'white',
          marginBottom: '8px'
        }
      }),
      React.createElement('label', {
        style: { display: 'block', fontSize: '11px', fontWeight: 'bold', marginBottom: '4px', color: '#aaa' }
      }, 'Spread (degrees)'),
      React.createElement('input', {
        type: 'number',
        value: enemy.attack.spread || 0,
        onChange: (e) => updateAttackField('spread', parseFloat(e.target.value) || 0),
        style: {
          width: '100%',
          padding: '6px',
          fontSize: '13px',
          borderRadius: '4px',
          border: '1px solid #555',
          background: '#1a1a1a',
          color: 'white',
          marginBottom: '8px'
        }
      }),
      enemy.attack.inaccuracy !== undefined && React.createElement(React.Fragment, null,
        React.createElement('label', {
          style: { display: 'block', fontSize: '11px', fontWeight: 'bold', marginBottom: '4px', color: '#aaa' }
        }, 'Inaccuracy (degrees)'),
        React.createElement('input', {
          type: 'number',
          value: enemy.attack.inaccuracy || 0,
          onChange: (e) => updateAttackField('inaccuracy', parseFloat(e.target.value) || 0),
          style: {
            width: '100%',
            padding: '6px',
            fontSize: '13px',
            borderRadius: '4px',
            border: '1px solid #555',
            background: '#1a1a1a',
            color: 'white'
          }
        })
      )
    ),
    React.createElement('button', {
      onClick: onSave,
      style: {
        width: '100%',
        padding: '12px',
        background: '#4CAF50',
        color: 'white',
        border: 'none',
        borderRadius: '4px',
        cursor: 'pointer',
        fontWeight: 'bold',
        fontSize: '14px'
      }
    }, '\u{1F4BE} Save Enemy to Backend')
  );
}

// ============= MAIN BEHAVIOR DESIGNER =============
function BehaviorDesigner() {
  const [states, setStates] = React.useState([]);
  const [selectedState, setSelectedState] = React.useState(null);
  const [behaviorName, setBehaviorName] = React.useState('MyBehavior');
  const [showCode, setShowCode] = React.useState(false);
  const [showTestArena, setShowTestArena] = React.useState(true);
  const [designerMode, setDesignerMode] = React.useState('state-machine'); // 'state-machine' or 'behaviors'

  // Enemy loading
  const [enemies, setEnemies] = React.useState([]);
  const [selectedEnemy, setSelectedEnemy] = React.useState(null);
  const [showEnemyPanel, setShowEnemyPanel] = React.useState(false);

  const addState = () => {
    const newState = {
      id: `state_${Date.now()}`,
      name: `state_${states.length + 1}`,
      x: 200 + (states.length * 50),
      y: 250,
      movement: [],
      attack: [],
      utility: [],
      onEnter: [],
      onExit: [],
      duration: null,
      next: null
    };
    setStates([...states, newState]);
    setSelectedState(newState);
    setShowEnemyPanel(false); // Hide Enemy Panel when selecting a state
  };

  const handleSelectState = (state) => {
    setSelectedState(state);
    setShowEnemyPanel(false); // Automatically hide Enemy Stats when clicking a state
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

  // Load enemies from backend on mount
  React.useEffect(() => {
    loadEnemies();
  }, []);

  const loadEnemies = async () => {
    try {
      const response = await fetch('/api/enemy-editor/list');
      const result = await response.json();
      if (result.success) {
        // Filter out charging_shooter and bullet definitions
        const filteredEnemies = result.enemies.filter(e =>
          e.id !== 'charging_shooter' &&
          e.id !== 'charging_shooter_bullet' &&
          !e.id.endsWith('_bullet')
        );
        setEnemies(filteredEnemies);
        console.log('[BehaviorDesigner] Loaded', filteredEnemies.length, 'enemies');
      }
    } catch (err) {
      console.error('[BehaviorDesigner] Failed to load enemies:', err);
    }
  };

  const generateDefaultStates = (enemy) => {
    // Generate state machine from enemy attack data and AI type
    const aiType = enemy.ai?.behaviorTree || 'BasicChaseAndShoot';
    const attack = enemy.attack || {};

    console.log('[BehaviorDesigner] Generating default states for', enemy.id, 'with AI:', aiType);

    // Create shoot block from attack data
    const shootBlock = {
      id: `Shoot_${Date.now()}`,
      name: 'Shoot',
      color: '#F44336',
      params: {
        angle: 'GetAngle()',
        speed: attack.speed || 25,
        bulletType: attack.bulletId || 'arrow'
      }
    };

    // If multi-shot, use ShootSpread
    if (attack.count > 1) {
      const halfSpread = (attack.spread || 45) / 2;
      const angles = [];
      for (let i = 0; i < attack.count; i++) {
        const angle = -halfSpread + (i * (attack.spread / (attack.count - 1)));
        angles.push(angle);
      }
      shootBlock.id = `ShootSpread_${Date.now()}`;
      shootBlock.name = 'Shoot Spread';
      shootBlock.params = {
        angles: angles,
        speed: attack.speed || 25,
        bulletType: attack.bulletId || 'arrow'
      };
    }

    // Generate states based on AI type
    let states = [];

    if (aiType === 'BasicChaseAndShoot') {
      // Goblin/Orc: Chase AND shoot simultaneously
      states = [{
        id: `state_${Date.now()}`,
        name: 'chase_and_shoot',
        x: 200,
        y: 250,
        // New category-based structure
        movement: [
          {
            id: `MoveToward_${Date.now()}`,
            name: 'Move Toward',
            color: '#4CAF50',
            params: { target: 'ToPlayer', speed: enemy.speed / 10 || 1.0 }
          }
        ],
        attack: [
          shootBlock
        ],
        utility: [],
        onEnter: [],
        onExit: [],
        duration: null,
        next: null
      }];
    } else if (aiType === 'RedDemonBT') {
      // Red Demon: Orbit while shooting spread
      states = [{
        id: `state_${Date.now()}`,
        name: 'combat',
        x: 200,
        y: 250,
        // New category-based structure
        movement: [
          {
            id: `Orbit_${Date.now()}`,
            name: 'Orbit',
            color: '#4CAF50',
            params: { target: 'ToPlayer', radius: 250, speed: 0.8, clockwise: true }
          }
        ],
        attack: [
          shootBlock
        ],
        utility: [],
        onEnter: [],
        onExit: [],
        duration: null,
        next: null
      }];
    } else if (aiType === 'ChargingShooter') {
      // Charging Shooter: Alternates between charging and shooting
      states = [
        {
          id: `state_${Date.now()}_1`,
          name: 'charge',
          x: 150,
          y: 250,
          movement: [
            {
              id: `Charge_${Date.now()}`,
              name: 'Charge',
              color: '#4CAF50',
              params: { target: 'ToPlayer', speed: 2.0, duration: 1000 }
            }
          ],
          attack: [],  // No shooting while charging
          utility: [],
          onEnter: [],
          onExit: [],
          duration: 2000,
          next: 'shoot'
        },
        {
          id: `state_${Date.now()}_2`,
          name: 'shoot',
          x: 350,
          y: 250,
          movement: [
            {
              id: `Stop_${Date.now()}`,
              name: 'Stop',
              color: '#4CAF50',
              params: {}
            }
          ],
          attack: [shootBlock],
          utility: [],
          onEnter: [],
          onExit: [],
          duration: 1000,
          next: 'charge'
        }
      ];
    } else if (aiType === 'StaticBoss') {
      // Static Boss: Stationary while shooting
      states = [{
        id: `state_${Date.now()}`,
        name: 'stationary_shoot',
        x: 200,
        y: 250,
        movement: [
          {
            id: `Stop_${Date.now()}`,
            name: 'Stop',
            color: '#4CAF50',
            params: {}
          }
        ],
        attack: [shootBlock],
        utility: [],
        onEnter: [],
        onExit: [],
        duration: null,
        next: null
      }];
    } else {
      // Default: simple chase and shoot simultaneously
      states = [{
        id: `state_${Date.now()}`,
        name: 'default_behavior',
        x: 200,
        y: 250,
        movement: [
          {
            id: `MoveToward_${Date.now()}`,
            name: 'Move Toward',
            color: '#4CAF50',
            params: { target: 'ToPlayer', speed: 1.0 }
          }
        ],
        attack: [shootBlock],
        utility: [],
        onEnter: [],
        onExit: [],
        duration: null,
        next: null
      }];
    }

    return states;
  };

  const loadEnemy = (enemy) => {
    setSelectedEnemy(enemy);
    setBehaviorName(enemy.name);

    // Load or generate state machine
    if (enemy.behavior && enemy.behavior.states) {
      // Enemy has state machine data
      console.log('[BehaviorDesigner] Loading existing states for', enemy.id);
      setStates(enemy.behavior.states);
    } else {
      // Generate default state machine from attack data
      console.log('[BehaviorDesigner] Generating default states for', enemy.id);
      const defaultStates = generateDefaultStates(enemy);
      setStates(defaultStates);

      // Add generated states to enemy object
      enemy.behavior = { states: defaultStates };
    }

    setSelectedState(null);
    setShowEnemyPanel(true);
  };

  const saveEnemy = async () => {
    if (!selectedEnemy) return;

    // Add current states to enemy's behavior
    const enemyToSave = {
      ...selectedEnemy,
      behavior: {
        states: states
      }
    };

    try {
      const response = await fetch('/api/enemy-editor/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enemy: enemyToSave })
      });

      const result = await response.json();
      if (result.success) {
        alert(`Enemy "${selectedEnemy.name}" saved to backend with state machine!`);
        await loadEnemies(); // Reload list
      } else {
        alert('Failed to save: ' + result.error);
      }
    } catch (err) {
      alert('Failed to save: ' + err.message);
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

  return React.createElement('div', {
    style: { width: '100%', height: '100vh', display: 'flex', flexDirection: 'column' }
  },
    React.createElement('div', {
      style: { padding: '15px', background: '#333', color: 'white', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }
    },
      React.createElement('div', {
        style: { display: 'flex', alignItems: 'center', gap: '15px' }
      },
        React.createElement('h2', { style: { margin: 0 } }, 'Enemy Editor'),
        React.createElement('div', { style: { display: 'flex', gap: '10px', alignItems: 'center' } },
          React.createElement('select', {
            onChange: (e) => {
              const enemy = enemies.find(en => en.id === e.target.value);
              if (enemy) loadEnemy(enemy);
            },
            value: selectedEnemy?.id || '',
            style: {
              flex: 1,
              padding: '8px',
              fontSize: '14px',
              borderRadius: '4px',
              border: 'none',
              background: '#4CAF50',
              color: 'white',
              fontWeight: 'bold',
              cursor: 'pointer'
            }
          },
            React.createElement('option', { value: '' }, 'Select Enemy...'),
            enemies.map(enemy =>
              React.createElement('option', {
                key: enemy.id,
                value: enemy.id
              }, enemy.name + ' (' + enemy.id + ')')
            )
          ),
          React.createElement('button', {
            onClick: () => {
              const newEnemy = {
                id: `custom_enemy_${Date.now()}`,
                name: 'New Enemy',
                sprite: 'chars:goblin',
                hp: 100,
                speed: 10,
                width: 1,
                height: 1,
                renderScale: 2,
                attack: {
                  bulletId: 'arrow',
                  cooldown: 1000,
                  speed: 25,
                  lifetime: 2000,
                  count: 1,
                  spread: 0
                }
              };
              loadEnemy(newEnemy);
            },
            style: {
              padding: '8px 16px',
              background: '#2196F3',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold',
              fontSize: '14px',
              whiteSpace: 'nowrap'
            }
          }, '+ New Enemy')
        ),
        React.createElement('input', {
          type: 'text',
          value: behaviorName,
          onChange: (e) => setBehaviorName(e.target.value),
          style: { padding: '8px', fontSize: '14px', borderRadius: '4px', border: 'none' },
          placeholder: 'Behavior name'
        })
      ),
      React.createElement('div', {
        style: { display: 'flex', gap: '10px' }
      },
        selectedEnemy && React.createElement('button', {
          onClick: saveEnemy,
          style: {
            padding: '10px 20px',
            background: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }
        }, 'Save Enemy'),
        React.createElement('button', {
          onClick: () => setShowEnemyPanel(!showEnemyPanel),
          style: {
            padding: '10px 20px',
            background: selectedEnemy ? '#FF5722' : '#666',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: selectedEnemy ? 'pointer' : 'not-allowed',
            fontWeight: 'bold'
          },
          disabled: !selectedEnemy
        }, (showEnemyPanel ? 'Hide' : 'Show') + ' Enemy Stats'),
        React.createElement('button', {
          onClick: addState,
          style: {
            padding: '10px 20px',
            background: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }
        }, '+ Add State'),
        React.createElement('button', {
          onClick: () => setShowTestArena(!showTestArena),
          style: {
            padding: '10px 20px',
            background: '#9C27B0',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }
        }, (showTestArena ? 'Hide' : 'Show') + ' Test Arena'),
        React.createElement('button', {
          onClick: () => setShowCode(!showCode),
          style: {
            padding: '10px 20px',
            background: '#2196F3',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }
        }, (showCode ? 'Hide' : 'Show') + ' Code'),
        React.createElement('button', {
          onClick: saveBehavior,
          style: {
            padding: '10px 20px',
            background: '#FF9800',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }
        }, 'Save Behavior')
      )
    ),
    React.createElement('div', {
      style: { flex: 1, display: 'flex', overflow: 'hidden' }
    },
      showEnemyPanel && selectedEnemy ?
        React.createElement(EnemyStatsPanel, {
          enemy: selectedEnemy,
          onUpdate: setSelectedEnemy,
          onSave: saveEnemy
        })
      :
        React.createElement('div', {
          style: { width: '350px', background: '#2a2a2a', borderRight: '1px solid #444', overflowY: 'auto', display: 'flex', flexDirection: 'column' }
        },
          selectedState ?
            React.createElement('div', {
              style: { padding: '15px', color: 'white' }
            },
              React.createElement('h3', {
                style: { margin: '0 0 15px 0', color: '#64B5F6', borderBottom: '2px solid #64B5F6', paddingBottom: '8px' }
              }, 'Inspector'),
              React.createElement('div', {
                style: { marginBottom: '20px', background: '#333', padding: '12px', borderRadius: '6px' }
              },
                React.createElement('label', {
                  style: { display: 'block', fontSize: '11px', fontWeight: 'bold', marginBottom: '6px', color: '#aaa', textTransform: 'uppercase' }
                }, 'State Name'),
                React.createElement('input', {
                  type: 'text',
                  value: selectedState.name,
                  onChange: (e) => updateState({ ...selectedState, name: e.target.value }),
                  style: {
                    width: '100%',
                    padding: '8px',
                    fontSize: '14px',
                    borderRadius: '4px',
                    border: '1px solid #555',
                    background: '#1a1a1a',
                    color: 'white'
                  }
                })
              ),
              React.createElement('div', {
                style: { marginBottom: '20px', background: '#333', padding: '12px', borderRadius: '6px' }
              },
                React.createElement('label', {
                  style: { display: 'block', fontSize: '11px', fontWeight: 'bold', marginBottom: '6px', color: '#aaa', textTransform: 'uppercase' }
                }, 'Duration (ms)'),
                React.createElement('input', {
                  type: 'number',
                  value: selectedState.duration || '',
                  onChange: (e) => updateState({ ...selectedState, duration: parseInt(e.target.value) || null }),
                  placeholder: 'Auto transition time',
                  style: {
                    width: '100%',
                    padding: '8px',
                    fontSize: '14px',
                    borderRadius: '4px',
                    border: '1px solid #555',
                    background: '#1a1a1a',
                    color: 'white'
                  }
                })
              ),
              React.createElement('div', {
                style: { marginBottom: '20px', background: '#333', padding: '12px', borderRadius: '6px' }
              },
                React.createElement('label', {
                  style: { display: 'block', fontSize: '11px', fontWeight: 'bold', marginBottom: '6px', color: '#aaa', textTransform: 'uppercase' }
                }, 'Next State'),
                React.createElement('select', {
                  value: selectedState.next || '',
                  onChange: (e) => updateState({ ...selectedState, next: e.target.value || null }),
                  style: {
                    width: '100%',
                    padding: '8px',
                    fontSize: '14px',
                    borderRadius: '4px',
                    border: '1px solid #555',
                    background: '#1a1a1a',
                    color: 'white'
                  }
                },
                  React.createElement('option', { value: '' }, 'None'),
                  states.filter(s => s.id !== selectedState.id).map(s =>
                    React.createElement('option', {
                      key: s.id,
                      value: s.name
                    }, s.name)
                  )
                )
              ),
              React.createElement('div', {
                style: { borderTop: '1px solid #444', paddingTop: '15px' }
              },
                React.createElement(StateEditor, {
                  state: selectedState,
                  onUpdate: updateState,
                  onDelete: deleteState,
                  selected: true,
                  onSelect: setSelectedState
                })
              )
            )
          :
            React.createElement('div', {
              style: { padding: '20px', color: '#888', textAlign: 'center' }
            },
              React.createElement('h3', {
                style: { color: '#64B5F6', marginBottom: '10px' }
              }, 'Inspector'),
              React.createElement('p', null, 'Select a state to edit its properties'),
              React.createElement('div', {
                style: { marginTop: '20px', padding: '15px', background: '#333', borderRadius: '6px', textAlign: 'left', fontSize: '13px' }
              },
                React.createElement('strong', {
                  style: { color: '#4CAF50' }
                }, 'Quick Start:'),
                React.createElement('ul', {
                  style: { marginTop: '10px', paddingLeft: '20px', lineHeight: '1.8' }
                },
                  React.createElement('li', null, 'Click "Add State" button above'),
                  React.createElement('li', null, 'Click & drag states to move'),
                  React.createElement('li', null, 'Shift+click to connect states'),
                  React.createElement('li', null, 'Click state to edit behaviors')
                )
              )
            )
        ),
      React.createElement('div', {
        style: { flex: 1, overflowY: 'auto', padding: '10px', background: '#1a1a1a', display: 'flex', flexDirection: 'column' }
      },
        // Designer mode tabs
        React.createElement('div', {
          style: {
            display: 'flex',
            gap: '5px',
            marginBottom: '10px',
            borderBottom: '2px solid #333',
            paddingBottom: '5px'
          }
        },
          React.createElement('button', {
            onClick: () => setDesignerMode('state-machine'),
            style: {
              padding: '8px 16px',
              background: designerMode === 'state-machine' ? '#4CAF50' : '#2a2a2a',
              color: designerMode === 'state-machine' ? 'white' : '#888',
              border: 'none',
              borderRadius: '4px 4px 0 0',
              cursor: 'pointer',
              fontWeight: designerMode === 'state-machine' ? 'bold' : 'normal',
              fontSize: '13px'
            }
          }, '🔄 State Machine'),
          React.createElement('button', {
            onClick: () => setDesignerMode('behaviors'),
            style: {
              padding: '8px 16px',
              background: designerMode === 'behaviors' ? '#4CAF50' : '#2a2a2a',
              color: designerMode === 'behaviors' ? 'white' : '#888',
              border: 'none',
              borderRadius: '4px 4px 0 0',
              cursor: 'pointer',
              fontWeight: designerMode === 'behaviors' ? 'bold' : 'normal',
              fontSize: '13px'
            }
          }, '⚙️ Behavior Builder')
        ),

        // Render selected designer
        designerMode === 'state-machine' ?
          (window.VisualStateMachine ?
            React.createElement(window.VisualStateMachine, {
              states: states,
              onStatesChange: setStates,
              selectedState: selectedState,
              onSelectState: handleSelectState
            })
          :
            React.createElement('div', {
              style: { padding: '20px', color: '#999' }
            }, 'Loading state machine editor...'))
        : designerMode === 'behaviors' ?
          (window.BehaviorAtomDesigner ?
            React.createElement(window.BehaviorAtomDesigner, {
              onExport: (behaviorData) => console.log('Behaviors exported:', behaviorData)
            })
          :
            React.createElement('div', {
              style: { padding: '20px', color: '#999' }
            }, 'Loading behavior designer...'))
        :
          React.createElement('div', {
            style: { padding: '20px', color: '#999' }
          }, 'Unknown designer mode')
      ),
      (showTestArena || showCode) && React.createElement('div', {
        style: { width: '450px', background: '#2d2d2d', borderLeft: '1px solid #444', display: 'flex', flexDirection: 'column' }
      },
        showTestArena && window.BehaviorTestArena && React.createElement('div', {
          style: { flex: showCode ? '1' : 'auto', padding: '15px', overflowY: 'auto', borderBottom: showCode ? '1px solid #444' : 'none' }
        },
          React.createElement(window.BehaviorTestArena, {
            states: states,
            behaviorName: behaviorName
          })
        ),
        showCode && React.createElement('div', {
          style: { flex: showTestArena ? '1' : 'auto', padding: '15px', overflowY: 'auto' }
        },
          React.createElement('h3', {
            style: { color: 'white', marginTop: 0 }
          }, 'Generated DSL Code'),
          React.createElement(DSLCodeViewer, {
            states: states
          })
        )
      )
    )
  );
}

// Export for use in other files
window.BehaviorDesigner = BehaviorDesigner;

// Render if this is the main page
if (document.getElementById('behavior-designer-root')) {
  ReactDOM.createRoot(document.getElementById('behavior-designer-root')).render(React.createElement(BehaviorDesigner));
}
