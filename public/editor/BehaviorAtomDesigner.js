/**
 * Behavior Atom Designer - Target-based behavior editor
 *
 * Concepts:
 * - TARGETS: Define reference points (player, nearest enemy, specific tile)
 * - MOVEMENT: Define how to move relative to targets (approach, orbit, flee, zigzag)
 * - ATTACK: Define shooting behaviors (shoot at target, patterns, aimed)
 * - CONDITIONS: When behaviors activate (distance, health, timers)
 */

console.log('[BehaviorAtomDesigner] Module loading...');

// ============= BEHAVIOR ATOM DESIGNER =============
function BehaviorAtomDesigner({ onExport }) {
  // Behavior state
  const [movementBehaviors, setMovementBehaviors] = React.useState([]);
  const [attackBehaviors, setAttackBehaviors] = React.useState([]);
  const [selectedBehaviorId, setSelectedBehaviorId] = React.useState(null);
  const [nextBehaviorId, setNextBehaviorId] = React.useState(1);

  // Canvas for visualization
  const canvasRef = React.useRef(null);
  const CANVAS_WIDTH = 800;
  const CANVAS_HEIGHT = 600;

  // Available targets
  const TARGET_TYPES = [
    { id: 'player', name: 'Player', color: '#2196F3', icon: '🎯' },
    { id: 'nearest_enemy', name: 'Nearest Enemy', color: '#F44336', icon: '👾' },
    { id: 'fixed_position', name: 'Fixed Position', color: '#4CAF50', icon: '📍' },
    { id: 'spawn_point', name: 'Spawn Point', color: '#FF9800', icon: '🏠' }
  ];

  // Movement patterns
  const MOVEMENT_PATTERNS = [
    { id: 'move_toward', name: 'Move Toward', icon: '→', description: 'Move directly toward target' },
    { id: 'orbit', name: 'Orbit', icon: '⭯', description: 'Circle around target at set radius' },
    { id: 'flee', name: 'Flee', icon: '←', description: 'Move away from target' },
    { id: 'zigzag', name: 'Zig Zag', icon: '⚡', description: 'Zigzag pattern toward target' },
    { id: 'strafe', name: 'Strafe', icon: '↔', description: 'Move perpendicular to target' },
    { id: 'wander', name: 'Wander', icon: '∿', description: 'Random wandering movement' },
    { id: 'charge', name: 'Charge', icon: '⚡→', description: 'Fast direct charge' },
    { id: 'maintain_distance', name: 'Keep Distance', icon: '↹', description: 'Stay at specific range' }
  ];

  // Attack patterns
  const ATTACK_PATTERNS = [
    { id: 'single_shot', name: 'Single Shot', icon: '•', description: 'Fire single bullet at target' },
    { id: 'spread', name: 'Spread', icon: '•••', description: 'Fire multiple bullets in spread' },
    { id: 'ring', name: 'Ring', icon: '⊚', description: 'Fire bullets in all directions' },
    { id: 'spiral', name: 'Spiral', icon: '🌀', description: 'Rotating spiral pattern' },
    { id: 'aimed', name: 'Aimed', icon: '🎯', description: 'Predicted aim at target' }
  ];

  // Draw visualization
  React.useEffect(() => {
    drawCanvas();
  }, [movementBehaviors, attackBehaviors, selectedBehaviorId]);

  const drawCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    // Draw background
    ctx.fillStyle = '#0d0d0d';
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    // Draw grid
    ctx.strokeStyle = '#1a1a1a';
    ctx.lineWidth = 1;
    for (let x = 0; x < CANVAS_WIDTH; x += 50) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, CANVAS_HEIGHT);
      ctx.stroke();
    }
    for (let y = 0; y < CANVAS_HEIGHT; y += 50) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(CANVAS_WIDTH, y);
      ctx.stroke();
    }

    // Draw enemy position (center)
    const enemyX = CANVAS_WIDTH / 2;
    const enemyY = CANVAS_HEIGHT / 2;

    ctx.fillStyle = '#FF5722';
    ctx.beginPath();
    ctx.arc(enemyX, enemyY, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'white';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Enemy', enemyX, enemyY + 25);

    // Draw movement behaviors
    movementBehaviors.forEach((behavior, index) => {
      const y = 100 + (index * 60);
      drawMovementBehavior(ctx, behavior, enemyX, y);
    });

    // Draw attack behaviors
    attackBehaviors.forEach((behavior, index) => {
      const y = 100 + (index * 60);
      const x = CANVAS_WIDTH - 200;
      drawAttackBehavior(ctx, behavior, x, y);
    });
  };

  const drawMovementBehavior = (ctx, behavior, enemyX, y) => {
    const isSelected = selectedBehaviorId === behavior.id;
    const targetType = TARGET_TYPES.find(t => t.id === behavior.target);
    const pattern = MOVEMENT_PATTERNS.find(p => p.id === behavior.pattern);

    // Box
    ctx.fillStyle = isSelected ? '#4CAF50' : '#2a2a2a';
    ctx.fillRect(20, y, 250, 50);
    ctx.strokeStyle = isSelected ? '#81C784' : '#444';
    ctx.lineWidth = 2;
    ctx.strokeRect(20, y, 250, 50);

    // Icon and text
    ctx.fillStyle = 'white';
    ctx.font = '20px Arial';
    ctx.textAlign = 'left';
    ctx.fillText(pattern?.icon || '?', 30, y + 30);

    ctx.font = 'bold 14px Arial';
    ctx.fillText(pattern?.name || 'Unknown', 60, y + 22);

    ctx.font = '11px Arial';
    ctx.fillStyle = targetType?.color || '#888';
    ctx.fillText(`Target: ${targetType?.icon} ${targetType?.name}`, 60, y + 38);

    // Draw visualization line to enemy
    ctx.strokeStyle = targetType?.color || '#888';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(270, y + 25);
    ctx.lineTo(enemyX - 15, CANVAS_HEIGHT / 2);
    ctx.stroke();
    ctx.setLineDash([]);
  };

  const drawAttackBehavior = (ctx, behavior, x, y) => {
    const isSelected = selectedBehaviorId === behavior.id;
    const targetType = TARGET_TYPES.find(t => t.id === behavior.target);
    const pattern = ATTACK_PATTERNS.find(p => p.id === behavior.pattern);

    // Box
    ctx.fillStyle = isSelected ? '#F44336' : '#2a2a2a';
    ctx.fillRect(x, y, 250, 50);
    ctx.strokeStyle = isSelected ? '#E57373' : '#444';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, 250, 50);

    // Icon and text
    ctx.fillStyle = 'white';
    ctx.font = '20px Arial';
    ctx.textAlign = 'left';
    ctx.fillText(pattern?.icon || '?', x + 10, y + 30);

    ctx.font = 'bold 14px Arial';
    ctx.fillText(pattern?.name || 'Unknown', x + 40, y + 22);

    ctx.font = '11px Arial';
    ctx.fillStyle = targetType?.color || '#888';
    ctx.fillText(`Target: ${targetType?.icon} ${targetType?.name}`, x + 40, y + 38);
  };

  const addMovementBehavior = () => {
    const newBehavior = {
      id: `move_${nextBehaviorId}`,
      type: 'movement',
      pattern: 'move_toward',
      target: 'player',
      params: {
        speed: 1.0,
        radius: 200,
        minDistance: 50,
        maxDistance: 400
      }
    };
    setMovementBehaviors([...movementBehaviors, newBehavior]);
    setNextBehaviorId(nextBehaviorId + 1);
    setSelectedBehaviorId(newBehavior.id);
  };

  const addAttackBehavior = () => {
    const newBehavior = {
      id: `attack_${nextBehaviorId}`,
      type: 'attack',
      pattern: 'single_shot',
      target: 'player',
      params: {
        cooldown: 1000,
        bulletSpeed: 25,
        count: 1,
        spread: 30,
        range: 300
      }
    };
    setAttackBehaviors([...attackBehaviors, newBehavior]);
    setNextBehaviorId(nextBehaviorId + 1);
    setSelectedBehaviorId(newBehavior.id);
  };

  const updateBehavior = (id, updates) => {
    setMovementBehaviors(movementBehaviors.map(b =>
      b.id === id ? { ...b, ...updates } : b
    ));
    setAttackBehaviors(attackBehaviors.map(b =>
      b.id === id ? { ...b, ...updates } : b
    ));
  };

  const deleteBehavior = (id) => {
    setMovementBehaviors(movementBehaviors.filter(b => b.id !== id));
    setAttackBehaviors(attackBehaviors.filter(b => b.id !== id));
    if (selectedBehaviorId === id) {
      setSelectedBehaviorId(null);
    }
  };

  const exportBehaviors = () => {
    const data = {
      movement: movementBehaviors,
      attack: attackBehaviors
    };

    console.log('Exported behaviors:', data);
    if (onExport) {
      onExport(data);
    }

    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    alert('Behaviors copied to clipboard!');
  };

  const selectedBehavior =
    movementBehaviors.find(b => b.id === selectedBehaviorId) ||
    attackBehaviors.find(b => b.id === selectedBehaviorId);

  return React.createElement('div', {
    style: {
      display: 'flex',
      gap: '10px',
      height: '100%',
      background: '#1a1a1a',
      padding: '10px',
      borderRadius: '8px'
    }
  },
    // Left panel - Canvas
    React.createElement('div', {
      style: {
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        gap: '10px'
      }
    },
      // Toolbar
      React.createElement('div', {
        style: {
          padding: '10px',
          background: '#2a2a2a',
          borderRadius: '6px',
          display: 'flex',
          gap: '10px',
          alignItems: 'center',
          flexWrap: 'wrap'
        }
      },
        React.createElement('div', { style: { color: '#64B5F6', fontWeight: 'bold', fontSize: '14px' } }, 'Target-Based Behavior Designer'),
        React.createElement('button', {
          onClick: addMovementBehavior,
          style: {
            padding: '8px 16px',
            background: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }
        }, '+ Movement'),
        React.createElement('button', {
          onClick: addAttackBehavior,
          style: {
            padding: '8px 16px',
            background: '#F44336',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }
        }, '+ Attack'),
        React.createElement('button', {
          onClick: exportBehaviors,
          disabled: movementBehaviors.length === 0 && attackBehaviors.length === 0,
          style: {
            padding: '8px 16px',
            background: (movementBehaviors.length === 0 && attackBehaviors.length === 0) ? '#666' : '#2196F3',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: (movementBehaviors.length === 0 && attackBehaviors.length === 0) ? 'not-allowed' : 'pointer',
            marginLeft: 'auto'
          }
        }, 'Export')
      ),

      // Canvas
      React.createElement('canvas', {
        ref: canvasRef,
        width: CANVAS_WIDTH,
        height: CANVAS_HEIGHT,
        onClick: (e) => {
          // TODO: Click behaviors to select
        },
        style: {
          border: '2px solid #444',
          borderRadius: '4px',
          background: '#0d0d0d'
        }
      }),

      // Instructions
      React.createElement('div', {
        style: {
          padding: '10px',
          background: '#2a2a2a',
          borderRadius: '6px',
          fontSize: '11px',
          color: '#888',
          lineHeight: '1.6'
        }
      },
        React.createElement('div', null, '• Add movement/attack behaviors using buttons above'),
        React.createElement('div', null, '• Movement defines how enemy moves relative to targets'),
        React.createElement('div', null, '• Attack defines when and how enemy shoots'),
        React.createElement('div', null, '• All behaviors run simultaneously (not sequential)')
      )
    ),

    // Right panel - Behavior editor
    React.createElement('div', {
      style: {
        width: '300px',
        background: '#2a2a2a',
        borderRadius: '6px',
        padding: '15px',
        display: 'flex',
        flexDirection: 'column',
        gap: '15px',
        overflowY: 'auto'
      }
    },
      selectedBehavior ?
        // Behavior editing panel
        React.createElement('div', { style: { display: 'flex', flexDirection: 'column', gap: '12px' } },
          React.createElement('div', { style: { color: '#64B5F6', fontWeight: 'bold', fontSize: '14px', marginBottom: '5px' } },
            selectedBehavior.type === 'movement' ? '🔄 Movement Behavior' : '🔫 Attack Behavior'
          ),

          // Pattern selector
          React.createElement('div', null,
            React.createElement('label', { style: { color: '#888', fontSize: '12px', display: 'block', marginBottom: '6px' } }, 'Pattern'),
            React.createElement('select', {
              value: selectedBehavior.pattern,
              onChange: (e) => updateBehavior(selectedBehavior.id, { pattern: e.target.value }),
              style: {
                width: '100%',
                padding: '8px',
                background: '#1a1a1a',
                color: 'white',
                border: '1px solid #444',
                borderRadius: '4px'
              }
            },
              (selectedBehavior.type === 'movement' ? MOVEMENT_PATTERNS : ATTACK_PATTERNS).map(p =>
                React.createElement('option', { key: p.id, value: p.id }, `${p.icon} ${p.name}`)
              )
            )
          ),

          // Target selector
          React.createElement('div', null,
            React.createElement('label', { style: { color: '#888', fontSize: '12px', display: 'block', marginBottom: '6px' } }, 'Target'),
            React.createElement('select', {
              value: selectedBehavior.target,
              onChange: (e) => updateBehavior(selectedBehavior.id, { target: e.target.value }),
              style: {
                width: '100%',
                padding: '8px',
                background: '#1a1a1a',
                color: 'white',
                border: '1px solid #444',
                borderRadius: '4px'
              }
            },
              TARGET_TYPES.map(t =>
                React.createElement('option', { key: t.id, value: t.id }, `${t.icon} ${t.name}`)
              )
            )
          ),

          // Parameters (conditional based on pattern)
          React.createElement('div', { style: { color: '#888', fontSize: '12px', fontWeight: 'bold', marginTop: '10px' } }, 'Parameters'),

          // Speed (for movement)
          selectedBehavior.type === 'movement' && React.createElement('div', null,
            React.createElement('label', { style: { color: '#888', fontSize: '11px', display: 'block', marginBottom: '4px' } }, 'Speed'),
            React.createElement('input', {
              type: 'number',
              value: selectedBehavior.params.speed,
              onChange: (e) => updateBehavior(selectedBehavior.id, {
                params: { ...selectedBehavior.params, speed: parseFloat(e.target.value) }
              }),
              step: '0.1',
              min: '0.1',
              max: '5',
              style: {
                width: '100%',
                padding: '6px',
                background: '#1a1a1a',
                color: 'white',
                border: '1px solid #444',
                borderRadius: '4px'
              }
            })
          ),

          // Range (for attack)
          selectedBehavior.type === 'attack' && React.createElement('div', null,
            React.createElement('label', { style: { color: '#888', fontSize: '11px', display: 'block', marginBottom: '4px' } }, 'Range (pixels)'),
            React.createElement('input', {
              type: 'number',
              value: selectedBehavior.params.range,
              onChange: (e) => updateBehavior(selectedBehavior.id, {
                params: { ...selectedBehavior.params, range: parseInt(e.target.value) }
              }),
              step: '50',
              min: '50',
              max: '800',
              style: {
                width: '100%',
                padding: '6px',
                background: '#1a1a1a',
                color: 'white',
                border: '1px solid #444',
                borderRadius: '4px'
              }
            })
          ),

          // Cooldown (for attack)
          selectedBehavior.type === 'attack' && React.createElement('div', null,
            React.createElement('label', { style: { color: '#888', fontSize: '11px', display: 'block', marginBottom: '4px' } }, 'Cooldown (ms)'),
            React.createElement('input', {
              type: 'number',
              value: selectedBehavior.params.cooldown,
              onChange: (e) => updateBehavior(selectedBehavior.id, {
                params: { ...selectedBehavior.params, cooldown: parseInt(e.target.value) }
              }),
              step: '100',
              min: '100',
              max: '5000',
              style: {
                width: '100%',
                padding: '6px',
                background: '#1a1a1a',
                color: 'white',
                border: '1px solid #444',
                borderRadius: '4px'
              }
            })
          ),

          // Delete button
          React.createElement('button', {
            onClick: () => deleteBehavior(selectedBehavior.id),
            style: {
              padding: '10px',
              background: '#f44336',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold',
              marginTop: 'auto'
            }
          }, 'Delete Behavior')
        )
      :
        // No selection placeholder
        React.createElement('div', { style: { color: '#666', fontSize: '13px', textAlign: 'center', marginTop: '40px' } },
          React.createElement('div', { style: { fontSize: '48px', marginBottom: '15px' } }, '⚙️'),
          React.createElement('div', null, 'Add or select a behavior to configure'),
          React.createElement('div', { style: { fontSize: '11px', marginTop: '15px', color: '#444', lineHeight: '1.6' } },
            React.createElement('div', null, `${movementBehaviors.length} movement`),
            React.createElement('div', null, `${attackBehaviors.length} attack`)
          )
        )
    )
  );
}

window.BehaviorAtomDesigner = BehaviorAtomDesigner;
console.log('[BehaviorAtomDesigner] Module loaded successfully');
