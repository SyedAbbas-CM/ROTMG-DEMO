/**
 * Waypoint Designer - Visual movement pattern creator
 *
 * Features:
 * - Click to place waypoints on test arena
 * - Configure waypoint types (static, track player, random)
 * - Add actions at waypoints (shoot, wait, speed change)
 * - Sequential path execution with loop option
 * - Live test preview
 * - Export to simple JSON format
 */

console.log('[WaypointDesigner] Module loading...');

// ============= WAYPOINT DESIGNER =============
function WaypointDesigner({ waypoints: waypointsProp, onWaypointsChange, onExport }) {
  const canvasRef = React.useRef(null);
  const waypoints = waypointsProp || [];
  const setWaypoints = onWaypointsChange || (() => {});
  const [selectedWaypointId, setSelectedWaypointId] = React.useState(null);
  const [nextWaypointId, setNextWaypointId] = React.useState(1);
  const [isPlacingMode, setIsPlacingMode] = React.useState(false);
  const [loopPath, setLoopPath] = React.useState(true);
  const [testEnemy, setTestEnemy] = React.useState(null);

  const CANVAS_WIDTH = 800;
  const CANVAS_HEIGHT = 600;
  const WAYPOINT_RADIUS = 12;

  React.useEffect(() => {
    drawCanvas();
  }, [waypoints, selectedWaypointId, testEnemy]);

  const drawCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    // Clear
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

    // Draw path lines connecting waypoints
    if (waypoints.length > 1) {
      ctx.strokeStyle = '#4CAF50';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);

      for (let i = 0; i < waypoints.length - 1; i++) {
        const wp1 = waypoints[i];
        const wp2 = waypoints[i + 1];

        ctx.beginPath();
        ctx.moveTo(wp1.pos.x, wp1.pos.y);
        ctx.lineTo(wp2.pos.x, wp2.pos.y);
        ctx.stroke();

        // Draw arrow
        drawArrow(ctx, wp1.pos.x, wp1.pos.y, wp2.pos.x, wp2.pos.y);
      }

      // Draw loop connection if enabled
      if (loopPath && waypoints.length > 2) {
        const first = waypoints[0];
        const last = waypoints[waypoints.length - 1];

        ctx.strokeStyle = '#FF9800';
        ctx.beginPath();
        ctx.moveTo(last.pos.x, last.pos.y);
        ctx.lineTo(first.pos.x, first.pos.y);
        ctx.stroke();
        drawArrow(ctx, last.pos.x, last.pos.y, first.pos.x, first.pos.y);
      }

      ctx.setLineDash([]);
    }

    // Draw waypoints
    waypoints.forEach((wp, index) => {
      const isSelected = selectedWaypointId === wp.id;
      const isFirst = index === 0;

      // Waypoint circle
      ctx.fillStyle = getWaypointColor(wp.type);
      ctx.beginPath();
      ctx.arc(wp.pos.x, wp.pos.y, WAYPOINT_RADIUS, 0, Math.PI * 2);
      ctx.fill();

      // Border
      ctx.strokeStyle = isSelected ? '#FFD700' : '#fff';
      ctx.lineWidth = isSelected ? 4 : 2;
      ctx.stroke();

      // Number label
      ctx.fillStyle = isFirst ? '#000' : '#fff';
      ctx.font = 'bold 14px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText((index + 1).toString(), wp.pos.x, wp.pos.y);

      // Action indicators
      if (wp.actions && wp.actions.length > 0) {
        ctx.fillStyle = '#F44336';
        ctx.beginPath();
        ctx.arc(wp.pos.x + WAYPOINT_RADIUS - 3, wp.pos.y - WAYPOINT_RADIUS + 3, 5, 0, Math.PI * 2);
        ctx.fill();
      }

      // First waypoint marker
      if (isFirst) {
        ctx.fillStyle = '#4CAF50';
        ctx.font = 'bold 10px Arial';
        ctx.fillText('START', wp.pos.x, wp.pos.y + WAYPOINT_RADIUS + 12);
      }
    });

    // Draw test enemy if active
    if (testEnemy) {
      ctx.fillStyle = '#FF5722';
      ctx.beginPath();
      ctx.arc(testEnemy.x, testEnemy.y, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  };

  const drawArrow = (ctx, x1, y1, x2, y2) => {
    const headlen = 10;
    const angle = Math.atan2(y2 - y1, x2 - x1);

    // Arrow head
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - headlen * Math.cos(angle - Math.PI / 6), y2 - headlen * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(x2 - headlen * Math.cos(angle + Math.PI / 6), y2 - headlen * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  };

  const getWaypointColor = (type) => {
    switch (type) {
      case 'static': return '#4CAF50';
      case 'track_player': return '#2196F3';
      case 'random': return '#9C27B0';
      default: return '#4CAF50';
    }
  };

  const handleCanvasClick = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (isPlacingMode) {
      // Place new waypoint
      const newWaypoint = {
        id: nextWaypointId,
        type: 'static',
        pos: { x, y },
        actions: []
      };
      setWaypoints([...waypoints, newWaypoint]);
      setNextWaypointId(nextWaypointId + 1);
      setSelectedWaypointId(newWaypoint.id);
      setIsPlacingMode(false);
    } else {
      // Check if clicking on existing waypoint
      const clickedWaypoint = waypoints.find(wp => {
        const dx = wp.pos.x - x;
        const dy = wp.pos.y - y;
        return Math.sqrt(dx * dx + dy * dy) <= WAYPOINT_RADIUS;
      });

      if (clickedWaypoint) {
        setSelectedWaypointId(clickedWaypoint.id);
      } else {
        setSelectedWaypointId(null);
      }
    }
  };

  const updateWaypoint = (id, updates) => {
    setWaypoints(waypoints.map(wp =>
      wp.id === id ? { ...wp, ...updates } : wp
    ));
  };

  const deleteWaypoint = (id) => {
    setWaypoints(waypoints.filter(wp => wp.id !== id));
    if (selectedWaypointId === id) {
      setSelectedWaypointId(null);
    }
  };

  const addAction = (waypointId, action) => {
    setWaypoints(waypoints.map(wp =>
      wp.id === waypointId ? { ...wp, actions: [...wp.actions, action] } : wp
    ));
  };

  const removeAction = (waypointId, actionIndex) => {
    setWaypoints(waypoints.map(wp =>
      wp.id === waypointId ? { ...wp, actions: wp.actions.filter((_, i) => i !== actionIndex) } : wp
    ));
  };

  const exportPath = () => {
    const pathData = {
      waypoints: waypoints.map(wp => ({
        id: wp.id,
        type: wp.type,
        pos: wp.pos,
        actions: wp.actions
      })),
      loop: loopPath
    };

    console.log('Exported path:', pathData);
    if (onExport) {
      onExport(pathData);
    }

    // Copy to clipboard
    navigator.clipboard.writeText(JSON.stringify(pathData, null, 2));
    alert('Path data copied to clipboard!');
  };

  const selectedWaypoint = waypoints.find(wp => wp.id === selectedWaypointId);

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
        React.createElement('div', { style: { color: '#64B5F6', fontWeight: 'bold', fontSize: '14px' } }, 'Movement Path Designer'),
        React.createElement('button', {
          onClick: () => setIsPlacingMode(!isPlacingMode),
          style: {
            padding: '8px 16px',
            background: isPlacingMode ? '#4CAF50' : '#444',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }
        }, isPlacingMode ? '✓ Click to Place' : '+ Add Waypoint'),
        React.createElement('label', {
          style: {
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            color: '#888',
            cursor: 'pointer'
          }
        },
          React.createElement('input', {
            type: 'checkbox',
            checked: loopPath,
            onChange: (e) => setLoopPath(e.target.checked)
          }),
          'Loop Path'
        ),
        React.createElement('button', {
          onClick: () => setWaypoints([]),
          style: {
            padding: '8px 16px',
            background: '#f44336',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }
        }, 'Clear All'),
        React.createElement('button', {
          onClick: exportPath,
          disabled: waypoints.length === 0,
          style: {
            padding: '8px 16px',
            background: waypoints.length === 0 ? '#666' : '#2196F3',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: waypoints.length === 0 ? 'not-allowed' : 'pointer',
            marginLeft: 'auto'
          }
        }, 'Export Path')
      ),

      // Canvas
      React.createElement('canvas', {
        ref: canvasRef,
        width: CANVAS_WIDTH,
        height: CANVAS_HEIGHT,
        onClick: handleCanvasClick,
        style: {
          border: '2px solid #444',
          borderRadius: '4px',
          cursor: isPlacingMode ? 'crosshair' : 'pointer',
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
        React.createElement('div', null, '• Click "Add Waypoint" then click on canvas to place'),
        React.createElement('div', null, '• Click waypoint to select and edit'),
        React.createElement('div', null, '• Waypoints are visited in order (1 → 2 → 3...)'),
        React.createElement('div', null, '• Add actions like "Shoot" or "Wait" at each waypoint')
      )
    ),

    // Right panel - Waypoint editor
    React.createElement('div', {
      style: {
        width: '280px',
        background: '#2a2a2a',
        borderRadius: '6px',
        padding: '15px',
        display: 'flex',
        flexDirection: 'column',
        gap: '15px'
      }
    },
      selectedWaypoint ?
        // Waypoint editing panel
        React.createElement('div', { style: { display: 'flex', flexDirection: 'column', gap: '12px' } },
          React.createElement('div', { style: { color: '#64B5F6', fontWeight: 'bold', fontSize: '14px', marginBottom: '5px' } },
            `Waypoint #${waypoints.indexOf(selectedWaypoint) + 1}`
          ),

          // Type selector
          React.createElement('div', null,
            React.createElement('label', { style: { color: '#888', fontSize: '12px', display: 'block', marginBottom: '6px' } }, 'Type'),
            React.createElement('select', {
              value: selectedWaypoint.type,
              onChange: (e) => updateWaypoint(selectedWaypoint.id, { type: e.target.value }),
              style: {
                width: '100%',
                padding: '8px',
                background: '#1a1a1a',
                color: 'white',
                border: '1px solid #444',
                borderRadius: '4px'
              }
            },
              React.createElement('option', { value: 'static' }, 'Static Position'),
              React.createElement('option', { value: 'track_player' }, 'Track Player'),
              React.createElement('option', { value: 'random' }, 'Random in Area')
            )
          ),

          // Position display
          React.createElement('div', null,
            React.createElement('div', { style: { color: '#888', fontSize: '12px', marginBottom: '6px' } }, 'Position'),
            React.createElement('div', { style: { color: '#fff', fontSize: '12px', fontFamily: 'monospace' } },
              `x: ${selectedWaypoint.pos.x.toFixed(0)}, y: ${selectedWaypoint.pos.y.toFixed(0)}`
            )
          ),

          // Actions
          React.createElement('div', null,
            React.createElement('div', { style: { color: '#888', fontSize: '12px', marginBottom: '6px' } }, 'Actions at Waypoint'),
            React.createElement('div', { style: { display: 'flex', flexDirection: 'column', gap: '6px' } },
              selectedWaypoint.actions.map((action, index) =>
                React.createElement('div', {
                  key: index,
                  style: {
                    padding: '6px 8px',
                    background: '#1a1a1a',
                    borderRadius: '4px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    fontSize: '12px'
                  }
                },
                  React.createElement('span', { style: { color: '#fff' } }, `${action.type}${action.duration ? ` (${action.duration}ms)` : ''}`),
                  React.createElement('button', {
                    onClick: () => removeAction(selectedWaypoint.id, index),
                    style: {
                      padding: '2px 6px',
                      background: '#f44336',
                      color: 'white',
                      border: 'none',
                      borderRadius: '3px',
                      cursor: 'pointer',
                      fontSize: '10px'
                    }
                  }, '×')
                )
              )
            ),

            // Add action buttons
            React.createElement('div', { style: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', marginTop: '8px' } },
              React.createElement('button', {
                onClick: () => addAction(selectedWaypoint.id, { type: 'shoot', target: 'player' }),
                style: {
                  padding: '6px',
                  background: '#F44336',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '11px'
                }
              }, '+ Shoot'),
              React.createElement('button', {
                onClick: () => addAction(selectedWaypoint.id, { type: 'wait', duration: 1000 }),
                style: {
                  padding: '6px',
                  background: '#9C27B0',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '11px'
                }
              }, '+ Wait 1s')
            )
          ),

          // Delete button
          React.createElement('button', {
            onClick: () => deleteWaypoint(selectedWaypoint.id),
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
          }, 'Delete Waypoint')
        )
      :
        // No selection placeholder
        React.createElement('div', { style: { color: '#666', fontSize: '13px', textAlign: 'center', marginTop: '40px' } },
          React.createElement('div', { style: { fontSize: '48px', marginBottom: '15px' } }, '📍'),
          React.createElement('div', null, 'Click a waypoint to edit'),
          React.createElement('div', { style: { fontSize: '11px', marginTop: '10px', color: '#444' } },
            `${waypoints.length} waypoint${waypoints.length !== 1 ? 's' : ''} placed`
          )
        )
    )
  );
}

window.WaypointDesigner = WaypointDesigner;
console.log('[WaypointDesigner] Module loaded successfully');
