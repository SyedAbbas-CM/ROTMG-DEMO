/**
 * Visual State Machine Editor - Graph-based state machine designer
 *
 * Features:
 * - Draggable state nodes
 * - Visual connections between states
 * - Click to add states
 * - Right-click to connect states
 * - Click state to edit behaviors
 */

console.log('[VisualStateMachine] Module loading...');

// ============= VISUAL STATE MACHINE =============
function VisualStateMachine({ states, onStatesChange, selectedState, onSelectState }) {
  const canvasRef = React.useRef(null);
  const [draggingNodeId, setDraggingNodeId] = React.useState(null);  // Store ID instead of reference
  const [connectingFromId, setConnectingFromId] = React.useState(null);  // Store ID instead of reference
  const [mousePos, setMousePos] = React.useState({ x: 0, y: 0 });
  const [offset, setOffset] = React.useState({ x: 0, y: 0 });

  const NODE_WIDTH = 120;
  const NODE_HEIGHT = 80;

  // Get node by ID (helper function)
  const getNodeById = (id) => states.find(s => s.id === id);

  React.useEffect(() => {
    drawCanvas();
  }, [states, selectedState, connectingFromId, mousePos, draggingNodeId]);

  const drawCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();

    // Clear
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = '#2a2a2a';
    ctx.lineWidth = 1;
    for (let x = 0; x < canvas.width; x += 50) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += 50) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    // Draw connections
    ctx.strokeStyle = '#4CAF50';
    ctx.lineWidth = 2;
    states.forEach(state => {
      if (state.next) {
        const targetState = states.find(s => s.name === state.next);
        if (targetState) {
          drawArrow(ctx,
            state.x + NODE_WIDTH / 2,
            state.y + NODE_HEIGHT / 2,
            targetState.x + NODE_WIDTH / 2,
            targetState.y + NODE_HEIGHT / 2
          );
        }
      }
    });

    // Draw connecting line (while dragging)
    if (connectingFromId) {
      const connectingFrom = getNodeById(connectingFromId);
      if (connectingFrom) {
        ctx.strokeStyle = '#FF9800';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(connectingFrom.x + NODE_WIDTH / 2, connectingFrom.y + NODE_HEIGHT / 2);
        ctx.lineTo(mousePos.x, mousePos.y);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    // Draw nodes
    states.forEach(state => {
      const isSelected = selectedState && selectedState.id === state.id;
      const isInitial = states[0] === state;

      // Node background
      ctx.fillStyle = isSelected ? '#2196F3' : (isInitial ? '#4CAF50' : '#333');
      ctx.fillRect(state.x, state.y, NODE_WIDTH, NODE_HEIGHT);

      // Node border
      ctx.strokeStyle = isSelected ? '#64B5F6' : (isInitial ? '#81C784' : '#666');
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeRect(state.x, state.y, NODE_WIDTH, NODE_HEIGHT);

      // State name
      ctx.fillStyle = 'white';
      ctx.font = 'bold 14px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(state.name, state.x + NODE_WIDTH / 2, state.y + 25);

      // Category-based behavior count with color coding
      const moveCount = state.movement?.length || 0;
      const attackCount = state.attack?.length || 0;
      const utilityCount = state.utility?.length || 0;

      ctx.font = '11px Arial';
      ctx.textAlign = 'left';

      // Movement (green)
      if (moveCount > 0) {
        ctx.fillStyle = '#4CAF50';
        ctx.fillText(`\u25CF ${moveCount}`, state.x + 5, state.y + 45);
      }

      // Attack (red)
      if (attackCount > 0) {
        ctx.fillStyle = '#F44336';
        ctx.fillText(`\u25CF ${attackCount}`, state.x + 40, state.y + 45);
      }

      // Utility (blue)
      if (utilityCount > 0) {
        ctx.fillStyle = '#2196F3';
        ctx.fillText(`\u25CF ${utilityCount}`, state.x + 75, state.y + 45);
      }

      // Show "empty" if no behaviors
      if (moveCount === 0 && attackCount === 0 && utilityCount === 0) {
        ctx.fillStyle = '#666';
        ctx.textAlign = 'center';
        ctx.fillText('(empty)', state.x + NODE_WIDTH / 2, state.y + 45);
      }

      ctx.textAlign = 'center';

      // Duration
      if (state.duration) {
        ctx.fillText(`${state.duration}ms`, state.x + NODE_WIDTH / 2, state.y + 60);
      }

      // Initial marker
      if (isInitial) {
        ctx.fillStyle = '#4CAF50';
        ctx.font = 'bold 10px Arial';
        ctx.fillText('START', state.x + NODE_WIDTH / 2, state.y + 75);
      }
    });
  };

  const drawArrow = (ctx, x1, y1, x2, y2) => {
    const headlen = 10;
    const angle = Math.atan2(y2 - y1, x2 - x1);

    // Line
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    // Arrow head
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - headlen * Math.cos(angle - Math.PI / 6), y2 - headlen * Math.sin(angle - Math.PI / 6));
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - headlen * Math.cos(angle + Math.PI / 6), y2 - headlen * Math.sin(angle + Math.PI / 6));
    ctx.stroke();
  };

  const handleMouseDown = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Check if clicking on a node
    const clickedNode = states.find(state =>
      x >= state.x && x <= state.x + NODE_WIDTH &&
      y >= state.y && y <= state.y + NODE_HEIGHT
    );

    if (clickedNode) {
      if (e.shiftKey) {
        // Shift+click to connect
        if (connectingFromId) {
          // Complete connection
          const updatedStates = states.map(s =>
            s.id === connectingFromId ? { ...s, next: clickedNode.name } : s
          );
          onStatesChange(updatedStates);
          setConnectingFromId(null);
        } else {
          setConnectingFromId(clickedNode.id);
        }
      } else {
        // Normal click to select/drag
        onSelectState(clickedNode);
        setDraggingNodeId(clickedNode.id);
        setOffset({
          x: x - clickedNode.x,
          y: y - clickedNode.y
        });
      }
    } else {
      // Cancel connecting
      setConnectingFromId(null);
    }
  };

  const handleMouseMove = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setMousePos({ x, y });

    if (draggingNodeId) {
      const updatedStates = states.map(state =>
        state.id === draggingNodeId
          ? { ...state, x: x - offset.x, y: y - offset.y }
          : state
      );
      onStatesChange(updatedStates);
    }
  };

  const handleMouseUp = () => {
    setDraggingNodeId(null);
  };

  // Double-click disabled per user request
  // const handleDoubleClick = (e) => {
  //   // User prefers manual state creation via buttons
  // };

  return React.createElement('div', { style: { background: '#1a1a1a', borderRadius: '8px', padding: '10px', height: '100%', display: 'flex', flexDirection: 'column' } },
    React.createElement('div', { style: { marginBottom: '10px', padding: '10px', background: '#2a2a2a', borderRadius: '6px', border: '1px solid #444' } },
      React.createElement('div', { style: { color: '#64B5F6', fontSize: '13px', fontWeight: 'bold', marginBottom: '8px' } }, 'State Machine Graph'),
      React.createElement('div', { style: { color: '#888', fontSize: '11px', lineHeight: '1.6' } },
        React.createElement('div', null, React.createElement('strong', null, 'Click'), ' state to select & edit'),
        React.createElement('div', null, React.createElement('strong', null, 'Click & drag'), ' to move states'),
        React.createElement('div', null, React.createElement('strong', null, 'Shift+click'), ' two states to connect them')
      )
    ),
    React.createElement('canvas', {
      ref: canvasRef,
      width: 800,
      height: 500,
      onMouseDown: handleMouseDown,
      onMouseMove: handleMouseMove,
      onMouseUp: handleMouseUp,
      style: {
        border: '2px solid #444',
        borderRadius: '4px',
        cursor: draggingNodeId ? 'grabbing' : (connectingFromId ? 'crosshair' : 'grab'),
        display: 'block',
        background: '#0d0d0d'
      }
    }),
    connectingFromId && React.createElement('div', { style: { marginTop: '10px', padding: '10px', background: '#FF9800', borderRadius: '4px', color: 'black', fontWeight: 'bold' } },
      'Connecting from "', getNodeById(connectingFromId)?.name, '" - Shift+click another state to connect'
    )
  );
}

window.VisualStateMachine = VisualStateMachine;
console.log('[VisualStateMachine] Module loaded successfully');
