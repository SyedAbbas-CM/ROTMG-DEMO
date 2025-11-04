/**
 * Behavior Test Arena - Visual testing environment for enemy behaviors
 *
 * 8x8 grid with:
 * - Test enemy running the designed behavior
 * - Static test players
 * - Walls/obstacles
 * - Real-time behavior execution
 */

// Simple DSL parser and executor for testing
class SimpleDSLExecutor {
  constructor() {
    this.stateTimers = {};
    this.currentStates = {};
    this.stateData = {};
    this.debug = true; // Enable debug logging
    console.log('[BehaviorTestArena] SimpleDSLExecutor initialized');
  }

  execute(entity, states, target, obstacles, deltaTime) {
    if (!entity.behaviorId) {
      if (this.debug) console.log('[Execute] No behaviorId on entity');
      return;
    }

    if (!states || states.length === 0) {
      if (this.debug) console.log('[Execute] No states provided');
      return;
    }

    // Initialize state
    if (!this.currentStates[entity.id]) {
      const initialState = states[0]?.name || 'idle';
      this.currentStates[entity.id] = initialState;
      this.stateTimers[entity.id] = 0;
      this.stateData[entity.id] = {};

      if (this.debug) console.log(`[Execute] Initialized entity ${entity.id} to state: ${initialState}`);

      // Run onEnter
      const state = states.find(s => s.name === this.currentStates[entity.id]);
      if (state) {
        if (this.debug) console.log(`[Execute] Running onEnter for state: ${state.name}, blocks:`, state.onEnter.length);
        this.executeBlocks(entity, state.onEnter, target, obstacles, this.stateData[entity.id]);
      }
    }

    const currentStateName = this.currentStates[entity.id];
    const state = states.find(s => s.name === currentStateName);
    if (!state) return;

    // Execute tick
    this.executeBlocks(entity, state.onTick, target, obstacles, this.stateData[entity.id]);

    // Update timer
    this.stateTimers[entity.id] += deltaTime;

    // Check transition
    if (state.duration && this.stateTimers[entity.id] >= state.duration) {
      if (state.next) {
        this.transitionTo(entity, states, state.next, target, obstacles);
      }
    }
  }

  transitionTo(entity, states, newStateName, target, obstacles) {
    const oldState = states.find(s => s.name === this.currentStates[entity.id]);
    const newState = states.find(s => s.name === newStateName);

    if (!newState) return;

    // Run onExit
    if (oldState) {
      this.executeBlocks(entity, oldState.onExit, target, obstacles, this.stateData[entity.id]);
    }

    // Transition
    this.currentStates[entity.id] = newStateName;
    this.stateTimers[entity.id] = 0;
    this.stateData[entity.id] = {};

    // Run onEnter
    this.executeBlocks(entity, newState.onEnter, target, obstacles, this.stateData[entity.id]);
  }

  executeBlocks(entity, blocks, target, obstacles, stateData) {
    if (!blocks || !Array.isArray(blocks)) return;
    for (const block of blocks) {
      this.executeBlock(entity, block, target, obstacles, stateData);
    }
  }

  executeBlock(entity, block, target, obstacles, stateData) {
    const blockType = block.id.split('_')[0];
    const params = block.params;

    if (this.debug && Math.random() < 0.01) { // Log 1% of the time to avoid spam
      console.log(`[ExecuteBlock] Type: ${blockType}, Params:`, params);
    }

    switch (blockType) {
      case 'MoveToward':
        if (target) {
          const dx = target.x - entity.x;
          const dy = target.y - entity.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist > 0) {
            const speed = (params.speed || 1.0) * 2;
            entity.vx = (dx / dist) * speed;
            entity.vy = (dy / dist) * speed;
          }
        }
        break;

      case 'MoveAway':
        if (target) {
          const dx = entity.x - target.x;
          const dy = entity.y - target.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const minDist = params.distance || 300;
          if (dist < minDist && dist > 0) {
            const speed = (params.speed || 1.0) * 2;
            entity.vx = (dx / dist) * speed;
            entity.vy = (dy / dist) * speed;
          } else {
            entity.vx = 0;
            entity.vy = 0;
          }
        }
        break;

      case 'Orbit':
        if (target) {
          if (!stateData.orbitAngle) {
            const dx = entity.x - target.x;
            const dy = entity.y - target.y;
            stateData.orbitAngle = Math.atan2(dy, dx);
          }
          const clockwise = params.clockwise !== false;
          const speed = params.speed || 1.0;
          stateData.orbitAngle += (clockwise ? -1 : 1) * speed * 0.02;

          const radius = params.radius || 150;
          const targetX = target.x + Math.cos(stateData.orbitAngle) * radius;
          const targetY = target.y + Math.sin(stateData.orbitAngle) * radius;

          const dx = targetX - entity.x;
          const dy = targetY - entity.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist > 0) {
            entity.vx = (dx / dist) * 2;
            entity.vy = (dy / dist) * 2;
          }
        }
        break;

      case 'Circle':
        if (!stateData.circleAngle) {
          stateData.circleAngle = 0;
          stateData.centerX = entity.x;
          stateData.centerY = entity.y;
        }
        const speed = params.speed || 1.0;
        stateData.circleAngle += speed * 0.02;

        const radius = params.radius || 100;
        const targetX = stateData.centerX + Math.cos(stateData.circleAngle) * radius;
        const targetY = stateData.centerY + Math.sin(stateData.circleAngle) * radius;

        const dx = targetX - entity.x;
        const dy = targetY - entity.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > 0) {
          entity.vx = (dx / dist) * 2;
          entity.vy = (dy / dist) * 2;
        }
        break;

      case 'Wander':
        if (!stateData.wanderTime) {
          stateData.wanderTime = 0;
          stateData.wanderAngle = Math.random() * Math.PI * 2;
        }
        stateData.wanderTime += 16;
        const changeInterval = params.changeInterval || 1000;
        if (stateData.wanderTime >= changeInterval) {
          stateData.wanderTime = 0;
          stateData.wanderAngle = Math.random() * Math.PI * 2;
        }
        entity.vx = Math.cos(stateData.wanderAngle) * 1.5;
        entity.vy = Math.sin(stateData.wanderAngle) * 1.5;
        break;

      case 'Stop':
        entity.vx = 0;
        entity.vy = 0;
        break;

      case 'SetSpeed':
        entity.speed = params.value || 1.0;
        break;

      case 'Shoot':
      case 'ShootPattern':
      case 'ShootSpread':
      case 'ShootRing':
        if (target) {
          const angle = Math.atan2(target.y - entity.y, target.x - entity.x);
          entity.lastShot = Date.now();
          entity.shootAngle = angle;
          entity.shootPattern = blockType;
          entity.shootParams = params;
        }
        break;
    }
  }
}

function BehaviorTestArena({ states, behaviorName }) {
  console.log('[BehaviorTestArena] Component mounted/updated with states:', states?.length, 'behaviorName:', behaviorName);

  const canvasRef = React.useRef(null);
  const [isRunning, setIsRunning] = React.useState(false);
  const [testEnemy, setTestEnemy] = React.useState(null);
  const [testPlayers, setTestPlayers] = React.useState([]);
  const [walls, setWalls] = React.useState([]);
  const [bullets, setBullets] = React.useState([]);
  const executorRef = React.useRef(new SimpleDSLExecutor());
  const animationRef = React.useRef(null);
  const lastTimeRef = React.useRef(Date.now());

  const TILE_SIZE = 50;
  const GRID_SIZE = 8;
  const CANVAS_SIZE = TILE_SIZE * GRID_SIZE;

  // Initialize test environment
  React.useEffect(() => {
    console.log('[BehaviorTestArena] Initializing test environment');

    // Test enemy in center
    const enemy = {
      id: 'test_enemy',
      behaviorId: behaviorName,
      x: CANVAS_SIZE / 2,
      y: CANVAS_SIZE / 2,
      vx: 0,
      vy: 0,
      speed: 1.5,
      size: 20,
      color: '#f44336',
      hp: 100
    };
    console.log('[BehaviorTestArena] Created test enemy:', enemy);
    setTestEnemy(enemy);

    // Test players
    const players = [
      { id: 'player1', x: 100, y: 100, size: 20, color: '#4CAF50' }
    ];
    console.log('[BehaviorTestArena] Created test players:', players);
    setTestPlayers(players);

    // Walls
    const wallsData = [
      { x: 200, y: 150, width: 50, height: 50 },
      { x: 100, y: 300, width: 50, height: 50 },
      { x: 300, y: 250, width: 50, height: 50 }
    ];
    console.log('[BehaviorTestArena] Created walls:', wallsData);
    setWalls(wallsData);
  }, [behaviorName]);

  // Game loop
  React.useEffect(() => {
    if (!isRunning || !testEnemy) {
      if (isRunning) console.log('[GameLoop] Not starting - testEnemy:', testEnemy);
      return;
    }

    console.log('[GameLoop] Starting game loop with', states.length, 'states');

    const gameLoop = () => {
      const now = Date.now();
      const deltaTime = Math.min(now - lastTimeRef.current, 100);
      lastTimeRef.current = now;

      // Update enemy behavior
      const target = testPlayers[0];
      if (target && states.length > 0) {
        executorRef.current.execute(testEnemy, states, target, walls, deltaTime);
      }

      // Update enemy position
      const newEnemy = { ...testEnemy };
      newEnemy.x += newEnemy.vx;
      newEnemy.y += newEnemy.vy;

      // Boundary check
      newEnemy.x = Math.max(newEnemy.size, Math.min(CANVAS_SIZE - newEnemy.size, newEnemy.x));
      newEnemy.y = Math.max(newEnemy.size, Math.min(CANVAS_SIZE - newEnemy.size, newEnemy.y));

      // Wall collision
      for (const wall of walls) {
        if (
          newEnemy.x + newEnemy.size > wall.x &&
          newEnemy.x - newEnemy.size < wall.x + wall.width &&
          newEnemy.y + newEnemy.size > wall.y &&
          newEnemy.y - newEnemy.size < wall.y + wall.height
        ) {
          newEnemy.x = testEnemy.x;
          newEnemy.y = testEnemy.y;
          newEnemy.vx = 0;
          newEnemy.vy = 0;
        }
      }

      setTestEnemy(newEnemy);

      // Update bullets
      setBullets(prev => {
        const updated = prev.map(b => ({
          ...b,
          x: b.x + b.vx,
          y: b.y + b.vy,
          life: b.life - deltaTime
        })).filter(b =>
          b.life > 0 &&
          b.x > 0 && b.x < CANVAS_SIZE &&
          b.y > 0 && b.y < CANVAS_SIZE
        );

        // Create new bullets if enemy is shooting
        if (newEnemy.lastShot && now - newEnemy.lastShot < 100) {
          const angle = newEnemy.shootAngle;
          const speed = 3;

          if (newEnemy.shootPattern === 'ShootPattern') {
            const pattern = newEnemy.shootParams.pattern;
            if (pattern === 'triple_forward') {
              for (const offset of [-30, 0, 30]) {
                const a = angle + (offset * Math.PI / 180);
                updated.push({
                  x: newEnemy.x,
                  y: newEnemy.y,
                  vx: Math.cos(a) * speed,
                  vy: Math.sin(a) * speed,
                  life: 3000,
                  color: '#ff9800'
                });
              }
            } else if (pattern === 'ring') {
              for (let i = 0; i < 12; i++) {
                const a = (Math.PI * 2 * i) / 12;
                updated.push({
                  x: newEnemy.x,
                  y: newEnemy.y,
                  vx: Math.cos(a) * speed,
                  vy: Math.sin(a) * speed,
                  life: 3000,
                  color: '#ff9800'
                });
              }
            }
          } else if (newEnemy.shootPattern === 'ShootRing') {
            const count = newEnemy.shootParams.count || 12;
            for (let i = 0; i < count; i++) {
              const a = (Math.PI * 2 * i) / count;
              updated.push({
                x: newEnemy.x,
                y: newEnemy.y,
                vx: Math.cos(a) * speed,
                vy: Math.sin(a) * speed,
                life: 3000,
                color: '#ff9800'
              });
            }
          } else {
            updated.push({
              x: newEnemy.x,
              y: newEnemy.y,
              vx: Math.cos(angle) * speed,
              vy: Math.sin(angle) * speed,
              life: 3000,
              color: '#ff9800'
            });
          }

          newEnemy.lastShot = 0;
        }

        return updated;
      });

      // Render
      render();

      animationRef.current = requestAnimationFrame(gameLoop);
    };

    animationRef.current = requestAnimationFrame(gameLoop);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isRunning, testEnemy, testPlayers, walls, states]);

  const render = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    // Clear
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    // Draw grid
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    for (let i = 0; i <= GRID_SIZE; i++) {
      ctx.beginPath();
      ctx.moveTo(i * TILE_SIZE, 0);
      ctx.lineTo(i * TILE_SIZE, CANVAS_SIZE);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, i * TILE_SIZE);
      ctx.lineTo(CANVAS_SIZE, i * TILE_SIZE);
      ctx.stroke();
    }

    // Draw walls
    ctx.fillStyle = '#666';
    for (const wall of walls) {
      ctx.fillRect(wall.x, wall.y, wall.width, wall.height);
    }

    // Draw test players
    ctx.fillStyle = '#4CAF50';
    for (const player of testPlayers) {
      ctx.beginPath();
      ctx.arc(player.x, player.y, player.size, 0, Math.PI * 2);
      ctx.fill();

      // Label
      ctx.fillStyle = '#fff';
      ctx.font = '10px Arial';
      ctx.fillText('Player', player.x - 15, player.y - 25);
      ctx.fillStyle = '#4CAF50';
    }

    // Draw bullets
    ctx.fillStyle = '#ff9800';
    for (const bullet of bullets) {
      ctx.beginPath();
      ctx.arc(bullet.x, bullet.y, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw test enemy
    if (testEnemy) {
      ctx.fillStyle = testEnemy.color;
      ctx.beginPath();
      ctx.arc(testEnemy.x, testEnemy.y, testEnemy.size, 0, Math.PI * 2);
      ctx.fill();

      // Direction indicator
      if (testEnemy.vx !== 0 || testEnemy.vy !== 0) {
        const angle = Math.atan2(testEnemy.vy, testEnemy.vx);
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(testEnemy.x, testEnemy.y);
        ctx.lineTo(
          testEnemy.x + Math.cos(angle) * 25,
          testEnemy.y + Math.sin(angle) * 25
        );
        ctx.stroke();
      }

      // State label
      const currentState = executorRef.current.currentStates[testEnemy.id] || 'none';
      const timer = Math.floor((executorRef.current.stateTimers[testEnemy.id] || 0) / 1000);
      ctx.fillStyle = '#fff';
      ctx.font = '11px Arial';
      ctx.fillText(`Enemy: ${currentState} (${timer}s)`, testEnemy.x - 40, testEnemy.y - 30);
    }
  };

  const handleCanvasClick = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Move player to clicked position
    setTestPlayers([{ id: 'player1', x, y, size: 20, color: '#4CAF50' }]);
  };

  const reset = () => {
    setIsRunning(false);
    executorRef.current = new SimpleDSLExecutor();
    lastTimeRef.current = Date.now();
    setBullets([]);

    // Reset enemy position
    setTestEnemy(prev => ({
      ...prev,
      x: CANVAS_SIZE / 2,
      y: CANVAS_SIZE / 2,
      vx: 0,
      vy: 0
    }));

    // Reset player position
    setTestPlayers([{ id: 'player1', x: 100, y: 100, size: 20, color: '#4CAF50' }]);
  };

  return React.createElement('div', { style: { background: '#2d2d2d', borderRadius: '8px', padding: '15px' } },
    React.createElement('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' } },
      React.createElement('h3', { style: { margin: 0, color: 'white' } }, 'Test Arena (8x8)'),
      React.createElement('div', { style: { display: 'flex', gap: '10px' } },
        React.createElement('button', {
          onClick: () => setIsRunning(!isRunning),
          style: {
            padding: '8px 16px',
            background: isRunning ? '#f44336' : '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }
        }, isRunning ? 'Pause' : 'Start'),
        React.createElement('button', {
          onClick: reset,
          style: {
            padding: '8px 16px',
            background: '#2196F3',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }
        }, 'Reset')
      )
    ),
    React.createElement('p', { style: { color: '#aaa', fontSize: '12px', margin: '5px 0 10px 0' } },
      'Click to move the player. Watch the enemy execute your behavior!'
    ),
    React.createElement('canvas', {
      ref: canvasRef,
      width: CANVAS_SIZE,
      height: CANVAS_SIZE,
      onClick: handleCanvasClick,
      style: {
        border: '2px solid #444',
        borderRadius: '4px',
        cursor: 'crosshair',
        display: 'block',
        background: '#1a1a1a'
      }
    }),
    React.createElement('div', { style: { marginTop: '10px', color: '#aaa', fontSize: '11px' } },
      React.createElement('div', null, '🟢 Green = Test Player (click to move)'),
      React.createElement('div', null, '🔴 Red = Test Enemy (running behavior)'),
      React.createElement('div', null, '⬜ Gray = Walls (blocks movement)'),
      React.createElement('div', null, '🟠 Orange = Bullets')
    )
  );
}

window.BehaviorTestArena = BehaviorTestArena;
