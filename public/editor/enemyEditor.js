const { useState, useEffect } = React;
const { Draggable } = ReactDraggable;

// ============= LOCAL STORAGE PERSISTENCE =============
const STORAGE_PREFIX = 'enemyEditor_';
const CURRENT_ENEMY_KEY = 'enemyEditor_currentEnemy';
let autoSaveTimer = null;

function saveToLocalStorage(enemyData) {
  try {
    const enemyId = enemyData.id || 'untitled';
    localStorage.setItem(STORAGE_PREFIX + enemyId, JSON.stringify(enemyData));
    localStorage.setItem(CURRENT_ENEMY_KEY, enemyId);
    console.log(`Auto-saved enemy: ${enemyId}`);
  } catch (e) {
    console.warn('localStorage full - enemy not saved locally');
  }
}

function scheduleAutoSave(enemyData) {
  if (autoSaveTimer) clearTimeout(autoSaveTimer);
  autoSaveTimer = setTimeout(() => {
    saveToLocalStorage(enemyData);
  }, 1000);
}

function loadFromLocalStorage(enemyId) {
  try {
    const data = localStorage.getItem(STORAGE_PREFIX + enemyId);
    if (!data) return null;
    return JSON.parse(data);
  } catch (e) {
    return null;
  }
}

function loadLastEnemy() {
  const lastEnemyId = localStorage.getItem(CURRENT_ENEMY_KEY);
  if (lastEnemyId) {
    return loadFromLocalStorage(lastEnemyId);
  }
  return null;
}

function Palette({behaviours,onAdd}){
  return (
    <div>
      <h4>Behaviours</h4>
      {behaviours.map(b=> (
        <div key={b} style={{margin:'4px 0',cursor:'pointer'}} onClick={()=>onAdd(b)}>{b}</div>
      ))}
    </div>
  );
}

function Node({state,selected,onSelect,onPosChange}){
  return (
    <Draggable
      position={{x:state.x,y:state.y}}
      onStop={(e, data)=>onPosChange(state.id,data.x,data.y)}>
      <div className="state-node" onClick={()=>onSelect(state)}>{state.id}</div>
    </Draggable>
  );
}

function ParamEditor({behaviour,onChange}){
  const [json,setJson]=useState(JSON.stringify(behaviour.params||{},null,2));
  return (
    <div style={{position:'absolute',right:0,top:0,width:300,height:'100%',borderLeft:'1px solid #ccc',background:'#f9f9f9',padding:8}}>
      <h4>{behaviour.type} params</h4>
      <textarea style={{width:'100%',height:'80%'}} value={json} onChange={e=>setJson(e.target.value)}/>
      <button onClick={()=>{try{onChange(JSON.parse(json));}catch{alert('invalid json')}}}>Apply</button>
    </div>
  );
}

function App(){
  const [states,setStates]=useState([]);
  const [selected,setSelected]=useState(null);
  const [selectedBehaviour,setSelectedBehaviour]=useState(null);
  const behaviours=['wander','follow','shoot','protect','teleportToTarget','tossObject','suicide'];

  // Basic enemy stats
  const [enemyId, setEnemyId] = useState('untitled');
  const [enemyName, setEnemyName] = useState('New Enemy');
  const [sprite, setSprite] = useState('default');
  const [hp, setHp] = useState(50);
  const [speed, setSpeed] = useState(20);
  const [width, setWidth] = useState(1);
  const [height, setHeight] = useState(1);
  const [renderScale, setRenderScale] = useState(2);

  // Attack properties
  const [bulletId, setBulletId] = useState('arrow');
  const [attackCooldown, setAttackCooldown] = useState(2000);
  const [bulletSpeed, setBulletSpeed] = useState(25);
  const [bulletLifetime, setBulletLifetime] = useState(2000);
  const [bulletCount, setBulletCount] = useState(1);
  const [bulletSpread, setBulletSpread] = useState(0);

  // AI behavior
  const [behaviorTree, setBehaviorTree] = useState('BasicChaseAndShoot');

  // Load last enemy on mount
  useEffect(() => {
    const lastEnemy = loadLastEnemy();
    if (lastEnemy) {
      setEnemyId(lastEnemy.id || 'untitled');
      setEnemyName(lastEnemy.name || 'New Enemy');
      setSprite(lastEnemy.sprite || 'default');
      setHp(lastEnemy.hp || 50);
      setSpeed(lastEnemy.speed || 20);
      setWidth(lastEnemy.width || 1);
      setHeight(lastEnemy.height || 1);
      setRenderScale(lastEnemy.renderScale || 2);

      if (lastEnemy.attack) {
        setBulletId(lastEnemy.attack.bulletId || 'arrow');
        setAttackCooldown(lastEnemy.attack.cooldown || 2000);
        setBulletSpeed(lastEnemy.attack.speed || 25);
        setBulletLifetime(lastEnemy.attack.lifetime || 2000);
        setBulletCount(lastEnemy.attack.count || 1);
        setBulletSpread(lastEnemy.attack.spread || 0);
      }

      if (lastEnemy.ai) {
        setBehaviorTree(lastEnemy.ai.behaviorTree || 'BasicChaseAndShoot');
      }
    }
  }, []);

  // Auto-save whenever anything changes
  useEffect(() => {
    const enemyData = {
      id: enemyId,
      name: enemyName,
      sprite,
      hp,
      speed,
      width,
      height,
      renderScale,
      attack: {
        bulletId,
        cooldown: attackCooldown,
        speed: bulletSpeed,
        lifetime: bulletLifetime,
        count: bulletCount,
        spread: bulletSpread
      },
      ai: {
        behaviorTree
      }
    };
    scheduleAutoSave(enemyData);
  }, [enemyId, enemyName, sprite, hp, speed, width, height, renderScale, bulletId, attackCooldown, bulletSpeed, bulletLifetime, bulletCount, bulletSpread, behaviorTree]);

  const addState=()=>{
    const id='state_'+Date.now();
    setStates([...states,{id,x:40,y:40,behaviours:[]}]);
  };
  const addBehaviour=b=>{
    if(!selected) return;
    const newB={type:b,params:{}};
    setStates(states.map(s=>s.id===selected.id?{...s,behaviours:[...s.behaviours,newB]}:s));
  };
  const setPos=(id,x,y)=>setStates(states.map(s=>s.id===id?{...s,x,y}:s));

  const ajv = new Ajv();
  const schema = window.enemySchema? window.enemySchema:{};

  // Save to backend
  const saveToBackend = async () => {
    const enemy = {
      id: enemyId,
      name: enemyName,
      sprite,
      hp,
      speed,
      width,
      height,
      renderScale,
      attack: {
        bulletId,
        cooldown: attackCooldown,
        speed: bulletSpeed,
        lifetime: bulletLifetime,
        count: bulletCount,
        spread: bulletSpread
      },
      ai: {
        behaviorTree
      }
    };

    try {
      const response = await fetch('/api/enemy-editor/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enemy })
      });
      const result = await response.json();
      if (result.success) {
        alert(`Enemy "${enemyName}" saved to backend!`);
      } else {
        alert('Failed to save: ' + result.error);
      }
    } catch (err) {
      alert('Failed to save: ' + err.message);
    }
  };

  window.exportEnemy=()=>{
    const enemy = {
      id: enemyId,
      name: enemyName,
      sprite,
      hp,
      speed,
      width,
      height,
      renderScale,
      attack: {
        bulletId,
        cooldown: attackCooldown,
        speed: bulletSpeed,
        lifetime: bulletLifetime,
        count: bulletCount,
        spread: bulletSpread
      },
      ai: {
        behaviorTree
      }
    };
    const json=JSON.stringify(enemy,null,2);
    const blob=new Blob([json],{type:'application/json'});
    const a=document.createElement('a');
    a.href=URL.createObjectURL(blob);
    a.download=`${enemyId}.json`;
    a.click();
  };

  return (
    <>
      <div style={{display:'flex',width:'100%',height:'100%'}}>
        <div style={{width:220,borderRight:'1px solid #ccc',padding:10,overflowY:'auto'}}>
          {/* Stats Editor */}
          <div style={{marginBottom:12,borderBottom:'1px solid #ccc',paddingBottom:12,overflowY:'auto',maxHeight:'70vh'}}>
            <h4 style={{margin:'0 0 8px 0'}}>Enemy Stats</h4>
            <label style={{fontSize:11}}>ID:</label>
            <input type="text" value={enemyId} onChange={e=>setEnemyId(e.target.value)} style={{width:'100%',marginBottom:4,fontSize:11}}/>
            <label style={{fontSize:11}}>Name:</label>
            <input type="text" value={enemyName} onChange={e=>setEnemyName(e.target.value)} style={{width:'100%',marginBottom:4,fontSize:11}}/>
            <label style={{fontSize:11}}>Sprite:</label>
            <input type="text" value={sprite} onChange={e=>setSprite(e.target.value)} style={{width:'100%',marginBottom:4,fontSize:11}}/>
            <label style={{fontSize:11}}>HP:</label>
            <input type="number" value={hp} onChange={e=>setHp(Number(e.target.value))} style={{width:'100%',marginBottom:4,fontSize:11}}/>
            <label style={{fontSize:11}}>Speed:</label>
            <input type="number" value={speed} onChange={e=>setSpeed(Number(e.target.value))} style={{width:'100%',marginBottom:4,fontSize:11}}/>
            <label style={{fontSize:11}}>Width:</label>
            <input type="number" value={width} onChange={e=>setWidth(Number(e.target.value))} style={{width:'100%',marginBottom:4,fontSize:11}}/>
            <label style={{fontSize:11}}>Height:</label>
            <input type="number" value={height} onChange={e=>setHeight(Number(e.target.value))} style={{width:'100%',marginBottom:4,fontSize:11}}/>
            <label style={{fontSize:11}}>Render Scale:</label>
            <input type="number" value={renderScale} onChange={e=>setRenderScale(Number(e.target.value))} style={{width:'100%',marginBottom:8,fontSize:11}}/>

            <h5 style={{margin:'8px 0 4px 0'}}>Attack</h5>
            <label style={{fontSize:11}}>Bullet ID:</label>
            <input type="text" value={bulletId} onChange={e=>setBulletId(e.target.value)} style={{width:'100%',marginBottom:4,fontSize:11}}/>
            <label style={{fontSize:11}}>Cooldown (ms):</label>
            <input type="number" value={attackCooldown} onChange={e=>setAttackCooldown(Number(e.target.value))} style={{width:'100%',marginBottom:4,fontSize:11}}/>
            <label style={{fontSize:11}}>Bullet Speed:</label>
            <input type="number" value={bulletSpeed} onChange={e=>setBulletSpeed(Number(e.target.value))} style={{width:'100%',marginBottom:4,fontSize:11}}/>
            <label style={{fontSize:11}}>Bullet Lifetime:</label>
            <input type="number" value={bulletLifetime} onChange={e=>setBulletLifetime(Number(e.target.value))} style={{width:'100%',marginBottom:4,fontSize:11}}/>
            <label style={{fontSize:11}}>Bullet Count:</label>
            <input type="number" value={bulletCount} onChange={e=>setBulletCount(Number(e.target.value))} style={{width:'100%',marginBottom:4,fontSize:11}}/>
            <label style={{fontSize:11}}>Bullet Spread:</label>
            <input type="number" value={bulletSpread} onChange={e=>setBulletSpread(Number(e.target.value))} style={{width:'100%',marginBottom:8,fontSize:11}}/>

            <h5 style={{margin:'8px 0 4px 0'}}>AI</h5>
            <label style={{fontSize:11}}>Behavior Tree:</label>
            <input type="text" value={behaviorTree} onChange={e=>setBehaviorTree(e.target.value)} style={{width:'100%',marginBottom:8,fontSize:11}}/>

            <button onClick={saveToBackend} style={{width:'100%',marginTop:8,background:'#4CAF50',color:'white',border:'none',padding:8,fontSize:12,cursor:'pointer'}}>Save to Backend</button>
          </div>

          <button onClick={addState}>+ State</button>
          <hr/>
          <Palette behaviours={behaviours} onAdd={addBehaviour}/>
        </div>
        <div id="canvas" style={{flex:1,position:'relative'}}>
          {states.map(s=>(<Node key={s.id} state={s} selected={selected} onSelect={(st)=>{setSelected(st);setSelectedBehaviour(null);}} onPosChange={setPos}/>))}
          {selectedBehaviour && <ParamEditor behaviour={selectedBehaviour} onChange={(params)=>{
            setStates(states.map(st=>st.id===selected.id?{...st,behaviours:st.behaviours.map(b=>b===selectedBehaviour?{...b,params}:b)}:st));
            setSelectedBehaviour(null);
          }}/>
          }
        </div>
      </div>
    </>
  );
}

fetch('/schema/enemySchema.json').then(r=>r.json()).then(js=>{window.enemySchema=js;ReactDOM.createRoot(document.body).render(<App/>);});

document.getElementById('exportBtn').onclick=()=>window.exportEnemy(); 