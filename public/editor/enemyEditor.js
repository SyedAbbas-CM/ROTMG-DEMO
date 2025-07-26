const { useState } = React;
const { Draggable } = ReactDraggable;

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

  window.exportEnemy=()=>{
    const enemy={id:'demo',name:'Demo',sprite:'demo',hp:50,states:{}};
    states.forEach(s=>{enemy.states[s.id]={behaviours:s.behaviours}});
    const valid = ajv.validate(schema,enemy);
    if(!valid){ alert('Validation errors, open console'); console.warn(ajv.errors); return; }
    const json=JSON.stringify(enemy,null,2);
    const blob=new Blob([json],{type:'application/json'});
    const a=document.createElement('a');
    a.href=URL.createObjectURL(blob);
    a.download='enemy.enemy.json';
    a.click();
  };

  return (
    <>
      <div style={{display:'flex',width:'100%',height:'100%'}}>
        <div style={{width:220,borderRight:'1px solid #ccc',padding:10,overflowY:'auto'}}>
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