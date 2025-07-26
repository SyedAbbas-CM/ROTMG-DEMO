import fs from 'fs';
import path from 'path';
import { pathToFileURL } from 'url';
import { spawnSync } from 'child_process';

/**
 * Write manifest & impl to an unverified folder, run a smoke-test in a child
 * Node process, then move into live capabilities folder on success.
 * @param {string} manifestStr – raw JSON string of schema.json
 * @param {string} implStr     – raw JS source of implementation.js
 * @returns {Promise<{ ok:boolean, error?:string }>}
 */
export async function defineComponent(manifestStr, implStr) {
  let manifest;
  try {
    manifest = JSON.parse(manifestStr);
  } catch (err) {
    return { ok: false, error: 'manifest_not_json' };
  }
  const capId = manifest.$id;
  if (!capId) return { ok:false, error:'missing_$id' };

  const [group, rest] = capId.split(':');
  const [name, versionRaw] = rest.split('@');
  const version = versionRaw || '0.0.0';

  const baseUnverified = path.resolve('src','capabilities','unverified',group,name,version);
  fs.mkdirSync(baseUnverified,{recursive:true});
  fs.writeFileSync(path.join(baseUnverified,'schema.json'), manifestStr);
  fs.writeFileSync(path.join(baseUnverified,'implementation.js'), implStr);

  // --------------------------------------------------
  // Simple smoke-test: ensure compile+invoke exist and run without throwing
  // --------------------------------------------------
  const testScript = `import('${pathToFileURL(path.join(baseUnverified,'implementation.js')).href}').then(m=>{
    if(typeof m.compile!=='function'||typeof m.invoke!=='function') process.exit(2);
    const node=m.compile({type:'${capId}'});
    const ctx={dt:0.016,bossMgr:{x:[0],y:[0],id:['b1'],worldId:['default']},bulletMgr:{addBullet:()=>{}}};
    try{m.invoke(node,{},ctx);}catch(e){process.exit(3);}process.exit(0);
  }).catch(()=>process.exit(4));`;
  const res = spawnSync(process.execPath,['-e',testScript],{stdio:'inherit'});
  if(res.status!==0){
    const rejDir = path.resolve('src','capabilities','rejected',`${group}_${name}_${Date.now()}`);
    fs.renameSync(baseUnverified, rejDir);
    return { ok:false, error:`smoke_test_failed(code ${res.status})` };
  }

  // Move folder into live tree (capabilities/Group/Name/Version)
  const liveDir = path.resolve('src','capabilities',group,name,version);
  fs.mkdirSync(path.dirname(liveDir),{recursive:true});
  fs.renameSync(baseUnverified, liveDir);
  return { ok:true };
} 