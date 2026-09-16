'use strict';
const $ = id => document.getElementById(id);
let frameId = null, imageUrl = null, busy = false, state = null, stateBusy = false;
let desiredSelection = '';
let specimenOptionsSignature = '';
const createdSpecimens = new Map();
const labels = {normal:'Good', anomalous:'Defective', unknown:'Unsure'};
const chosenLabel = () => document.querySelector('input[name=label]:checked')?.value;
function ready(){ $('shutter').disabled = busy || !frameId || !$('specimen').value || !chosenLabel(); }
function status(message, error=false){$('status').textContent=message;$('status').classList.toggle('error',error);}
async function request(path, body){const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await response.json();if(!response.ok)throw Error(data.error||'Request failed');return data;}
function selectBlock(){const established=state?.specimens.find(specimen=>specimen.specimen_id===$('specimen').value)?.label;document.querySelectorAll('input[name=label]').forEach(input=>{input.checked=!!established&&input.value===established;});ready();}
function renderSpecimens(specimens){const merged=new Map(createdSpecimens);for(const specimen of specimens)merged.set(specimen.specimen_id,specimen);const rows=Array.from(merged.values());const signature=JSON.stringify(rows);if(signature!==specimenOptionsSignature){$('specimen').replaceChildren(new Option('Select a block',''),...rows.map(s=>new Option(`${s.name} · ${s.specimen_id.slice(-4)}${s.label?' · '+labels[s.label]:''}`,s.specimen_id)));specimenOptionsSignature=signature;}if($('specimen').value!==desiredSelection)$('specimen').value=desiredSelection;}
async function refresh(){
 if(stateBusy)return;stateBusy=true;
 try{const response=await fetch('/api/state');if(!response.ok)throw Error('Server unavailable');const next=await response.json();
 renderSpecimens(next.specimens);
 $('counter').replaceChildren(document.createTextNode(next.count+' '),Object.assign(document.createElement('span'),{textContent:'/ 50'}));
 $('mode').textContent=next.synthetic?'SYNTHETIC FIXTURE':'LIVE · CAMERA 1 LOW';
 document.querySelectorAll('input[name=label]').forEach(input=>input.disabled=next.synthetic&&input.value!=='unknown');
 if(next.mac_sync)$('sync').textContent=`Mac copies: ${next.mac_sync.copied_count} · Last sync ${next.mac_sync.last_sync_utc}. Saved on Intel first.`;
 if(!state||JSON.stringify(state.recent)!==JSON.stringify(next.recent)){
  $('gallery').replaceChildren(...next.recent.map(row=>{const card=document.createElement('article');card.className='card';const image=document.createElement('img');image.src='/images/'+row.path;image.alt=labels[row.label]+' '+row.specimen_id;const text=document.createElement('p');text.textContent=`${labels[row.label]} · ${row.specimen_id}\n${row.evidence_kind.toUpperCase()} · RGB PNG`;card.append(image,text);return card;}));
 }state=next;ready();
 }catch(error){status(error.message,true);}finally{stateBusy=false;}
}
async function preview(){
 try{if(!busy){const response=await fetch('/api/preview');if(!response.ok){const error=await response.json();throw Error(error.error||'Camera unavailable');}const id=response.headers.get('X-Frame-ID');const url=URL.createObjectURL(await response.blob());const decoded=new Image();decoded.src=url;await decoded.decode();
 if(!busy){const previous=imageUrl;$('preview').src=url;imageUrl=url;frameId=id;if($('status').textContent==='Waiting for the camera…')status('Choose a block and label, then save a photo.');if(previous)URL.revokeObjectURL(previous);ready();}else URL.revokeObjectURL(url);
 }}catch(error){frameId=null;ready();if(!busy)status(error.message,true);}finally{setTimeout(preview,170);}
}
async function shutter(){
 if($('shutter').disabled)return;busy=true;ready();const payload={frame_id:frameId,specimen_id:$('specimen').value,label:chosenLabel()};status('Saving the displayed frame…');
 try{await request('/api/capture',payload);status(`Saved ${labels[payload.label]} photo. Keep this block ID for new angles.`);await refresh();}
 catch(error){status(error.message,true);}finally{busy=false;frameId=null;ready();}
}
$('new-block').addEventListener('click',async()=>{try{const row=await request('/api/specimens',{});createdSpecimens.set(row.specimen_id,row);desiredSelection=row.specimen_id;renderSpecimens(state?.specimens||[]);selectBlock();status(`${row.name} selected. Choose its label before saving.`);await refresh();}catch(error){status(error.message,true);}});
$('specimen').addEventListener('change',()=>{desiredSelection=$('specimen').value;selectBlock();});document.querySelectorAll('input[name=label]').forEach(input=>input.addEventListener('change',ready));$('shutter').addEventListener('click',shutter);
document.addEventListener('keydown',event=>{const target=event.target;const editing=['SELECT','TEXTAREA','BUTTON'].includes(target.tagName)||(target.tagName==='INPUT'&&target.type!=='radio')||target.isContentEditable;if(event.code==='Space'&&!event.repeat&&!editing){event.preventDefault();shutter();}});
refresh();preview();setInterval(refresh,1500);
