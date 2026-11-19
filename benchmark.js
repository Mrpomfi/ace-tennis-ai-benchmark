import { AutoTokenizer, CLIPTextModelWithProjection } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2';

const $=id=>document.getElementById(id);
const video=$('video'), input=$('videoFile'), run=$('run'), status=$('status'), progress=$('progress');
let objectUrl=null, frames=[], busy=false;
const BASE='https://raw.githubusercontent.com/Mrpomfi/ace-tennis-ai/main/models/courtside-yolov11n.onnx';
const WORLD='https://huggingface.co/Instemic/yolo-world-onnx/resolve/main/yolov8s-worldv2.onnx';
const CLIP_MODEL='Xenova/clip-vit-base-patch32';

input.addEventListener('change',()=>{
  const f=input.files?.[0]; if(!f)return;
  if(objectUrl)URL.revokeObjectURL(objectUrl);
  objectUrl=URL.createObjectURL(f);video.src=objectUrl;video.classList.remove('hidden');
  video.onloadedmetadata=()=>{$('videoInfo').textContent=`${f.name} · ${video.videoWidth}×${video.videoHeight} · ${video.duration.toFixed(2)} s`;run.disabled=false;};
});
$('threshold').addEventListener('input',e=>$('thresholdValue').textContent=Number(e.target.value).toFixed(2).replace('.',','));
function say(s){status.textContent=s}
function wait(ms){return new Promise(r=>setTimeout(r,ms))}
async function seekVideo(t){
  if(Math.abs(video.currentTime-t)<0.015 && video.readyState>=2 && !video.seeking)return;
  await new Promise((resolve,reject)=>{
    let timer;
    let settled=false;
    const cleanup=()=>{video.removeEventListener('seeked',check);video.removeEventListener('timeupdate',check);video.removeEventListener('error',failed);clearTimeout(timer)};
    const finish=()=>{if(settled)return;settled=true;cleanup();requestAnimationFrame(()=>requestAnimationFrame(resolve));};
    const check=()=>{if(!video.seeking && Math.abs(video.currentTime-t)<0.08 && video.readyState>=2)finish();};
    const failed=()=>{if(settled)return;settled=true;cleanup();reject(new Error('Video-Frame konnte nicht angesprungen werden.'))};
    video.addEventListener('seeked',check);
    video.addEventListener('timeupdate',check);
    video.addEventListener('error',failed,{once:true});
    timer=setTimeout(()=>{if(!settled){settled=true;cleanup();reject(new Error('Zeitüberschreitung beim Lesen eines Video-Frames. Bitte Video erneut auswählen oder die Framezahl reduzieren.'))}},15000);
    video.currentTime=t;
    check();
  });
}
async function extractFrames(count=24){
  const old=video.currentTime,duration=video.duration,canvas=document.createElement('canvas'),ctx=canvas.getContext('2d',{willReadFrequently:true});
  canvas.width=video.videoWidth;canvas.height=video.videoHeight;
  if(!canvas.width||!canvas.height||!Number.isFinite(duration))throw new Error('Video ist noch nicht vollständig bereit.');
  const out=[];
  try{
    for(let i=0;i<count;i++){
      const t=duration*(i/(count-1||1));
      await seekVideo(t);
      ctx.drawImage(video,0,0,canvas.width,canvas.height);
      const c=document.createElement('canvas');c.width=canvas.width;c.height=canvas.height;c.getContext('2d').drawImage(canvas,0,0);
      out.push({time:t,canvas:c});
    }
  }finally{
    try{await seekVideo(old)}catch{}
  }
  return out;
}
function makeInput(canvas){
  const size=640, scale=Math.min(size/canvas.width,size/canvas.height),sw=Math.round(canvas.width*scale),sh=Math.round(canvas.height*scale),ox=(size-sw)/2,oy=(size-sh)/2;
  const c=document.createElement('canvas');c.width=size;c.height=size;const x=c.getContext('2d',{willReadFrequently:true});
  x.fillStyle='rgb(114,114,114)';x.fillRect(0,0,size,size);x.drawImage(canvas,ox,oy,sw,sh);
  const px=x.getImageData(0,0,size,size).data,plane=size*size,data=new Float32Array(plane*3);
  for(let i=0;i<plane;i++){const p=i*4;data[i]=px[p]/255;data[i+plane]=px[p+1]/255;data[i+2*plane]=px[p+2]/255;}
  return {tensor:new ort.Tensor('float32',data,[1,3,size,size]),scale,ox,oy};
}
function sigmoid(x){return 1/(1+Math.exp(-x))}
function iou(a,b){const x1=Math.max(a.x,b.x),y1=Math.max(a.y,b.y),x2=Math.min(a.x+a.w,b.x+b.w),y2=Math.min(a.y+a.h,b.y+b.h);const inter=Math.max(0,x2-x1)*Math.max(0,y2-y1);return inter/(a.w*a.h+b.w*b.h-inter+1e-9)}
function nms(boxes,limit=.45){boxes.sort((a,b)=>b.score-a.score);const keep=[];for(const b of boxes){if(keep.every(k=>iou(k,b)<limit))keep.push(b);if(keep.length>=30)break;}return keep}
function mapBox(x1,y1,x2,y2,score,meta,canvas){
 const {scale,ox,oy}=meta;const x=Math.max(0,(x1-ox)/scale),y=Math.max(0,(y1-oy)/scale),r=Math.min(canvas.width,(x2-ox)/scale),b=Math.min(canvas.height,(y2-oy)/scale);
 if(r<=x||b<=y)return null;return{x:x/canvas.width,y:y/canvas.height,w:(r-x)/canvas.width,h:(b-y)/canvas.height,score};
}
function parseBase(out,meta,canvas,threshold){
 const t=out[Object.keys(out)[0]],v=t.data,d=t.dims,boxes=[];
 if(d.length===3&&d[2]===6){for(let i=0;i<d[1];i++){const j=i*6,score=v[j+4],cls=Math.round(v[j+5]);if(cls===0&&score>=threshold){const b=mapBox(v[j],v[j+1],v[j+2],v[j+3],score,meta,canvas);if(b)boxes.push(b)}}}
 else if(d.length===3&&d[1]>=5&&d[1]<=128){const ch=d[1],n=d[2];for(let i=0;i<n;i++){let best=-1,cls=-1;for(let c=4;c<ch;c++){if(v[c*n+i]>best){best=v[c*n+i];cls=c-4}}if(cls===0&&best>=threshold){const cx=v[i],cy=v[n+i],w=v[2*n+i],h=v[3*n+i],b=mapBox(cx-w/2,cy-h/2,cx+w/2,cy+h/2,best,meta,canvas);if(b)boxes.push(b)}}}
 else throw Error('Unbekanntes CourtSide-Ausgabeformat: '+d.join('×'));
 return nms(boxes);
}
function parseWorld(out,meta,canvas,threshold){
 const t=out[Object.keys(out)[0]],v=t.data,d=t.dims,boxes=[];
 if(d.length!==3)throw Error('Unbekanntes YOLO-World-Ausgabeformat: '+d.join('×'));
 const channelsFirst=d[1]<d[2],channels=channelsFirst?d[1]:d[2],n=channelsFirst?d[2]:d[1];
 if(channels<5)throw Error('YOLO-World-Ausgabe enthält keine Klassenwerte.');
 for(let i=0;i<n;i++){
  const get=c=>channelsFirst?v[c*n+i]:v[i*channels+c];
  let score=0;for(let c=4;c<channels;c++)score=Math.max(score,sigmoid(get(c)));
  if(score<threshold)continue;
  const cx=get(0),cy=get(1),w=get(2),h=get(3),b=mapBox(cx-w/2,cy-h/2,cx+w/2,cy+h/2,score,meta,canvas);if(b)boxes.push(b);
 }
 return nms(boxes);
}
async function loadOrt(){
 if(!window.ort?.InferenceSession)throw Error('ONNX Runtime konnte nicht geladen werden.');
 ort.env.wasm.wasmPaths={mjs:'https://cdn.jsdelivr.net/gh/Mrpomfi/ace-tennis-ai@main/vendor/onnxruntime-web/ort-wasm-simd-threaded.js',wasm:'https://cdn.jsdelivr.net/gh/Mrpomfi/ace-tennis-ai@main/vendor/onnxruntime-web/ort-wasm-simd-threaded.wasm'};
 ort.env.wasm.numThreads=1;
}
async function createClipFeatures(){
 say('CLIP-Textencoder wird geladen. Das kann beim ersten Mal etwas dauern …');
 const tokenizer=await AutoTokenizer.from_pretrained(CLIP_MODEL);
 const model=await CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL,{dtype:'q8'});
 const prompts=['a photo of a tennis racket','a close-up photo of a tennis racquet'];
 const tokens=tokenizer(prompts,{padding:true,truncation:true});
 const result=await model(tokens);
 const d=result.text_embeds.dims,raw=result.text_embeds.data,features=new Float32Array(1*prompts.length*512);
 for(let i=0;i<prompts.length;i++){let norm=0;for(let j=0;j<512;j++)norm+=raw[i*512+j]**2;norm=Math.sqrt(norm)||1;for(let j=0;j<512;j++)features[i*512+j]=raw[i*512+j]/norm;}
 model.dispose?.();return new ort.Tensor('float32',features,[1,prompts.length,512]);
}
async function runTest(){
 if(busy)return;busy=true;run.disabled=true;progress.value=0;$('frames').innerHTML='';
 let base=null,world=null,textFeatures=null;
 try{
  say('Video-Frames werden vorbereitet …');frames=await extractFrames(24);
  await loadOrt();
  const threshold=Number($('threshold').value),baseTimes=[],worldTimes=[],baseResults=[];
  let baseTotal=0,worldTotal=0;
  say('CourtSide-Modell wird geladen …');
  const baseBytes=await (await fetch(BASE)).arrayBuffer();
  base=await ort.InferenceSession.create(baseBytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
  for(let i=0;i<frames.length;i++){
   const f=frames[i],meta=makeInput(f.canvas),t0=performance.now();
   if(base.inputNames.length!==1)throw Error('CourtSide erwartet einen einzelnen Bildeingang; gefunden: '+base.inputNames.join(', '));
   const out=await base.run({[base.inputNames[0]]:meta.tensor});
   const boxes=parseBase(out,meta,f.canvas,threshold);baseResults.push(boxes);
   baseTimes.push(performance.now()-t0);baseTotal+=boxes.length;
   progress.value=Math.round((i+1)/(frames.length*2)*100);
   say(`CourtSide: Frame ${i+1}/${frames.length}`);
   await wait(0);
  }
  base.release?.();base=null;
  say('CourtSide abgeschlossen. YOLO-World wird geladen …');
  const worldBytes=await (await fetch(WORLD)).arrayBuffer();
  world=await ort.InferenceSession.create(worldBytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
  const requiredInputs=['images','txt_feats'];
  const missing=requiredInputs.filter(name=>!world.inputNames.includes(name));
  if(missing.length)throw Error('YOLO-World-Eingänge fehlen: '+missing.join(', ')+'. Gefunden: '+world.inputNames.join(', '));
  if(world.outputNames.length!==1)throw Error('Unerwartete YOLO-World-Ausgaben: '+world.outputNames.join(', '));
  textFeatures=await createClipFeatures();
  if(textFeatures.dims[1]!==2||textFeatures.dims[2]!==512)throw Error('CLIP-Textmerkmale haben eine unerwartete Form: '+textFeatures.dims.join('×'));
  for(let i=0;i<frames.length;i++){
   const f=frames[i],meta=makeInput(f.canvas),t0=performance.now();
   const out=await world.run({images:meta.tensor,txt_feats:textFeatures});
   const boxes=parseWorld(out,meta,f.canvas,threshold);worldTimes.push(performance.now()-t0);worldTotal+=boxes.length;
   drawFrame(f,baseResults[i],boxes,i);
   progress.value=50+Math.round((i+1)/(frames.length*2)*100);
   say(`YOLO-World: Frame ${i+1}/${frames.length} · CourtSide: ${baseResults[i].length} · YOLO-World: ${boxes.length}`);
   await wait(0);
  }
  $('baseCount').textContent=baseTotal;$('worldCount').textContent=worldTotal;
  $('baseTime').textContent=(baseTimes.reduce((a,b)=>a+b,0)/baseTimes.length).toFixed(0)+' ms';
  $('worldTime').textContent=(worldTimes.reduce((a,b)=>a+b,0)/worldTimes.length).toFixed(0)+' ms';
  say('Vergleich abgeschlossen. Bitte prüfe die Bildüberlagerungen: Blau = CourtSide, Orange = YOLO-World. Die Trefferzahl allein sagt noch nicht, welches Modell den echten Schläger gefunden hat.');
 }catch(e){console.error(e);say('Test konnte nicht abgeschlossen werden: '+(e?.message||e)+'\\nBitte Screenshot dieser Meldung senden.');}
 finally{textFeatures?.dispose?.();base?.release?.();world?.release?.();busy=false;run.disabled=false;}
}
function drawFrame(f,a,b,i){
 const c=document.createElement('canvas');c.width=480;c.height=Math.round(480*f.canvas.height/f.canvas.width);const x=c.getContext('2d');x.drawImage(f.canvas,0,0,c.width,c.height);
 function boxes(arr,color){x.strokeStyle=color;x.lineWidth=3;for(const z of arr){x.strokeRect(z.x*c.width,z.y*c.height,z.w*c.width,z.h*c.height);x.fillStyle=color;x.font='12px sans-serif';x.fillText(z.score.toFixed(2),z.x*c.width,Math.max(12,z.y*c.height-3));}}
 boxes(a,'#38a4ff');boxes(b,'#ff9c42');c.title='Frame '+(i+1)+' · '+f.time.toFixed(2)+' s';$('frames').append(c);
}
run.addEventListener('click',runTest);
