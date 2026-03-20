// app.js — head-vs-gaze region logic + improved down detection + dot hints + two-stage messaging
// + Teleprompter (notes input modal + wheel UI + speech-driven autoadvance)
// NOTE: Teleprompter hides overlay visuals but keeps tracking running for LOOK UP cue.

// FaceLandmarker and FilesetResolver are loaded via classic <script> tag in start()
// because vision_bundle.js contains Emscripten WASM glue that is invalid in strict ES module mode.
let FaceLandmarker, FilesetResolver;

const overlay  = document.getElementById("overlay");
const octx     = overlay.getContext("2d");
const video    = document.getElementById("cam");

const topBar   = document.getElementById("topBar");
const bottomBar= document.getElementById("bottomBar");
const statusEl = document.getElementById("status");
const startCalBtn  = document.getElementById("startCalBtn");

const metricsPanel = document.getElementById("metricsPanel");

// LOOK UP cue
const lookUpCue = document.getElementById("lookUpCue");

// Calibration UI
const calLayer      = document.getElementById("calLayer");
const blackout      = document.getElementById("blackout");
const primeStartBtn = document.getElementById("primeStartBtn");

const centerStage   = document.getElementById("centerStage");
const centerCanvas  = document.getElementById("centerCanvas");
const cctx          = centerCanvas.getContext("2d");
const centerHint    = document.getElementById("centerHint");
const centerPctEl   = document.getElementById("centerPct");

const calHUD        = document.getElementById("calCenterHUD");
const calInstruction= document.getElementById("calInstruction");
const calTarget     = document.getElementById("calTarget");
const calProgress   = document.getElementById("calProgress");

let faceLandmarker, running=false;
let lastResult = null;
let hasCalibrated = false;

// ---------------------------
// Teleprompter UI + Speech
// ---------------------------
let teleprompterActive = false;
let notesModalOpen = false;
let notesText = "";

let tpBtn = null;
let notesModal = null;
let notesTextarea = null;
let notesDoneBtn = null;
let notesCancelBtn = null;
let exitPrompterBtnEl = null;
let voiceScrollBtnEl = null;

let teleprompterEl = null;
let tpContentEl = null;
let tpExitBtn = null;

let voiceScrollEnabled = true;

let tpLines = [];
let tpTokens = [];
let tpMatchTokens = [];

let tpWheelWrapEl = null;
let tpWheelEl = null;
let tpPrevEl = null;
let tpCurEl = null;
let tpNextEl = null;

let currentLineIndex = 0;
let currentTokenPos = 0;

let tpAnimating = false;
const TP_ANIM_MS = 420;

const STOPWORDS = new Set([
  "the","a","an","and","or","but","so","to","of","in","on","for","with","at","by","from",
  "is","are","was","were","be","been","being","that","this","it","as","into","over","than",
  "i","you","we","they","he","she","my","your","our","their","me","us","them"
]);

let recognition = null;
let recognitionRunning = false;
let speechSupported = false;

// Landmark indices (MediaPipe)
const RIGHT_EYE=[33,7,163,144,145,153,154,155,133,246,161,160,159,158,157,173];
const LEFT_EYE =[263,249,390,373,374,380,381,382,362,466,388,387,386,385,384,398];
const R_CORNER_OUT=33, R_CORNER_IN=133, R_LID_UP=159, R_LID_DN=145;
const L_CORNER_OUT=263, L_CORNER_IN=362, L_LID_UP=386, L_LID_DN=374;
const RIGHT_IRIS_CENTER=468, LEFT_IRIS_CENTER=473;
const FACE_OVAL=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];

const CHIN_INDEX = 152;
const FOREHEAD_INDEX = 10;

const EAR_CLOSED=0.18, EAR_HALF_BLINK=0.24;

/* Models */
let modelLX=null, modelLY=null, modelRX=null, modelRY=null;
let affineL=null, affineR=null;

/* Calibration / head state */
let isCalibrating=false;
let overlaysEnabled=false;

let baselineHead=null;
let headCalib = null;     // { center, scaleH, angle, scaleNorm, chinY, foreheadY, chinOffset, faceSpan }
let headVelEMA=0;

/* Depth calibration */
let currentDepthRatio = 1;      // 1 = baseline distance (NEAR)
let depthCalib = null;          // { nearScale, farScale }

/* Region tracking (left/center/right/down) */
let regionTimes = { left: 0, center: 0, right: 0, down: 0 }; // ms
let lastRegionTimestamp = null;

/* Two-stage dot calibration scales */
let currentCalStage = "near";   // "near" or "far"
let nearScales = [];
let farScales = [];

/* Dot hint element */
let dotHintEl = null;

/* LOOK UP detection state */
const DOWN_TRIGGER_MS = 1300;
const UP_CLEAR_MS = 250;
let downStartMs = null;
let upSinceMs = null;
let cueActive = false;

/* ---------- Canvas sizing ---------- */
function resizeCanvasToCSS(){
  const cssW = Math.max(1, Math.floor(window.innerWidth));
  const cssH = Math.max(1, Math.floor(window.innerHeight));
  const dpr  = window.devicePixelRatio || 1;

  if (overlay.width !== Math.floor(cssW * dpr) || overlay.height !== Math.floor(cssH * dpr)) {
    overlay.width  = Math.floor(cssW * dpr);
    overlay.height = Math.floor(cssH * dpr);
  }
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);

  if (centerCanvas.width !== Math.floor(cssW * dpr) || centerCanvas.height !== Math.floor(cssH * dpr)) {
    centerCanvas.width  = Math.floor(cssW * dpr);
    centerCanvas.height = Math.floor(cssH * dpr);
  }
  cctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", resizeCanvasToCSS);
document.addEventListener("fullscreenchange", resizeCanvasToCSS);
document.addEventListener("webkitfullscreenchange", resizeCanvasToCSS);

/* ---------- Fullscreen helper ---------- */
async function goFullscreen(){
  const elem = document.documentElement;
  try{
    if(elem.requestFullscreen) await elem.requestFullscreen();
    else if(elem.webkitRequestFullscreen) await elem.webkitRequestFullscreen();
  }catch{}
  await new Promise(r=>setTimeout(r,80));
  resizeCanvasToCSS();
}

/* ---------- Video mapping ---------- */
function videoRectCSS(){
  const cw = overlay.clientWidth;
  const ch = overlay.clientHeight;
  const vw = video.videoWidth  || 1280;
  const vh = video.videoHeight || 720;
  const scale = Math.min(cw/vw, ch/vh);
  const dw = vw * scale;
  const dh = vh * scale;
  const ox = (cw - dw) / 2;
  const oy = (ch - dh) / 2;
  return { x:ox, y:oy, w:dw, h:dh };
}

function mapToCSS(pt, rect){
  return {
    x: rect.x + pt.x * rect.w,
    y: rect.y + pt.y * rect.h
  };
}

/* ---------- Utils ---------- */
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const dist2D=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
const lerp=(a,b,t)=>a+(b-a)*t;
const mean = arr => arr.length ? arr.reduce((s,v)=>s+v,0)/arr.length : 0;

/* ---------- LOOK UP cue helpers ---------- */
function setLookUpCueActive(on){
  cueActive = !!on;
  if(!lookUpCue) return;
  lookUpCue.classList.toggle("active", cueActive);
  lookUpCue.setAttribute("aria-hidden", cueActive ? "false" : "true");
}
function resetLookUpCueState(){
  downStartMs = null;
  upSinceMs = null;
  if(cueActive) setLookUpCueActive(false);
}
function updateLookUpCue(downNow){
  const eligible =
    hasCalibrated &&
    (overlaysEnabled || teleprompterActive) &&
    !isCalibrating &&
    !centeringActive;

  if(!eligible){
    resetLookUpCueState();
    return;
  }

  const t = performance.now();

  if(downNow){
    upSinceMs = null;
    if(downStartMs === null) downStartMs = t;
    if(!cueActive && (t - downStartMs) >= DOWN_TRIGGER_MS){
      setLookUpCueActive(true);
    }
  } else {
    downStartMs = null;
    if(cueActive){
      if(upSinceMs === null) upSinceMs = t;
      if((t - upSinceMs) >= UP_CLEAR_MS){
        setLookUpCueActive(false);
      }
    }
  }
}

/* ---------- Eye features ---------- */
function eyeBox(lm, idxs){
  let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
  for(const i of idxs){
    const p=lm[i];
    if(p.x<minX)minX=p.x;
    if(p.y<minY)minY=p.y;
    if(p.x>maxX)maxX=p.x;
    if(p.y>maxY)maxY=p.y;
  }
  const pad=Math.max(4,(maxX-minX+maxY-minY)*0.14);
  return { x:minX-pad, y:minY-pad, w:(maxX+pad)-(minX-pad), h:(maxY+pad)-(minY-pad) };
}

function irisPoints(lm){
  return {
    L:{x:lm[LEFT_IRIS_CENTER].x,y:lm[LEFT_IRIS_CENTER].y},
    R:{x:lm[RIGHT_IRIS_CENTER].x,y:lm[RIGHT_IRIS_CENTER].y}
  };
}

function perEyeFeatures(side, lm, box){
  const iris = side==="L"
    ? {x:lm[LEFT_IRIS_CENTER].x,y:lm[LEFT_IRIS_CENTER].y}
    : {x:lm[RIGHT_IRIS_CENTER].x,y:lm[RIGHT_IRIS_CENTER].y};

  const nx = (iris.x - (box.x+box.w/2)) / (box.w*0.5);
  const ny = (iris.y - (box.y+box.h/2)) / (box.h*0.5);

  const upIdx= side==="L"?L_LID_UP:R_LID_UP;
  const dnIdx= side==="L"?L_LID_DN:R_LID_DN;
  const inIdx= side==="L"?L_CORNER_IN:R_CORNER_IN;
  const ouIdx= side==="L"?L_CORNER_OUT:R_CORNER_OUT;

  const upY=lm[upIdx].y, dnY=lm[dnIdx].y;
  const inX=lm[inIdx].x, ouX=lm[ouIdx].x;
  const horiz=Math.abs(inX-ouX), vert=Math.abs(upY-dnY);
  const aperture=horiz>0?vert/horiz:0;

  const topM=Math.max(0, iris.y-upY);
  const botM=Math.max(0, dnY-iris.y);
  const occ=(botM-topM)/(topM+botM+1e-6);

  const nx2   = nx*nx;
  const ny2   = ny*ny;
  const nxy   = nx*ny;
  const occNx = occ*nx;
  const occNy = occ*ny;
  const apNx  = aperture*nx;
  const apNy  = aperture*ny;

  const feat = [
    nx, ny, occ, aperture,
    nx2, ny2, nxy,
    occNx, occNy, apNx, apNy,
    1
  ];

  return { feat, nx, ny, occ, aperture, iris };
}

/* ---------- Ridge + affine ---------- */
function fitRidge(X,y,lambda=1e-3){
  const n=X.length,f=X[0].length, XtX=new Float64Array(f*f).fill(0), Xty=new Float64Array(f).fill(0);
  for(let i=0;i<n;i++){
    const xi=X[i], yi=y[i];
    for(let a=0;a<f;a++){
      Xty[a]+=xi[a]*yi;
      const base=a*f;
      for(let b=0;b<f;b++) XtX[base+b]+=xi[a]*xi[b];
    }
  }
  for(let d=0;d<f;d++) XtX[d*f+d]+=lambda;
  const A=new Float64Array(XtX), b=new Float64Array(Xty);
  for(let i=0;i<f;i++){
    let piv=A[i*f+i];
    if(Math.abs(piv)<1e-12){
      for(let r=i+1;r<f;r++){
        if(Math.abs(A[r*f+i])>Math.abs(piv)){
          for(let c=i;c<f;c++){
            const t=A[i*f+c]; A[i*f+c]=A[r*f+c]; A[r*f+c]=t;
          }
          const tb=b[i]; b[i]=b[r]; b[r]=tb;
          piv=A[i*f+i]; break;
        }
      }
    }
    const inv=1/piv;
    for(let c=i;c<f;c++) A[i*f+c]*=inv;
    b[i]*=inv;
    for(let r=0;r<f;r++){
      if(r===i) continue;
      const factor=A[r*f+i]; if(!factor) continue;
      for(let c=i;c<f;c++) A[r*f+c]-=factor*A[i*f+c];
      b[r]-=factor*b[i];
    }
  }
  return { w:b };
}
const predict=(m,feat)=>{ let s=0; for(let i=0;i<m.w.length;i++) s+=m.w[i]*feat[i]; return s; };

function fitAffine2D(predPts, targetPts, lambda=1e-6){
  if(predPts.length!==targetPts.length || predPts.length<3) return null;
  const X=predPts.map(p=>[p.x,p.y,1]), Yx=targetPts.map(t=>t.x), Yy=targetPts.map(t=>t.y);
  const Ax=fitRidge(X,Yx,lambda).w, Ay=fitRidge(X,Yy,lambda).w;
  return { Ax, Ay };
}
function applyAffine2D(A,p){
  if(!A) return p;
  return {
    x:A.Ax[0]*p.x+A.Ax[1]*p.y+A.Ax[2],
    y:A.Ay[0]*p.x+A.Ay[1]*p.y+A.Ay[2]
  };
}

/* ---------- Camera / Model ---------- */
async function setupCamera(){
  if(!navigator.mediaDevices?.getUserMedia){
    throw new Error("Camera API unavailable — open the page over HTTPS and try again.");
  }
  statusEl.textContent = "Requesting camera access…";
  let stream;
  try{
    stream = await navigator.mediaDevices.getUserMedia({
      video:{ facingMode:"user", width:{ ideal:1280 }, height:{ ideal:720 } },
      audio: false
    });
  }catch(err){
    const friendly = {
      NotAllowedError:       "Camera permission denied — allow access in your browser settings and refresh.",
      PermissionDeniedError: "Camera permission denied — allow access in your browser settings and refresh.",
      NotFoundError:         "No camera found — connect a camera and refresh.",
      DevicesNotFoundError:  "No camera found — connect a camera and refresh.",
      NotReadableError:      "Camera is in use by another app — close it and refresh.",
      TrackStartError:       "Camera is in use by another app — close it and refresh.",
    }[err.name];
    throw new Error(friendly || `Camera error (${err.name}): ${err.message}`);
  }
  video.srcObject = stream;
  // Wait for enough video data. Use both canplay/loadeddata so we don't miss
  // the event if readyState already advanced past 1 (loadedmetadata).
  await new Promise((resolve, reject) => {
    if(video.readyState >= 2){ resolve(); return; }
    video.addEventListener("canplay",    resolve, { once: true });
    video.addEventListener("loadeddata", resolve, { once: true });
    video.play().catch(reject);
  });
  if(video.paused) await video.play();
  resizeCanvasToCSS();
}
async function setupFaceLandmarker(){
  statusEl.textContent = "Loading face model…";
  try{
    const filesetResolver = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions:{ modelAssetPath:"https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" },
      numFaces:1, runningMode:"VIDEO",
      outputFaceBlendshapes:false, outputFacialTransformationMatrixes:true,
      minFaceDetectionConfidence:0.6, minTrackingConfidence:0.6, minFacePresenceConfidence:0.6,
    });
  }catch(err){
    throw new Error(`Failed to load face model — check your internet connection. (${err.message})`);
  }
  statusEl.textContent = "Ready — press Start Calibration";
}

/* ---------- Drawing sizes ---------- */
const DOT_RADIUS = 2.0;
const PUPIL_DOT_R = 4;
const IRIS_RADIUS_RATIO = 0.15;
const RING_LINE_W = 1.8;

/* Pupils + rings (overlay mode) */
function drawPupilsAndRings(lm, L, R){
  const leftWidth  = Math.hypot(lm[L_CORNER_OUT].x - lm[L_CORNER_IN].x, lm[L_CORNER_OUT].y - lm[L_CORNER_IN].y);
  const rightWidth = Math.hypot(lm[R_CORNER_OUT].x - lm[R_CORNER_IN].x, lm[R_CORNER_OUT].y - lm[R_CORNER_IN].y);
  const rL = Math.max(3.5, leftWidth  * IRIS_RADIUS_RATIO);
  const rR = Math.max(3.5, rightWidth * IRIS_RADIUS_RATIO);

  octx.fillStyle="#56b3ff";
  if(L){ octx.beginPath(); octx.arc(L.x,L.y,PUPIL_DOT_R,0,Math.PI*2); octx.fill(); }
  if(R){ octx.beginPath(); octx.arc(R.x,R.y,PUPIL_DOT_R,0,Math.PI*2); octx.fill(); }

  octx.lineWidth = RING_LINE_W;
  octx.strokeStyle = "rgba(124,249,165,0.95)";
  if(L){ octx.beginPath(); octx.arc(L.x, L.y, rL, 0, Math.PI*2); octx.stroke(); }
  if(R){ octx.beginPath(); octx.arc(R.x, R.y, rR, 0, Math.PI*2); octx.stroke(); }
}

/* Face mesh dots (overlay mode) */
function drawFaceMeshDots(lm){
  octx.fillStyle = "rgba(86,179,255,0.9)";
  for(let i=0;i<lm.length;i++){
    const p=lm[i];
    octx.beginPath();
    octx.arc(p.x, p.y, DOT_RADIUS, 0, Math.PI*2);
    octx.fill();
  }
  for(let i=0;i<FACE_OVAL.length;i++){
    const a=lm[FACE_OVAL[i]], b=lm[FACE_OVAL[(i+1)%FACE_OVAL.length]];
    for(const t of [0.2,0.4,0.6,0.8]){
      const x=a.x+(b.x-a.x)*t, y=a.y+(b.y-a.y)*t;
      octx.beginPath();
      octx.arc(x,y,DOT_RADIUS,0,Math.PI*2);
      octx.fill();
    }
  }
}

/* ---------- Gaze rays / dot ---------- */
function drawGazeRays(L,R,G){
  if(!G) return;
  octx.beginPath();
  octx.fillStyle="rgba(255,255,255,0.95)";
  octx.arc(G.x,G.y,12,0,Math.PI*2);
  octx.fill();

  octx.beginPath();
  octx.strokeStyle="rgba(255,255,255,0.35)";
  octx.lineWidth=2;
  octx.arc(G.x,G.y,18,0,Math.PI*2);
  octx.stroke();

  octx.setLineDash([8,6]);
  octx.lineWidth=2.5;
  octx.strokeStyle="rgba(255,255,255,0.6)";
  if(L){
    octx.beginPath();
    octx.moveTo(L.x,L.y);
    octx.lineTo(G.x,G.y);
    octx.stroke();
  }
  if(R){
    octx.beginPath();
    octx.moveTo(R.x,R.y);
    octx.lineTo(G.x,G.y);
    octx.stroke();
  }
  octx.setLineDash([]);
}

/* ---------- Head pose + clutch ---------- */
const HEAD_VEL_TRIG=2.5, CLUTCH_HOLD_FRAMES=8;
let clutchFrames=0, prevHeadMid=null, gazeEMA=null, prevGaze=null;

function computeHeadPose(lm){
  const Lc={x:lm[L_CORNER_IN].x,y:lm[L_CORNER_IN].y};
  const Rc={x:lm[R_CORNER_IN].x,y:lm[R_CORNER_IN].y};
  const mid={x:(Lc.x+Rc.x)/2,y:(Lc.y+Rc.y)/2};
  const vec={x:Lc.x-Rc.x,y:Lc.y-Rc.y};
  return { mid, vec, angle:Math.atan2(vec.y,vec.x), scale:Math.hypot(vec.x,vec.y) };
}
function stabilizePointByHead(p, head, base){
  if(!base||!head) return p;
  let x=p.x-head.mid.x, y=p.y-head.mid.y;
  const dAng=base.angle-head.angle, cs=Math.cos(dAng), sn=Math.sin(dAng);
  let xr=x*cs - y*sn, yr=x*sn + y*cs;
  const s = head.scale>1e-3 ? (base.scale/head.scale) : 1.0;
  xr*=s; yr*=s;
  return { x:xr+base.mid.x, y:yr+base.mid.y };
}

/* ---------- Curve helpers ---------- */
function cumulativeLengths(pts){
  const L=[0]; let acc=0;
  for(let i=1;i<=pts.length;i++){
    const a=pts[i-1], b=pts[i%pts.length];
    acc += Math.hypot(b.x-a.x,b.y-a.y);
    L.push(acc);
  }
  return {L,total:acc};
}
function resampleClosedCurve(pts, N){
  const {L,total}=cumulativeLengths(pts);
  const out=[];
  for(let k=0;k<N;k++){
    const t=(k/N)*total;
    let i=0;
    while(i<L.length-1 && L[i+1]<t) i++;
    const segLen=L[i+1]-L[i];
    const u=segLen>0?(t-L[i])/segLen:0;
    const a=pts[i%pts.length], b=pts[(i+1)%pts.length];
    out.push({x:a.x+(b.x-a.x)*u, y:a.y+(b.y-a.y)*u});
  }
  return out;
}
function chaikinSubdivide(pts, iterations=1){
  let cur=pts.slice();
  for(let it=0; it<iterations; it++){
    const n=cur.length, out=[];
    for(let i=0;i<n;i++){
      const a=cur[i], b=cur[(i+1)%n];
      out.push({x:0.75*a.x+0.25*b.x, y:0.75*a.y+0.25*b.y});
      out.push({x:0.25*a.x+0.75*b.x, y:0.25*a.y+0.75*b.y});
    }
    cur=out;
  }
  return cur;
}
function morphPolylines(A,B,t){
  const n=Math.min(A.length,B.length);
  const out=new Array(n);
  for(let i=0;i<n;i++){
    out[i]={ x: A[i].x + (B[i].x-A[i].x)*t, y: A[i].y + (B[i].y-A[i].y)*t };
  }
  return out;
}
function drawClosedPath(ctx, pts){
  if(!pts.length) return;
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for(let i=1;i<pts.length;i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.closePath();
}
function drawClosedSpline(ctx, pts){
  const n=pts.length;
  if(n<2) return;
  ctx.beginPath();
  for(let i=0;i<n;i++){
    const p0=pts[(i-1+n)%n], p1=pts[i], p2=pts[(i+1)%n], p3=pts[(i+2)%n];
    const cp1x = p1.x + (p2.x - p0.x)/6;
    const cp1y = p1.y + (p2.y - p0.y)/6;
    const cp2x = p2.x - (p3.x - p1.x)/6;
    const cp2y = p2.y - (p3.y - p1.y)/6;
    if(i===0) ctx.moveTo(p1.x,p1.y);
    ctx.bezierCurveTo(cp1x,cp1y, cp2x,cp2y, p2.x,p2.y);
  }
  ctx.closePath();
}

/* Rectangle polyline centered at (cx,cy) of size w x h */
function rectPolyline(cx,cy,w,h,perEdge=64){
  const hw=w/2, hh=h/2;
  const corners=[
    {x:cx-hw,y:cy-hh},
    {x:cx+hw,y:cy-hh},
    {x:cx+hw,y:cy+hh},
    {x:cx-hw,y:cy+hh}
  ];
  const pts=[];
  for(let e=0;e<4;e++){
    const a=corners[e], b=corners[(e+1)%4];
    for(let i=0;i<perEdge;i++){
      const t=i/(perEdge);
      pts.push({ x:a.x+(b.x-a.x)*t, y:a.y+(b.y-a.y)*t });
    }
  }
  return pts;
}

/* ---------- Metrics / regions ---------- */
function resetRegionStats(){
  regionTimes.left = regionTimes.center = regionTimes.right = regionTimes.down = 0;
  lastRegionTimestamp = null;
  updateMetricsPanel();
}
function updateMetricsPanel(){
  if(!metricsPanel) return;

  // Hide metrics while teleprompter is active (cleaner prompter view)
  if(teleprompterActive){
    metricsPanel.textContent = "";
    return;
  }

  const fmt = ms => (ms/1000).toFixed(1);
  const dStr = currentDepthRatio.toFixed(2);
  metricsPanel.textContent =
    `Left: ${fmt(regionTimes.left)}s | Center: ${fmt(regionTimes.center)}s | Right: ${fmt(regionTimes.right)}s | Down: ${fmt(regionTimes.down)}s | Depth: ${dStr}×`;
}

/**
 * Returns: isDown (boolean) for cue logic.
 */
function updateRegionStats(headPose, gazePoint, chin, forehead){
  const now = performance.now();
  if(!lastRegionTimestamp){
    lastRegionTimestamp = now;
    return false;
  }
  const dt = now - lastRegionTimestamp;
  lastRegionTimestamp = now;

  const w = overlay.clientWidth || 1;
  const h = overlay.clientHeight || 1;

  let horizRegion = null;
  let isDown = false;

  const hasHead = !!headPose;
  const hasGaze = !!gazePoint;

  // Precompute gaze-based region/down if available
  let gazeHoriz = null;
  let gazeDown  = false;
  if(hasGaze){
    const gxNorm = gazePoint.x / w;
    const gyNorm = gazePoint.y / h;

    if(gxNorm < 1/3)      gazeHoriz = "left";
    else if(gxNorm < 2/3) gazeHoriz = "center";
    else                  gazeHoriz = "right";

    gazeDown = gyNorm > 0.75;
  }

  if(hasHead){
    const midXNorm = headPose.mid.x / w;
    const centerHoriz = (midXNorm >= 1/3 && midXNorm < 2/3);
    const leftHoriz   = midXNorm < 1/3;
    const rightHoriz  = midXNorm >= 2/3;

    // ---- HEAD-BASED "DOWN" via forehead–chin span + distance ----
    let downByHead = false;

    if(
      headCalib &&
      chin && forehead &&
      typeof headCalib.chinY === "number" &&
      typeof headCalib.foreheadY === "number"
    ){
      const baseSpan = headCalib.faceSpan ?? (headCalib.chinY - headCalib.foreheadY);
      const curSpan  = chin.y - forehead.y;

      // Depth correction
      let depthRatio = 1;
      if(baselineHead && headPose.scale > 1e-3){
        depthRatio = baselineHead.scale / headPose.scale; // >1 = farther
      }
      const expectedSpan = baseSpan / depthRatio;

      // IMPORTANT FIX:
      // If gaze is missing, be EVEN MORE willing to count head-down.
      // (This helps when eyes are occluded / not trackable.)
      const factor = hasGaze ? 0.90 : 1.02;
      if(curSpan < expectedSpan * factor){
        downByHead = true;
      }
    }

    // Backup "chin drop" heuristic (more sensitive when gaze is missing)
    if(!downByHead && headCalib && chin && typeof headCalib.chinY === "number"){
      const dy = chin.y - headCalib.chinY;
      const hPx = h;
      const baseThresh      = hPx * 0.015;
      const relaxedNoGaze   = hPx * 0.006;

      if(dy > baseThresh) {
        downByHead = true;
      } else if(!hasGaze && dy > relaxedNoGaze) {
        downByHead = true;
      }
    }

    const lookingStraight = centerHoriz && !downByHead;

    if(lookingStraight && hasGaze){
      horizRegion = gazeHoriz;
      isDown      = gazeDown;
    } else {
      if(leftHoriz)        horizRegion = "left";
      else if(centerHoriz) horizRegion = "center";
      else if(rightHoriz)  horizRegion = "right";

      // Combine head + gaze for "down" classification:
      if(downByHead){
        if(hasGaze){
          isDown = gazeDown && downByHead;
        } else {
          // KEY behavior: NO GAZE → trust head-down alone
          isDown = true;
        }
      } else if(hasGaze && gazeDown){
        isDown = true;
      }
    }
  } else if(hasGaze){
    horizRegion = gazeHoriz;
    isDown      = gazeDown;
  } else {
    return false;
  }

  if(horizRegion) regionTimes[horizRegion] += dt;
  if(isDown)      regionTimes.down       += dt;

  updateMetricsPanel();
  return isDown;
}

/* ---------- Main overlay render (gaze + regions) ---------- */
function drawFrame(result){
  const cssW = overlay.clientWidth, cssH = overlay.clientHeight;
  octx.setTransform(window.devicePixelRatio||1,0,0,window.devicePixelRatio||1,0,0);
  octx.clearRect(0,0,cssW,cssH);

  // We "process" tracking in two cases:
  // - overlaysEnabled (normal mode) OR
  // - teleprompterActive (hidden visuals, but tracking runs)
  const trackingOn = overlaysEnabled || teleprompterActive;
  const drawOverlayVisuals = overlaysEnabled && !teleprompterActive;

  if(!trackingOn) return;

  const rect = videoRectCSS();
  const hasFace = !!(result && result.faceLandmarks && result.faceLandmarks.length);
  if(!hasFace){
    // No face → cue logic should relax
    updateLookUpCue(false);
    return;
  }

  const lm = result.faceLandmarks[0].map(p => mapToCSS(p, rect));
  const chinPt = lm[CHIN_INDEX];
  const foreheadPt = lm[FOREHEAD_INDEX];

  const head = computeHeadPose(lm);
  if(prevHeadMid){
    const v=Math.hypot(head.mid.x-prevHeadMid.x,head.mid.y-prevHeadMid.y);
    headVelEMA=0.8*headVelEMA+0.2*v;
  }
  prevHeadMid=head.mid;

  if(drawOverlayVisuals){
    drawFaceMeshDots(lm);
  }

  const rightEAR = (()=>{ const horiz=dist2D(lm[R_CORNER_OUT],lm[R_CORNER_IN]); const vert=dist2D(lm[R_LID_UP],lm[R_LID_DN]); return horiz>0?vert/horiz:0; })();
  const leftEAR  = (()=>{ const horiz=dist2D(lm[L_CORNER_OUT],lm[L_CORNER_IN]); const vert=dist2D(lm[L_LID_UP],lm[L_LID_DN]); return horiz>0?vert/horiz:0; })();
  const rightClosed=rightEAR<EAR_CLOSED, leftClosed=leftEAR<EAR_CLOSED;

  const boxR=eyeBox(lm,RIGHT_EYE), boxL=eyeBox(lm,LEFT_EYE);

  const iris=irisPoints(lm);
  const leftPupil  = leftClosed ? null : iris.L;
  const rightPupil = rightClosed ? null : iris.R;

  if(drawOverlayVisuals){
    drawPupilsAndRings(lm, leftPupil, rightPupil);
  }

  // Depth ratio from head scale vs baseline (NEAR)
  if(baselineHead && head.scale>1e-3){
    currentDepthRatio = clamp(baselineHead.scale / head.scale, 0.5, 2.0);
  } else {
    currentDepthRatio = 1;
  }

  // Gaze estimation
  let preds=[], weights=[], cues={L:null,R:null};
  if(!leftClosed){
    const L=perEyeFeatures("L",lm,boxL); cues.L=L;
    if(modelLX&&modelLY){
      let g={x:predict(modelLX,L.feat), y:predict(modelLY,L.feat)};
      if(affineL) g=applyAffine2D(affineL,g);
      preds.push(g);
      weights.push( clamp((leftEAR-0.18)/0.20,0,1) );
    }
  }
  if(!rightClosed){
    const R=perEyeFeatures("R",lm,boxR); cues.R=R;
    if(modelRX&&modelRY){
      let g={x:predict(modelRX,R.feat), y:predict(modelRY,R.feat)};
      if(affineR) g=applyAffine2D(affineR,g);
      preds.push(g);
      weights.push( clamp((rightEAR-0.18)/0.20,0,1) );
    }
  }

  let G=null;

  if(preds.length){
    let sx=0,sy=0,sw=0;
    for(let i=0;i<preds.length;i++){
      sx+=preds[i].x*weights[i];
      sy+=preds[i].y*weights[i];
      sw+=weights[i];
    }
    let gx=clamp(sx/(sw||1),0,cssW), gy=clamp(sy/(sw||1),0,cssH);

    if(headCalib && !baselineHead){
      baselineHead = {
        mid:{x:headCalib.center.x,y:headCalib.center.y},
        angle:headCalib.angle,
        scale:headCalib.scaleNorm || head.scale
      };
    }
    if(!baselineHead) baselineHead=head;

    const Gst=stabilizePointByHead({x:gx,y:gy}, head, baselineHead);
    gx=clamp(Gst.x,0,cssW);
    gy=clamp(Gst.y,0,cssH);

    const nxAvg = ( (cues.L?cues.L.nx:0) + (cues.R?cues.R.nx:0) ) /
                  ( (cues.L?1:0)+(cues.R?1:0) || 1 );
    const nyAvg = ( (cues.L?cues.L.ny:0) + (cues.R?cues.R.ny:0) ) /
                  ( (cues.L?1:0)+(cues.R?1:0) || 1 );
    const eyeDevMag=Math.hypot(nxAvg,nyAvg);
    const headMoving=headVelEMA>HEAD_VEL_TRIG;

    if(headMoving && eyeDevMag<0.22){
      if(prevGaze){
        gx=lerp(gx,prevGaze.x,0.95);
        gy=lerp(gy,prevGaze.y,0.95);
      }
      clutchFrames=CLUTCH_HOLD_FRAMES;
    } else if(clutchFrames>0 && eyeDevMag<(0.22*1.15)){
      if(prevGaze){
        gx=lerp(gx,prevGaze.x,0.90);
        gy=lerp(gy,prevGaze.y,0.90);
      }
      clutchFrames--;
    }

    const maxStep=(clutchFrames>0)?30:70;
    if(prevGaze){
      let dx=gx-prevGaze.x, dy=gy-prevGaze.y, step=Math.hypot(dx,dy);
      if(step>maxStep){
        const s=maxStep/step;
        gx=prevGaze.x+dx*s;
        gy=prevGaze.y+dy*s;
      }
    }
    const alpha=(clutchFrames>0)?0.10:0.18;
    if(!gazeEMA) gazeEMA={x:gx,y:gy};
    gazeEMA.x=(1-alpha)*gx+alpha*gazeEMA.x;
    gazeEMA.y=(1-alpha)*gy+alpha*gazeEMA.y;
    G={x:clamp(gazeEMA.x,0,cssW), y:clamp(gazeEMA.y,0,cssH)};
    prevGaze=G;
  }

  // Head-vs-gaze region tracking + DOWN boolean returned
  const downNow = updateRegionStats(head, G, chinPt, foreheadPt);

  // LOOK UP cue uses the SAME down signal (works even when gaze is missing)
  updateLookUpCue(!!downNow);

  if(drawOverlayVisuals){
    drawGazeRays(leftPupil,rightPupil,G);
  }
}

/* ---------- Face Centering stage (rectangle → face-outline morph) ---------- */
let centeringActive=false, centerHoldMs=0, greenFlashMs=0;
const CENTER_HOLD_REQUIRED_MS = 1800; // ≤ 2 seconds

const OUTER_RES = 512;
const TARGET_STROKE = "rgba(255,255,255,0.80)";

function showCenterStage(show){
  centeringActive = show;
  if(show){
    blackout.style.opacity = "0";
    video.classList.add("preview");
    centerStage.style.display="block";
    document.getElementById("calCenterHUD").style.display="none";
  }else{
    video.classList.remove("preview");
    centerStage.style.display="none";
  }
}

function drawCenteringOverlay(){
  if(!centeringActive) return;

  const dpr = window.devicePixelRatio || 1;
  const cssW=centerCanvas.clientWidth  || window.innerWidth;
  const cssH=centerCanvas.clientHeight || window.innerHeight;
  cctx.setTransform(dpr,0,0,dpr,0,0);
  cctx.clearRect(0,0,cssW,cssH);

  const guideW = Math.min(cssW, cssH) * 0.36;
  const guideH = guideW * 1.25;
  const cx = cssW/2;
  const cy = cssH*0.47;

  cctx.save();
  cctx.fillStyle = "rgba(0,0,0,0.55)";
  cctx.fillRect(0,0,cssW,cssH);

  cctx.globalCompositeOperation = "destination-out";
  cctx.translate(cx, cy);
  cctx.scale(1, guideH/guideW);

  const feather = guideW * 0.35;
  const innerR  = guideW/2 - feather*0.3;
  const outerR  = guideW/2 + feather;

  const grad = cctx.createRadialGradient(0,0, innerR, 0,0, outerR);
  grad.addColorStop(0.0, "rgba(0,0,0,1)");
  grad.addColorStop(0.6, "rgba(0,0,0,0.7)");
  grad.addColorStop(1.0, "rgba(0,0,0,0)");

  cctx.fillStyle = grad;
  cctx.beginPath();
  cctx.arc(0,0, outerR, 0, Math.PI*2);
  cctx.fill();

  cctx.restore();

  const rectW = guideW * 1.18;
  const rectH = guideH * 1.18;
  let targetRect = rectPolyline(cx, cy, rectW, rectH, 64);
  targetRect = resampleClosedCurve(targetRect, OUTER_RES);

  const res = lastResult;
  let faceRing=null, faceBBox=null, head=null, liveDense=null, lm2=null;
  if(res?.faceLandmarks?.length){
    const r = videoRectCSS();
    lm2 = res.faceLandmarks[0].map(p=>mapToCSS(p,r));
    faceRing = FACE_OVAL.map(i=>({x:lm2[i].x,y:lm2[i].y}));

    let minX=1e9,minY=1e9,maxX=-1e9,maxY=-1e9;
    for(const p of faceRing){
      if(p.x<minX)minX=p.x;
      if(p.y<minY)minY=p.y;
      if(p.x>maxX)maxX=p.x;
      if(p.y>maxY)maxY=p.y;
    }
    const bw=maxX-minX, bh=maxY-minY, bx=minX, by=minY;
    faceBBox = { x:bx, y:by, w:bw, h:bh, cx:bx+bw/2, cy:by+bh/2 };
    head = computeHeadPose(lm2);

    liveDense = resampleClosedCurve(faceRing, OUTER_RES);
    liveDense = chaikinSubdivide(liveDense, 1);
    liveDense = resampleClosedCurve(liveDense, OUTER_RES);

    cctx.lineWidth = 2.5;
    cctx.strokeStyle = "rgba(255,255,255,0.95)";
    drawClosedSpline(cctx, liveDense);
    cctx.stroke();
  }

  if(!faceRing){
    centerHint.textContent = "No face detected";
    centerPctEl.textContent="0%";
    centerHoldMs=0;

    cctx.lineWidth = 2.5;
    cctx.strokeStyle = TARGET_STROKE;
    drawClosedPath(cctx, targetRect);
    cctx.stroke();
    return;
  }

  const desiredH = guideH * 0.90;

  const posOff = { dx: faceBBox.cx - cx, dy: faceBBox.cy - cy };
  const sizeOff = { sx: faceBBox.w - guideW, sy: faceBBox.h - desiredH };
  const angleDeg = Math.abs(head.angle*180/Math.PI);
  const tiltPenalty   = clamp((angleDeg/8),0,1);
  const centerPenalty = Math.max(
    clamp(Math.abs(posOff.dx)/(guideW*0.5),0,1),
    clamp(Math.abs(posOff.dy)/(guideH*0.5),0,1)
  );
  const scalePenalty  = clamp(Math.abs(sizeOff.sy)/(desiredH*0.50),0,1);
  const bad = Math.max(tiltPenalty, centerPenalty, scalePenalty);
  const goodScore = clamp(1 - bad, 0, 1);

  const stillV = headVelEMA<1.2 ? 1 : clamp(1.6/headVelEMA, 0, 1);
  const increment = 16 * goodScore * stillV;
  if(goodScore>0.75 && stillV>0.6){
    centerHoldMs = Math.min(centerHoldMs + increment, CENTER_HOLD_REQUIRED_MS);
  }else{
    centerHoldMs = Math.max(0, centerHoldMs - 24);
  }
  const tMorph = clamp(centerHoldMs/CENTER_HOLD_REQUIRED_MS, 0, 1);
  const pct = Math.round(tMorph*100);
  centerPctEl.textContent = pct + "%";

  let hint="";
  if(centerPenalty>0.25){
    hint += (faceBBox.cx<cx? "Move right. " : "Move left. ");
    hint += (faceBBox.cy<cy? "Lower a bit. " : "Raise a bit. ");
  }
  if(scalePenalty>0.25){
    hint += (faceBBox.h<desiredH? "Move closer. " : "Move back. ");
  }
  if(tiltPenalty>0.25){
    hint += "Level your head. ";
  }
  centerHint.textContent = hint.trim();

  cctx.lineWidth = 2.0;
  cctx.strokeStyle = TARGET_STROKE;
  if(liveDense){
    const rectDense = targetRect;
    const targetMorph = morphPolylines(rectDense, liveDense, tMorph);

    if(tMorph < 0.25){
      drawClosedPath(cctx, targetMorph);
      cctx.stroke();
    } else {
      drawClosedSpline(cctx, targetMorph);
      cctx.stroke();
    }
  } else {
    drawClosedPath(cctx, targetRect);
    cctx.stroke();
  }

  if(centerHoldMs >= CENTER_HOLD_REQUIRED_MS){
    if(greenFlashMs === 0) greenFlashMs = performance.now();
    const phase = (performance.now() - greenFlashMs);
    const blink = 0.5 + 0.5*Math.sin(phase*0.02);

    cctx.save();
    cctx.strokeStyle = `rgba(34,197,94,${0.35 + 0.4*blink})`;
    cctx.lineWidth = 8;
    if(liveDense){ drawClosedSpline(cctx, liveDense); cctx.stroke(); }
    cctx.restore();

    if(phase > 600){
      const r = videoRectCSS();
      const lmSnap = lastResult.faceLandmarks[0].map(p=>mapToCSS(p,r));
      const headNow = computeHeadPose(lmSnap);
      const chinNow = lmSnap[CHIN_INDEX];
      const foreheadNow = lmSnap[FOREHEAD_INDEX];

      const faceBBoxMidY = faceBBox.cy + faceBBox.h/2;
      const center = { x: faceBBox.cx, y: faceBBox.cy };
      const chinY = chinNow ? chinNow.y : faceBBoxMidY;
      const foreheadY = foreheadNow ? foreheadNow.y : (faceBBox.cy - faceBBox.h/2);
      const faceSpan = chinY - foreheadY;

      headCalib = {
        center,
        scaleH: faceBBox.h,
        angle: headNow.angle,
        scaleNorm: headNow.scale,
        chinY,
        foreheadY,
        chinOffset: chinY - center.y,
        faceSpan
      };
      requestAnimationFrame(()=>transitionToDots());
    }
  } else {
    greenFlashMs = 0;
  }
}

/* ---------- Transition to dots ---------- */
function transitionToDots(){
  overlaysEnabled = false;
  blackout.style.opacity="1";
  showCenterStage(false);
  setTimeout(()=>{ startDotCalibration(); }, 900);
}

/* ---------- Dot hint helpers ---------- */
function ensureDotHintEl(){
  if(dotHintEl) return;
  dotHintEl = document.createElement("div");
  dotHintEl.id = "dotHint";
  dotHintEl.style.position = "absolute";
  dotHintEl.style.padding = "6px 10px";
  dotHintEl.style.borderRadius = "8px";
  dotHintEl.style.background = "rgba(0,0,0,0.75)";
  dotHintEl.style.color = "#f9fafb";
  dotHintEl.style.fontSize = "13px";
  dotHintEl.style.pointerEvents = "none";
  dotHintEl.style.opacity = "0";
  dotHintEl.style.transition = "opacity 250ms ease";
  dotHintEl.style.whiteSpace = "nowrap";
  dotHintEl.style.zIndex = "11005";
  calLayer.appendChild(dotHintEl);
}

function showDotHint(text, nx, ny){
  ensureDotHintEl();
  dotHintEl.textContent = text;

  dotHintEl.style.opacity = "0";
  dotHintEl.style.visibility = "hidden";
  dotHintEl.style.left = "0px";
  dotHintEl.style.top  = "0px";
  dotHintEl.style.transform = "none";

  const layerRect = calLayer.getBoundingClientRect();
  const dotRect   = calTarget.getBoundingClientRect();
  dotHintEl.getBoundingClientRect();

  const cx = dotRect.left + dotRect.width/2;
  const cy = dotRect.top  + dotRect.height/2;
  const margin = 12;

  let targetX, targetY, transform;

  const isTopCenter    = Math.abs(nx - 0.5) < 0.15 && ny < 0.33;
  const isBottomCenter = Math.abs(nx - 0.5) < 0.15 && ny > 0.66;

  if(isTopCenter){
    targetX   = cx - layerRect.left;
    targetY   = dotRect.bottom + margin - layerRect.top;
    transform = "translate(-50%, 0)";
  } else if(isBottomCenter){
    targetX   = cx - layerRect.left;
    targetY   = dotRect.top - margin - layerRect.top;
    transform = "translate(-50%, -100%)";
  } else if(nx < 0.5){
    targetX   = dotRect.right + margin - layerRect.left;
    targetY   = cy - layerRect.top;
    transform = "translate(0, -50%)";
  } else {
    targetX   = dotRect.left - margin - layerRect.left;
    targetY   = cy - layerRect.top;
    transform = "translate(-100%, -50%)";
  }

  dotHintEl.style.left = `${targetX}px`;
  dotHintEl.style.top  = `${targetY}px`;
  dotHintEl.style.transform = transform;

  dotHintEl.style.visibility = "visible";
  dotHintEl.style.opacity = "1";
}

function hideDotHint(){
  if(!dotHintEl) return;
  dotHintEl.style.opacity = "0";
}

/* ---------- Dot Calibration flow (two passes) ---------- */
startCalBtn.addEventListener("click", () => runCalibrationImmediate());
primeStartBtn.addEventListener("click", () => runCalibrationImmediate());

const EDGE=0.01;
const CENTER=[0.50,0.50];
const CAL_PATH=[
  CENTER,
  [EDGE,0.50],[1-EDGE,0.50],[0.50,EDGE],[0.50,1-EDGE],
  [EDGE,EDGE],[1-EDGE,EDGE],[1-EDGE,1-EDGE],[EDGE,1-EDGE],
  CENTER
];
const MOVE_MS=1500, DWELL_MS=1800, READY_MS=3000, MIN_KEEP=12;

async function runCalibrationImmediate(){
  if(isCalibrating) return;

  // If teleprompter is active, exit cleanly before recalibrating
  if(teleprompterActive){
    exitTeleprompterMode(true);
  }

  stopSpeechRecognition();

  resetLookUpCueState();

  await goFullscreen();
  topBar.classList.add("hidden");
  bottomBar.classList.add("hidden");

  calLayer.style.display="block";
  calLayer.setAttribute("aria-hidden","false");

  isCalibrating=true;
  overlaysEnabled=false;
  statusEl.textContent="Calibrating…";

  blackout.style.opacity="0";
  primeStartBtn.style.display="none";
  calTarget.style.display="none";
  document.getElementById("calCenterHUD").style.display="none";
  showCenterStage(true);
  centerHoldMs = 0;
}

/* Shared calibration buffers */
let calBufs;
function allocCalBuffers(){
  calBufs = {
    XL:[], YLx:[], YLy:[], PL:[], TL:[],
    XR:[], YRx:[], YRy:[], PR:[], TR:[]
  };
  return calBufs;
}

/* One full 10-dot pass at currentCalStage ("near" or "far") */
async function runDotPass(){
  const {XL,YLx,YLy,XR,YRx,YRy,PL,TL,PR,TR} = calBufs;

  calProgress.textContent = `1 / ${CAL_PATH.length}`;

  {
    const [nx0,ny0]=CAL_PATH[0];
    const hintState = { pulses:0 };
    await dwellAndSample(
      DWELL_MS,true,
      XL,YLx,YLy,XR,YRx,YRy,
      nx0,ny0,PL,TL,PR,TR,
      hintState,true
    );
  }

  for(let i=1;i<CAL_PATH.length && isCalibrating;i++){
    const [nx,ny]=CAL_PATH[i];
    await tweenDotTo(nx,ny,MOVE_MS);
    calProgress.textContent=`${i+1} / ${CAL_PATH.length}`;

    const hintState = { pulses:0 };
    await dwellAndSample(
      DWELL_MS,true,
      XL,YLx,YLy,XR,YRx,YRy,
      nx,ny,PL,TL,PR,TR,
      hintState,true
    );
  }
}

/* Wait for user to move back far enough before PASS 2 */
async function waitForFarDistance(nearMeanScale){
  const THRESH_RATIO = 1.40;
  const MIN_HOLD_MS = 700;
  const TIMEOUT_MS  = 20000;

  let start = performance.now();
  let okSince = null;

  while(isCalibrating && performance.now() - start < TIMEOUT_MS){
    await nextAnimationFrame();

    const res = lastResult;
    if(!res?.faceLandmarks?.length){
      okSince = null;
      continue;
    }
    const rect = videoRectCSS();
    const lm = res.faceLandmarks[0].map(p=>mapToCSS(p,rect));
    const headPose = computeHeadPose(lm);
    const scale = headPose.scale;

    if(scale <= 0) { okSince = null; continue; }

    const depthRatio = nearMeanScale / scale;

    if(depthRatio >= THRESH_RATIO){
      if(okSince === null) okSince = performance.now();
      if(performance.now() - okSince >= MIN_HOLD_MS){
        break;
      }
    } else {
      okSince = null;
    }
  }
}

function startDotCalibration(){
  if(!isCalibrating) return;

  calHUD.style.display = "block";
  void calHUD.offsetWidth;
  calHUD.style.opacity = "1";

  nearScales = [];
  farScales  = [];
  currentCalStage = "near";

  calInstruction.textContent = "Follow the dot with your eyes only.";
  calProgress.textContent = `0 / ${CAL_PATH.length}`;

  const [nx0,ny0]=CAL_PATH[0];
  placeDot(nx0,ny0,true,true);

  setTimeout(()=>runDotSequence(), READY_MS);
}

async function runDotSequence(){
  allocCalBuffers();

  currentCalStage = "near";
  calInstruction.textContent = "Follow the dot with your eyes only.";

  await runDotPass();
  if(!isCalibrating) return;

  const nearMeanScale = mean(nearScales);

  currentCalStage = "far";
  calInstruction.textContent =
    "Take a couple steps back and look at the dot.";

  const [nx0,ny0] = CENTER;
  placeDot(nx0,ny0,true,true);
  calProgress.textContent = `0 / ${CAL_PATH.length}`;

  if(nearMeanScale > 0){
    await waitForFarDistance(nearMeanScale);
  }
  if(!isCalibrating) return;

  calInstruction.textContent = "Follow the dot with your eyes only.";
  await new Promise(r => setTimeout(r, READY_MS));
  if(!isCalibrating) return;

  await runDotPass();
  if(!isCalibrating) return;

  const {XL,YLx,YLy,XR,YRx,YRy,PL,TL,PR,TR} = calBufs;

  try{
    if(XL.length>=6){
      modelLX=fitRidge(XL,YLx,1e-3);
      modelLY=fitRidge(XL,YLy,1e-3);
    } else {
      modelLX=modelLY=null;
    }
    if(XR.length>=6){
      modelRX=fitRidge(XR,YRx,1e-3);
      modelRY=fitRidge(XR,YRy,1e-3);
    } else {
      modelRX=modelRY=null;
    }

    if(modelLX&&modelLY&&PL.length>=6&&TL.length===PL.length){
      affineL=fitAffine2D(PL,TL,1e-6);
    } else {
      affineL=null;
    }
    if(modelRX&&modelRY&&PR.length>=6&&TR.length===PR.length){
      affineR=fitAffine2D(PR,TR,1e-6);
    } else {
      affineR=null;
    }
    if(!modelLX && !modelRX) throw new Error("Insufficient samples");
  }catch{
    stopCalibration("Calibration failed — not enough clean samples.");
    return;
  }

  const nearMean = nearScales.length ? mean(nearScales) : (baselineHead?.scale || 1);
  const farMean  = farScales.length ? mean(farScales)  : null;
  depthCalib = { nearScale:nearMean, farScale:farMean };

  if(lastResult?.faceLandmarks?.length){
    const rect=videoRectCSS();
    const lm = lastResult.faceLandmarks[0].map(p=>mapToCSS(p,rect));
    baselineHead = computeHeadPose(lm);

    if(headCalib){
      baselineHead.mid.x = lerp(baselineHead.mid.x, headCalib.center.x, 0.6);
      baselineHead.mid.y = lerp(baselineHead.mid.y, headCalib.center.y, 0.6);
      baselineHead.angle = lerp(baselineHead.angle, headCalib.angle, 0.6);
    }

    baselineHead.scale = nearMean;
  }

  hasCalibrated = true;
  stopCalibration("Calibration ✓");
}

/* ---------- Stop calibration & reveal overlays ---------- */
function stopCalibration(msg){
  isCalibrating=false;
  setTargetPct(0);
  calTarget.classList.remove("show");
  hideDotHint();
  resetLookUpCueState();

  if(msg) statusEl.textContent=msg;

  blackout.style.opacity="0";
  calLayer.style.display="none";
  calLayer.setAttribute("aria-hidden","true");

  topBar.classList.remove("hidden");

  if(hasCalibrated){
    bottomBar.classList.add("hidden");

    // Only show overlay visuals if NOT in teleprompter mode
    overlaysEnabled = !teleprompterActive;

    startCalBtn.classList.remove("hidden");
    startCalBtn.textContent = "Recalibrate";
    if(tpBtn) tpBtn.classList.remove("hidden");

    resetRegionStats();
  } else {
    bottomBar.classList.remove("hidden");
    overlaysEnabled = false;
  }

  resizeCanvasToCSS();
}

/* ---------- Sampling during dwell ---------- */
async function dwellAndSample(
  ms, collect,
  XL,YLx,YLy, XR,YRx,YRy,
  nx,ny, PL=[],TL=[], PR=[],TR=[],
  hintState={ pulses:0 }, isRoot=true
){
  const t0=performance.now();
  let keepL=0, keepR=0;
  while(performance.now()-t0 < ms && isCalibrating){
    await nextAnimationFrame();
    setTargetPct( clamp((performance.now()-t0)/ms,0,1) );
    const res=lastResult;
    if(!res?.faceLandmarks?.length) continue;
    const rect=videoRectCSS();
    const lm = res.faceLandmarks[0].map(p=>mapToCSS(p,rect));

    const headPose = computeHeadPose(lm);
    const headScale = headPose.scale;
    if(currentCalStage === "near") nearScales.push(headScale);
    else if(currentCalStage === "far") farScales.push(headScale);

    const leftEAR = (()=>{ const horiz=dist2D(lm[L_CORNER_OUT],lm[L_CORNER_IN]); const vert=dist2D(lm[L_LID_UP],lm[L_LID_DN]); return horiz>0?vert/horiz:0; })();
    const rightEAR= (()=>{ const horiz=dist2D(lm[R_CORNER_OUT],lm[R_CORNER_IN]); const vert=dist2D(lm[R_LID_UP],lm[R_LID_DN]); return horiz>0?vert/horiz:0; })();
    const leftHalf=leftEAR<EAR_HALF_BLINK, rightHalf=rightEAR<EAR_HALF_BLINK;

    const boxL=eyeBox(lm,LEFT_EYE), boxR=eyeBox(lm,RIGHT_EYE);

    const tx = nx * overlay.clientWidth, ty = ny * overlay.clientHeight;

    if(collect){
      if(!leftHalf){
        const L=perEyeFeatures("L",lm,boxL);
        if(Math.hypot(L.nx,L.ny)<=1.6){
          XL.push(L.feat); YLx.push(tx); YLy.push(ty); keepL++;
          if(modelLX&&modelLY){
            PL.push({x:predict(modelLX,L.feat), y:predict(modelLY,L.feat)});
            TL.push({x:tx,y:ty});
          }
        }
      }
      if(!rightHalf){
        const R=perEyeFeatures("R",lm,boxR);
        if(Math.hypot(R.nx,R.ny)<=1.6){
          XR.push(R.feat); YRx.push(tx); YRy.push(ty); keepR++;
          if(modelRX&&modelRY){
            PR.push({x:predict(modelRX,R.feat), y:predict(modelRY,R.feat)});
            TR.push({x:tx,y:ty});
          }
        }
      }
    }
  }

  if(collect && (keepL<MIN_KEEP/3 || keepR<MIN_KEEP/3)){
    hintState.pulses++;
    if(hintState.pulses >= 3){
      showDotHint("Open Your Eyes!", nx, ny);
    }
    await dwellAndSample(
      700, collect,
      XL,YLx,YLy,XR,YRx,YRy,
      nx,ny,PL,TL,PR,TR,
      hintState,false
    );
  }

  if(isRoot){
    hideDotHint();
  }
}

/* ---------- Helpers ---------- */
function placeDot(nx,ny,show=false,fade=false){
  calTarget.style.left=`${nx*100}%`;
  calTarget.style.top =`${ny*100}%`;
  if(show){
    calTarget.style.display="block";
    if(fade){
      calTarget.classList.remove("show");
      requestAnimationFrame(()=>calTarget.classList.add("show"));
    }else{
      calTarget.classList.add("show");
    }
  }
}
function setTargetPct(p){
  calTarget.querySelector(".inner").style.setProperty("--pct", String(p));
}
function nextAnimationFrame(){ return new Promise(r=>requestAnimationFrame(()=>r())); }
function easeInOutCubic(x){
  return x<0.5 ? 4*x*x*x : 1 - Math.pow(-2*x+2,3)/2;
}
async function tweenDotTo(nx,ny,durationMs){
  const startLeft=parseFloat(calTarget.style.left)||0;
  const startTop =parseFloat(calTarget.style.top)||0;
  const sx=startLeft/100, sy=startTop/100, ex=nx, ey=ny;
  const t0=performance.now();
  let t;
  do{
    await nextAnimationFrame();
    t=(performance.now()-t0)/durationMs;
    const k=easeInOutCubic(clamp(t,0,1));
    const x=sx+(ex-sx)*k, y=sy+(ey-sy)*k;
    calTarget.style.left=`${x*100}%`;
    calTarget.style.top =`${y*100}%`;
  } while(t<1 && isCalibrating);
}

/* ============================================================
   TELEPROMPTER: UI + parsing + wheel + speech autoadvance
   ============================================================ */

function injectTeleprompterStyles(){
  // Styles are defined in style.css — nothing to inject.
}

function setupTeleprompterUI(){
  // Wire up existing HTML elements — no DOM creation needed.
  tpBtn            = document.getElementById("notesBtn");
  notesModal       = document.getElementById("notesModal");
  notesTextarea    = document.getElementById("notesTextarea");
  notesDoneBtn     = document.getElementById("notesDoneBtn");
  notesCancelBtn   = document.getElementById("notesCancelBtn");
  teleprompterEl   = document.getElementById("teleprompter");
  tpContentEl      = document.getElementById("teleprompterContent");
  tpExitBtn        = document.getElementById("tpExitBtn");
  exitPrompterBtnEl = document.getElementById("exitPrompterBtn");
  voiceScrollBtnEl  = document.getElementById("voiceScrollBtn");

  const notesClearBtn = document.getElementById("notesClearBtn");

  // Teleprompter button
  tpBtn.addEventListener("click", ()=>{
    if(isCalibrating) return;
    if(!hasCalibrated){
      statusEl.textContent = "Calibrate first, then open teleprompter.";
      return;
    }
    openNotesModal();
  });

  // Notes modal buttons
  notesCancelBtn.addEventListener("click", closeNotesModal);
  notesDoneBtn.addEventListener("click", ()=>{
    const t = (notesTextarea.value || "").trim();
    if(!t){ statusEl.textContent = "Paste some notes first."; return; }
    closeNotesModal();
    applyNotesToTeleprompter(t);
    enterTeleprompterMode();
  });
  notesClearBtn.addEventListener("click", ()=>{
    notesTextarea.value = "";
    notesText = "";
  });
  notesModal.addEventListener("click", (e)=>{
    if(e.target === notesModal) closeNotesModal();
  });

  // Exit teleprompter buttons
  tpExitBtn.addEventListener("click", ()=>exitTeleprompterMode(false));
  exitPrompterBtnEl.addEventListener("click", ()=>exitTeleprompterMode(false));

  // Voice scroll toggle
  voiceScrollBtnEl.addEventListener("click", ()=>{
    voiceScrollEnabled = !voiceScrollEnabled;
    voiceScrollBtnEl.textContent = `Voice Scroll: ${voiceScrollEnabled ? "On" : "Off"}`;
    if(!voiceScrollEnabled) stopSpeechRecognition();
    else if(teleprompterActive) startSpeechRecognition();
  });

  initSpeechRecognition();

  // Keyboard controls
  document.addEventListener("keydown", (e)=>{
    if(notesModalOpen){
      if(e.key === "Escape"){ e.preventDefault(); closeNotesModal(); }
      return;
    }
    if(teleprompterActive){
      if(e.key === "Escape"){ e.preventDefault(); exitTeleprompterMode(false); return; }
      if(e.key === " " || e.key === "ArrowDown"){ e.preventDefault(); advanceLine(); }
      if(e.key === "ArrowUp"){ e.preventDefault(); goBackLine(); }
    }
  });
}

function openNotesModal(){
  if(!notesModal) return;
  notesModalOpen = true;
  notesModal.classList.add("active");
  notesModal.removeAttribute("aria-hidden");
  notesTextarea.value = notesText || "";
  setTimeout(()=>notesTextarea.focus(), 50);
}

function closeNotesModal(){
  if(!notesModal) return;
  notesModalOpen = false;
  notesModal.classList.remove("active");
  notesModal.setAttribute("aria-hidden", "true");
  notesText = notesTextarea.value || notesText;
}

/* ---------- Teleprompter text processing ---------- */
function splitIntoDisplayLines(text){
  const raw = (text || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const baseLines = raw.split("\n").map(s=>s.trim()).filter(Boolean);

  // Wrap super-long lines so they still fit as single "teleprompter lines"
  const out = [];
  const MAX_CHARS = 120;

  for(const line of baseLines){
    if(line.length <= MAX_CHARS){
      out.push(line);
      continue;
    }
    const words = line.split(/\s+/);
    let cur = "";
    for(const w of words){
      const next = cur ? (cur + " " + w) : w;
      if(next.length > MAX_CHARS && cur){
        out.push(cur);
        cur = w;
      } else {
        cur = next;
      }
    }
    if(cur) out.push(cur);
  }
  return out;
}

function tokenize(text){
  return (text || "")
    .toLowerCase()
    .replace(/[\u2019’]/g, "'")
    .replace(/[^a-z0-9'\s]/g, " ")
    .split(/\s+/)
    .map(s=>s.trim())
    .filter(Boolean);
}

function tokenMatch(spokenTok, expectedTok){
  if(!spokenTok || !expectedTok) return false;
  if(spokenTok === expectedTok) return true;

  // Let a slightly shorter token match the start of the longer one (helps ASR variants)
  const a = spokenTok;
  const b = expectedTok;
  const minLen = Math.min(a.length, b.length);

  if(minLen >= 4){
    if(a.slice(0,4) === b.slice(0,4)) return true;
  }

  // Prefix tolerance for longer words
  if(a.length >= 6 && b.length >= 6){
    if(a.startsWith(b.slice(0,5)) || b.startsWith(a.slice(0,5))) return true;
  }

  return false;
}

/* ---------- Wheel rendering ---------- */
function buildTeleprompterDOM(lines){
  if(!tpContentEl) return;

  tpLines = lines.slice();
  tpTokens = tpLines.map(t => tokenize(t));

  // For long lines, match on “important” words (less fragile)
  tpMatchTokens = tpTokens.map(tokens => {
    if(tokens.length >= 7){
      const filtered = tokens.filter(t => !STOPWORDS.has(t));
      if(filtered.length >= 3) return filtered;
    }
    return tokens;
  });

  const wrap = document.createElement("div");
  wrap.className = "tp-wheelWrap";

  const wheel = document.createElement("div");
  wheel.className = "tp-wheel";

  const prev = document.createElement("div");
  prev.className = "tp-line tp-prev";

  const cur = document.createElement("div");
  cur.className = "tp-line tp-current";

  const next = document.createElement("div");
  next.className = "tp-line tp-next";

  wheel.append(prev, cur, next);
  wrap.appendChild(wheel);

  tpContentEl.replaceChildren(wrap);

  tpWheelWrapEl = wrap;
  tpWheelEl = wheel;
  tpPrevEl = prev;
  tpCurEl = cur;
  tpNextEl = next;
}

function findFirstSpeakableLine(){
  for(let i=0;i<tpMatchTokens.length;i++){
    if(tpMatchTokens[i] && tpMatchTokens[i].length) return i;
  }
  return 0;
}

function setWheelText(el, txt){
  if(!el) return;
  const t = (txt ?? "").trim();
  el.textContent = t.length ? txt : " ";
}

function findNextSpeakableIndex(fromIdx){
  for(let i=fromIdx+1; i<tpMatchTokens.length; i++){
    if(tpMatchTokens[i] && tpMatchTokens[i].length) return i;
  }
  return -1;
}

function findPrevSpeakableIndex(fromIdx){
  for(let i=fromIdx-1; i>=0; i--){
    if(tpMatchTokens[i] && tpMatchTokens[i].length) return i;
  }
  return -1;
}

function renderWheel(){
  if(!tpPrevEl || !tpCurEl || !tpNextEl) return;

  const prevIdx = findPrevSpeakableIndex(currentLineIndex);
  const nextIdx = findNextSpeakableIndex(currentLineIndex);

  setWheelText(tpPrevEl, prevIdx >= 0 ? tpLines[prevIdx] : "");
  setWheelText(tpCurEl, tpLines[currentLineIndex] ?? "");
  setWheelText(tpNextEl, nextIdx >= 0 ? tpLines[nextIdx] : "");
}

function setCurrentLine(idx){
  currentLineIndex = clamp(idx, 0, Math.max(0, tpLines.length-1));
  currentTokenPos = 0;
  renderWheel();
}

function animateToLine(targetIdx, dir){
  if(tpAnimating || !tpWheelEl) return;
  tpAnimating = true;

  renderWheel();

  tpWheelEl.classList.remove("shift-up","shift-down");
  void tpWheelEl.offsetWidth;

  tpWheelEl.classList.add(dir > 0 ? "shift-up" : "shift-down");

  setTimeout(()=>{
    tpWheelEl.classList.add("no-anim");
    tpWheelEl.classList.remove("shift-up","shift-down");

    setCurrentLine(targetIdx);

    void tpWheelEl.offsetWidth;
    tpWheelEl.classList.remove("no-anim");

    tpAnimating = false;
  }, TP_ANIM_MS);
}

function advanceLine(){
  const nextIdx = findNextSpeakableIndex(currentLineIndex);
  if(nextIdx < 0) return;
  animateToLine(nextIdx, +1);
}

function goBackLine(){
  const prevIdx = findPrevSpeakableIndex(currentLineIndex);
  if(prevIdx < 0) return;
  animateToLine(prevIdx, -1);
}

function applyNotesToTeleprompter(text){
  notesText = (text ?? "");
  const lines = splitIntoDisplayLines(notesText);

  buildTeleprompterDOM(lines);

  currentLineIndex = findFirstSpeakableLine();
  currentTokenPos = 0;

  renderWheel();
}

/* ---------- Enter / Exit teleprompter mode ---------- */
function enterTeleprompterMode(){
  if(!teleprompterEl) return;
  teleprompterActive = true;

  // Hide overlay visuals (but keep tracking running)
  overlaysEnabled = false;

  teleprompterEl.classList.add("active");
  teleprompterEl.removeAttribute("aria-hidden");

  // Ensure LOOK UP cue stays above prompter overlay
  if(lookUpCue) lookUpCue.style.zIndex = "15000";

  // Swap top-bar buttons
  if(tpBtn) tpBtn.classList.add("hidden");
  if(exitPrompterBtnEl) exitPrompterBtnEl.classList.remove("hidden");
  if(voiceScrollBtnEl)  voiceScrollBtnEl.classList.remove("hidden");

  if(voiceScrollEnabled) startSpeechRecognition();
  statusEl.textContent = voiceScrollEnabled && speechSupported
    ? "Teleprompter active — speak to advance (or Space/↓)."
    : "Teleprompter active — use Space/↓ to advance.";
}

function exitTeleprompterMode(silent){
  teleprompterActive = false;
  if(teleprompterEl){
    teleprompterEl.classList.remove("active");
    teleprompterEl.setAttribute("aria-hidden", "true");
  }

  stopSpeechRecognition();

  // Restore overlay visuals if calibrated
  overlaysEnabled = !!hasCalibrated;

  // Swap top-bar buttons back
  if(tpBtn) tpBtn.classList.remove("hidden");
  if(exitPrompterBtnEl) exitPrompterBtnEl.classList.add("hidden");
  if(voiceScrollBtnEl)  voiceScrollBtnEl.classList.add("hidden");

  if(!silent) statusEl.textContent = "Exited teleprompter.";
}

/* ---------- Speech Recognition ---------- */
function initSpeechRecognition(){
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if(!SR){
    speechSupported = false;
    return;
  }
  speechSupported = true;

  recognition = new SR();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = "en-US";

  recognition.onstart = ()=>{
    recognitionRunning = true;
  };

  recognition.onerror = (e)=>{
    recognitionRunning = false;
    if(teleprompterActive){
      statusEl.textContent = `Mic error: ${e?.error || "unknown"}`;
    }
  };

  recognition.onend = ()=>{
    recognitionRunning = false;
    // Keep it running while teleprompter is active
    if(teleprompterActive){
      try{
        recognition.start();
      }catch{}
    }
  };

  recognition.onresult = (event)=>{
    let finalText = "";
    let interimText = "";

    for(let i=event.resultIndex; i<event.results.length; i++){
      const res = event.results[i];
      const t = (res?.[0]?.transcript || "").trim();
      if(!t) continue;

      if(res.isFinal) finalText += " " + t;
      else interimText = t; // latest interim
    }

    const spoken = (finalText.trim() || interimText.trim());
    if(!spoken) return;

    // Debug: confirm SpeechRecognition is actually producing text
    if(teleprompterActive && statusEl){
      const short = spoken.length > 60 ? spoken.slice(0,60) + "…" : spoken;
      statusEl.textContent = `Heard: “${short}”`;
    }

    processSpokenText(spoken, !!finalText.trim());
  };
}

function startSpeechRecognition(){
  if(!speechSupported || !recognition) return;
  if(recognitionRunning) return;

  try{
    recognition.start();
  }catch{
    // Some browsers throw if called twice quickly
  }
}

function stopSpeechRecognition(){
  if(!speechSupported || !recognition) return;
  try{
    recognition.stop();
  }catch{}
  recognitionRunning = false;
}

/* Match spoken words against current line, advance when complete */
function processSpokenText(spoken, isFinal){
  if(!teleprompterActive || !voiceScrollEnabled) return;
  if(!tpMatchTokens.length) return;

  const expected = tpMatchTokens[currentLineIndex] || [];
  if(!expected.length) return;

  const spokenTokens = tokenize(spoken);
  if(!spokenTokens.length) return;

  // Match spoken tokens against expected tokens IN ORDER
  let pos = currentTokenPos;

  for(const st of spokenTokens){
    if(pos >= expected.length) break;

    if(tokenMatch(st, expected[pos])){
      pos++;
      continue;
    }

    // small lookahead helps when recognition drops a word
    if((pos + 1) < expected.length && tokenMatch(st, expected[pos + 1])){
      pos += 2;
      continue;
    }
    if((pos + 2) < expected.length && tokenMatch(st, expected[pos + 2])){
      pos += 3;
      continue;
    }
  }

  if(pos > currentTokenPos) currentTokenPos = pos;

  // ✅ Only advance when we’ve matched ALL expected tokens
  if(currentTokenPos >= expected.length){
    advanceLine();
    return;
  }

  // Optional: if final result is very close on a long line, let it pass
  if(isFinal && expected.length >= 10){
    const ratio = currentTokenPos / expected.length;
    if(ratio >= 0.92){
      advanceLine();
    }
  }
}

/* ---------- Boot ---------- */
(function start(){
  window.__appStarted = true;
  (async ()=>{
    try{
      setupTeleprompterUI();

      // Load mediapipe via a classic <script> tag. vision_bundle.js is a
      // CommonJS module — it assigns to `module.exports`. Browsers don't
      // define `module`, so without a shim the assignment throws at runtime
      // and no globals are created. Provide a temporary shim so the CJS
      // export lands somewhere we can read, then clean it up.
      statusEl.textContent = "Loading face detection model…";

      const _cjsShim = { exports: {} };
      window.module  = _cjsShim;           // CJS: module.exports = {...}
      window.exports = _cjsShim.exports;   // CJS: exports.Foo = ...

      try {
        await new Promise((resolve, reject) => {
          const s = document.createElement("script");
          s.src = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/vision_bundle.js";
          s.onload  = resolve;
          s.onerror = () => reject(new Error(
            "Failed to fetch face detection model from CDN — check your network and try refreshing."
          ));
          document.head.appendChild(s);
        });
      } finally {
        // Always remove the shim so it doesn't pollute later code
        delete window.module;
        delete window.exports;
      }

      // The CJS bundle assigns to module.exports; try that first, then fall
      // back to any window globals the bundle may have also set.
      const _mpExports = _cjsShim.exports;
      FaceLandmarker  = _mpExports.FaceLandmarker  || window.FaceLandmarker;
      FilesetResolver = _mpExports.FilesetResolver || window.FilesetResolver;

      if(!FaceLandmarker || !FilesetResolver){
        // Last resort: scan every key of the exports object for the classes
        for(const key of Object.keys(_mpExports)){
          const v = _mpExports[key];
          if(v && typeof v === "object"){
            if(!FaceLandmarker  && v.FaceLandmarker)  FaceLandmarker  = v.FaceLandmarker;
            if(!FilesetResolver && v.FilesetResolver) FilesetResolver = v.FilesetResolver;
          }
          if(FaceLandmarker && FilesetResolver) break;
        }
      }

      if(!FaceLandmarker || !FilesetResolver){
        const exportKeys = Object.keys(_mpExports).slice(0, 20).join(", ") || "none";
        throw new Error(
          `Mediapipe loaded but classes not found. ` +
          `module.exports keys: [${exportKeys}]. Open DevTools console for details.`
        );
      }

      await setupCamera();
      await setupFaceLandmarker();
      running=true;
      statusEl.textContent="Ready — press Start Calibration";

      topBar.classList.add("hidden");
      bottomBar.classList.add("hidden");
      calLayer.style.display="block";
      calLayer.setAttribute("aria-hidden","false");
      blackout.style.opacity="1";
      primeStartBtn.style.display="block";

      resizeCanvasToCSS();
      updateMetricsPanel();

      (function tick(){
        if(!running) return;
        const now=performance.now();
        lastResult=faceLandmarker.detectForVideo(video,now);
        drawFrame(lastResult);
        drawCenteringOverlay();
        requestAnimationFrame(tick);
      })();
    }catch(e){
      console.error(e);
      const msg = e.message || "Initialization failed — check the console for details.";
      statusEl.textContent = msg;
      alert(msg);
    }
  })();
})();
