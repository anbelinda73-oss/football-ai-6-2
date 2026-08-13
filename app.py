import os
import sqlite3
import tempfile
import threading
import time
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from ultralytics import YOLO


APP_VERSION = "6.2"
DATA_DIR = Path(os.getenv("FOOTBALL_AI_DATA_DIR", Path(tempfile.gettempdir()) / "football_ai"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "jobs.sqlite3"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))

app = FastAPI(title=f"FOOTBALL AI {APP_VERSION}")
model = YOLO(os.getenv("YOLO_MODEL", "yolo11n.pt"))
model_lock = threading.Lock()


def db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                message TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )


def set_job(job_id, status=None, progress=None, message=None):
    fields, values = ["updated_at = ?"], [time.time()]
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if progress is not None:
        fields.append("progress = ?")
        values.append(int(progress))
    if message is not None:
        fields.append("message = ?")
        values.append(message)
    values.append(job_id)
    with db() as conn:
        conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", values)


def get_job(job_id):
    with db() as conn:
        return conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()


init_db()


INDEX_HTML = r'''<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>FOOTBALL AI 6.2</title>
  <style>
    *{box-sizing:border-box} body{margin:0;background:#07140f;color:#f4fff9;font-family:system-ui,-apple-system,"Noto Sans KR",sans-serif}
    main{max-width:920px;margin:auto;padding:28px 18px 60px} h1{margin:0;font-size:clamp(30px,6vw,52px)}
    .sub{color:#9fc6b2;margin:7px 0 25px}.card{background:#10231a;border:1px solid #254c38;border-radius:18px;padding:20px;box-shadow:0 16px 40px #0005}
    .steps{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:18px}.step{padding:10px;border-radius:10px;background:#193326;color:#a7cbb8;text-align:center;font-size:14px}.step.on{background:#18b866;color:#03150b;font-weight:800}
    .pick{display:block;border:2px dashed #397b57;border-radius:14px;padding:22px;text-align:center;cursor:pointer}.pick:hover{background:#163322} input[type=file]{display:none}
    .video-wrap{display:none;position:relative;margin-top:18px;background:#000;border-radius:12px;overflow:hidden}video{display:block;width:100%;max-height:62vh}canvas{position:absolute;left:0;right:0;top:0;bottom:42px;width:100%;height:calc(100% - 42px);cursor:crosshair}
    .hint{color:#bbd5c7;margin:12px 0}.actions{display:flex;gap:10px;flex-wrap:wrap}button,.download{border:0;border-radius:11px;padding:13px 18px;font-weight:800;font-size:16px;cursor:pointer;text-decoration:none}
    button{background:#21d475;color:#042012}button:disabled{opacity:.4;cursor:not-allowed}.secondary{background:#294639;color:white}.download{display:none;background:#f6c945;color:#211b00}
    .status{display:none;margin-top:18px}.bar{height:13px;background:#294438;border-radius:99px;overflow:hidden}.fill{height:100%;width:0;background:#22d879;transition:width .3s}.msg{margin-top:9px;color:#bcd6c8}.error{color:#ff9c9c}
    @media(max-width:600px){.steps{grid-template-columns:1fr}.card{padding:14px}}
  </style>
</head>
<body><main><h1>FOOTBALL AI</h1><p class="sub">영상에서 선수를 한 번 클릭하면 자동으로 추적합니다.</p>
<section class="card">
  <div class="steps"><div class="step on" id="s1">1. 영상 선택</div><div class="step" id="s2">2. 선수 클릭</div><div class="step" id="s3">3. 자동 분석</div></div>
  <label class="pick" for="file"><strong>축구 영상을 선택하세요</strong><br><small>MP4, MOV 등 · 최대 ''' + str(MAX_UPLOAD_MB) + r'''MB</small></label><input id="file" type="file" accept="video/*">
  <div class="video-wrap" id="wrap"><video id="video" controls playsinline></video><canvas id="canvas"></canvas></div>
  <p class="hint" id="hint">영상을 선택하면 여기에서 추적할 선수를 클릭할 수 있습니다.</p>
  <div class="actions"><button id="start" disabled>분석 시작</button><button class="secondary" id="reset" type="button">다시 선택</button><a class="download" id="download">결과 영상 다운로드</a></div>
  <div class="status" id="status"><div class="bar"><div class="fill" id="fill"></div></div><div class="msg" id="msg"></div></div>
</section></main>
<script>
const file=document.querySelector('#file'), video=document.querySelector('#video'), canvas=document.querySelector('#canvas'), wrap=document.querySelector('#wrap');
const start=document.querySelector('#start'), reset=document.querySelector('#reset'), hint=document.querySelector('#hint'), statusEl=document.querySelector('#status'), fill=document.querySelector('#fill'), msg=document.querySelector('#msg'), download=document.querySelector('#download');
let point=null, objectUrl=null, busy=false;
function steps(n){for(let i=1;i<=3;i++)document.querySelector('#s'+i).classList.toggle('on',i===n)}
function draw(){const c=canvas.getContext('2d'); canvas.width=video.clientWidth*devicePixelRatio; canvas.height=video.clientHeight*devicePixelRatio; c.scale(devicePixelRatio,devicePixelRatio); c.clearRect(0,0,canvas.width,canvas.height); if(point){c.strokeStyle='#ff3030';c.lineWidth=4;c.beginPath();c.arc(point.dx,point.dy,15,0,Math.PI*2);c.stroke();c.beginPath();c.moveTo(point.dx-22,point.dy);c.lineTo(point.dx+22,point.dy);c.moveTo(point.dx,point.dy-22);c.lineTo(point.dx,point.dy+22);c.stroke()}}
file.onchange=()=>{if(!file.files[0])return;if(objectUrl)URL.revokeObjectURL(objectUrl);objectUrl=URL.createObjectURL(file.files[0]);video.src=objectUrl;wrap.style.display='block';point=null;start.disabled=true;hint.textContent='선수가 잘 보이는 장면에서 영상을 멈추고, 선수의 몸 가운데를 클릭하세요.';steps(2);download.style.display='none';statusEl.style.display='none'};
video.onloadedmetadata=draw; window.onresize=draw;
canvas.onclick=e=>{if(busy)return;const r=canvas.getBoundingClientRect();const dx=e.clientX-r.left,dy=e.clientY-r.top;point={dx,dy,x:dx/r.width*video.videoWidth,y:dy/r.height*video.videoHeight,time:video.currentTime};draw();start.disabled=false;hint.textContent=`선택 완료 (${video.currentTime.toFixed(1)}초). 이제 분석 시작을 누르세요.`};
reset.onclick=()=>{if(busy)return;file.value='';video.removeAttribute('src');wrap.style.display='none';point=null;start.disabled=true;steps(1);hint.textContent='영상을 선택하면 여기에서 추적할 선수를 클릭할 수 있습니다.';download.style.display='none';statusEl.style.display='none'};
start.onclick=async()=>{if(!point||!file.files[0])return;busy=true;start.disabled=true;steps(3);statusEl.style.display='block';fill.style.width='0%';msg.className='msg';msg.textContent='영상을 올리고 있습니다…';const fd=new FormData();fd.append('video',file.files[0]);fd.append('target_x',point.x);fd.append('target_y',point.y);fd.append('target_time',point.time);try{const r=await fetch('/analyze',{method:'POST',body:fd});const j=await r.json();if(!r.ok)throw new Error(j.detail||'업로드 실패');poll(j.job)}catch(e){fail(e.message)}};
async function poll(id){try{const r=await fetch('/status/'+id);const j=await r.json();if(!r.ok)throw new Error(j.detail||'상태 확인 실패');fill.style.width=(j.progress||0)+'%';msg.textContent=j.message||`분석 중… ${j.progress||0}%`;if(j.status==='done'){busy=false;fill.style.width='100%';msg.textContent='완료되었습니다.';download.href='/result/'+id;download.style.display='inline-block';start.disabled=false;return}if(j.status==='error')throw new Error(j.message||'분석 실패');setTimeout(()=>poll(id),1500)}catch(e){fail(e.message)}}
function fail(t){busy=false;start.disabled=false;msg.className='msg error';msg.textContent=t}
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
def root():
    return INDEX_HTML


@app.get("/health")
def health():
    return {"ok": True, "engine": "YOLO + BoT-SORT", "version": APP_VERSION}


def run_analysis(job_id: str, inp: Path, out: Path, target_x: float, target_y: float, target_time: float):
    cap = None
    writer = None
    try:
        set_job(job_id, status="processing", progress=0, message="분석을 시작합니다.")
        cap = cv2.VideoCapture(str(inp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            raise ValueError("영상 파일을 읽을 수 없습니다.")
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("결과 영상 파일을 만들 수 없습니다.")

        target_frame = int(max(0.0, target_time) * fps)
        selected, last, lost, frame_no = None, None, 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            with model_lock:
                result = model.track(frame, persist=True, tracker="botsort.yaml", classes=[0], verbose=False)[0]
            candidates = []
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.int().cpu().tolist()
                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(float, box)
                    candidates.append((track_id, x1, y1, x2, y2, (x1+x2)/2, (y1+y2)/2))
            if frame_no >= target_frame and selected is None and candidates:
                inside = [c for c in candidates if c[1] <= target_x <= c[3] and c[2] <= target_y <= c[4]]
                selected = min(inside or candidates, key=lambda c: (c[5]-target_x)**2+(c[6]-target_y)**2)[0]
            current = next((c for c in candidates if c[0] == selected), None)
            if current:
                _, x1, y1, x2, y2, cx, cy = current
                last, lost = (cx, cy), 0
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                cv2.circle(frame, (int(cx), int(cy)), 12, (0, 0, 255), 3)
                cv2.putText(frame, "26", (int(x1), max(24, int(y1)-7)), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2)
            elif selected is not None:
                lost += 1
                if last and lost < 30:
                    cv2.circle(frame, (int(last[0]), int(last[1])), 12, (0, 255, 255), 2)
            writer.write(frame)
            frame_no += 1
            if total > 0 and frame_no % max(1, int(fps)) == 0:
                progress = min(99, int(frame_no / total * 100))
                set_job(job_id, progress=progress, message=f"선수를 추적하고 있습니다… {progress}%")
        if selected is None:
            raise ValueError("선수를 찾지 못했습니다. 선수가 크게 보이는 장면에서 몸 가운데를 클릭해 주세요.")
        set_job(job_id, status="done", progress=100, message="분석이 완료되었습니다.")
    except Exception as exc:
        set_job(job_id, status="error", message=str(exc))
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        inp.unlink(missing_ok=True)


@app.post("/analyze")
async def analyze(video: UploadFile = File(...), target_x: float = Form(...), target_y: float = Form(...), target_time: float = Form(0)):
    job_id = uuid.uuid4().hex
    inp, out = DATA_DIR / f"{job_id}_input.mp4", DATA_DIR / f"{job_id}_tracked.mp4"
    size = 0
    try:
        with inp.open("wb") as stream:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(413, f"영상은 최대 {MAX_UPLOAD_MB}MB까지 올릴 수 있습니다.")
                stream.write(chunk)
    except Exception:
        inp.unlink(missing_ok=True)
        raise
    now = time.time()
    with db() as conn:
        conn.execute("INSERT INTO jobs(id,status,progress,message,created_at,updated_at) VALUES(?,?,?,?,?,?)", (job_id,"queued",0,"대기 중입니다.",now,now))
    threading.Thread(target=run_analysis, args=(job_id, inp, out, target_x, target_y, target_time), daemon=True).start()
    return {"job": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    row = get_job(job_id)
    if row is None:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    return {"status": row["status"], "progress": row["progress"], "message": row["message"]}


@app.get("/result/{job_id}")
def result(job_id: str):
    row = get_job(job_id)
    if row is None:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    if row["status"] != "done":
        raise HTTPException(409, row["message"] or "아직 분석 중입니다.")
    path = DATA_DIR / f"{job_id}_tracked.mp4"
    if not path.exists():
        raise HTTPException(404, "결과 영상 파일이 없습니다.")
    return FileResponse(path, media_type="video/mp4", filename="football_ai_26_tracked.mp4")
