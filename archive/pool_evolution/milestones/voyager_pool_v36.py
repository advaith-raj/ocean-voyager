#!/usr/bin/env python3
"""Ocean Voyager v36 unified pool autonomy.

SEARCH: configurable sequence bank; switches sequence after wall events.
SAFETY: confirmed wall contact -> stop -> bounded turn until camera view clears -> resume.
SENSORS: RGB and MLX90640 are independent OFF/MONITOR/ACTIVE producers.
TARGET: either ACTIVE sensor may create a candidate; RGB (when visible) is primary visual servo.
POSE: display-only dead reckoning (x,y,heading) with wall-anchor corrections for replay.

Coordinate convention: start at midpoint of near short edge: x=0, y=0, heading=0 deg (+Y).
GPIO preserved from v34/v35: left 17/27, right 22/23.
"""
from gpiozero import DigitalOutputDevice
from time import sleep, monotonic
from pathlib import Path
from datetime import datetime
from collections import deque
import argparse, csv, json, math, subprocess, threading
import cv2, numpy as np

ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Motors / start
ap.add_argument('--forward-left-sec',type=float,default=1.7); ap.add_argument('--forward-right-sec',type=float,default=2.0)
ap.add_argument('--forward-gap-sec',type=float,default=0.0); ap.add_argument('--start-delay-sec',type=float,default=0.0)
# Camera
ap.add_argument('--width',type=int,default=640); ap.add_argument('--height',type=int,default=480); ap.add_argument('--fps',type=int,default=10)
ap.add_argument('--max-frame-age',type=float,default=.45)
# Search sequence bank. F is calibrated sequential forward cycle; L/R are continuous one-pump pulses; O=off.
ap.add_argument('--sequences',default='F:14,L:0.35,F:12,R:0.25;F:12,R:0.35,F:15,L:0.25;F:16,L:0.25,F:10,L:0.20,F:8,R:0.45;F:10,R:0.25,F:14,L:0.40,F:8')
ap.add_argument('--sequence-max-turn-sec',type=float,default=1.0); ap.add_argument('--sequence-switch-on-cycle',action=argparse.BooleanOptionalAction,default=True)
# Wall contact + recovery
ap.add_argument('--collision-edge-max',type=float,default=.002); ap.add_argument('--collision-scale-min',type=float,default=1000.)
ap.add_argument('--collision-confirm-frames',type=int,default=3); ap.add_argument('--recovery-pause-sec',type=float,default=.35)
ap.add_argument('--recovery-turn-sec',type=float,default=.55); ap.add_argument('--recovery-max-pulses',type=int,default=5)
ap.add_argument('--open-confirm-frames',type=int,default=3); ap.add_argument('--recovery-forward-sec',type=float,default=3.0)
ap.add_argument('--default-recovery-turn',choices=['left','right'],default='left')
# Sensor modes
ap.add_argument('--vision-mode',choices=['off','monitor','active'],default='active')
ap.add_argument('--thermal-mode',choices=['off','monitor','active'],default='monitor')
# RGB
ap.add_argument('--model',default='best.pt'); ap.add_argument('--vision-conf',type=float,default=.35); ap.add_argument('--vision-imgsz',type=int,default=640)
ap.add_argument('--vision-every-frames',type=int,default=3); ap.add_argument('--vision-candidate-hits',type=int,default=1)
# Thermal
ap.add_argument('--thermal-polarity',choices=['warm','cold','both'],default='warm'); ap.add_argument('--thermal-threshold-c',type=float,default=.8)
ap.add_argument('--thermal-min-blob-pixels',type=int,default=4); ap.add_argument('--thermal-fov-x-deg',type=float,default=55.)
ap.add_argument('--thermal-rate-hz',type=float,default=2.0); ap.add_argument('--thermal-candidate-hits',type=int,default=2)
ap.add_argument('--thermal-window-sec',type=float,default=1.8); ap.add_argument('--thermal-bearing-spread-deg',type=float,default=15.)
# Candidate / target approach
ap.add_argument('--candidate-timeout-sec',type=float,default=3.0); ap.add_argument('--target-deadband',type=float,default=.14)
ap.add_argument('--target-hard-error',type=float,default=.35); ap.add_argument('--target-turn-min-sec',type=float,default=.16); ap.add_argument('--target-turn-max-sec',type=float,default=.55)
ap.add_argument('--approach-forward-min-sec',type=float,default=.30); ap.add_argument('--approach-forward-max-sec',type=float,default=1.60)
ap.add_argument('--approach-area-near',type=float,default=.07); ap.add_argument('--approach-growth-gain',type=float,default=4.0)
ap.add_argument('--target-observe-sec',type=float,default=.18); ap.add_argument('--target-lost-sec',type=float,default=1.5)
ap.add_argument('--reacquire-turn-sec',type=float,default=.30); ap.add_argument('--reacquire-attempts',type=int,default=6)
ap.add_argument('--done-area',type=float,default=.10); ap.add_argument('--done-hits',type=int,default=3)
ap.add_argument('--done-thermal-delta-c',type=float,default=2.5,help='supporting evidence only; RGB area remains primary DONE rule')
ap.add_argument('--target-done',choices=['hold','resume'],default='hold')
# Display pose / replay
ap.add_argument('--pool-short-ft',type=float,default=5.0); ap.add_argument('--pool-long-ft',type=float,default=7.0)
ap.add_argument('--display-forward-ft-s',type=float,default=.175,help='display-only; ~7 ft / 40 s initial estimate')
ap.add_argument('--display-left-deg-s',type=float,default=30.0); ap.add_argument('--display-right-deg-s',type=float,default=45.0)
ap.add_argument('--wall-anchor',action=argparse.BooleanOptionalAction,default=True); ap.add_argument('--wall-clearance-ft',type=float,default=.15)
# Output
ap.add_argument('--output',default='outputs/pool_v36'); ap.add_argument('--save-rgb-sec',type=float,default=2.0); ap.add_argument('--status-sec',type=float,default=1.0)
ap.add_argument('--dry-run',action='store_true')
args=ap.parse_args(); OUT=Path(args.output); OUT.mkdir(parents=True,exist_ok=True); (OUT/'rgb').mkdir(exist_ok=True); (OUT/'thermal').mkdir(exist_ok=True)

L1=DigitalOutputDevice(17,initial_value=False); L2=DigitalOutputDevice(27,initial_value=False); R1=DigitalOutputDevice(22,initial_value=False); R2=DigitalOutputDevice(23,initial_value=False)
def motors_off(): L1.off(); L2.off(); R1.off(); R2.off()
RUN_T0=monotonic(); _elock=threading.Lock(); _last_event=RUN_T0
def rel(t=None): return (monotonic() if t is None else t)-RUN_T0
def event(msg):
 global _last_event
 now=monotonic()
 with _elock:
  print(f'[{rel(now):8.3f}s | event Δ {now-_last_event:6.3f}s] {msg}',flush=True); _last_event=now

class Motor:
 def __init__(self):
  self.lock=threading.Lock(); self.quit=threading.Event(); self.mode='OFF'; self.epoch=0; self.thread=threading.Thread(target=self._run,daemon=True); self.thread.start()
 def get(self):
  with self.lock:return self.mode
 def set(self,m,why=''):
  with self.lock:
   if m==self.mode:return
   self.mode=m; self.epoch+=1; ep=self.epoch
  event(f'MOTOR epoch={ep} -> {m}'+(f' | {why}' if why else ''))
 def same(self,m):
  with self.lock:return not self.quit.is_set() and self.mode==m
 def wait(self,s,m):
  end=monotonic()+max(0,s)
  while monotonic()<end:
   if not self.same(m):return False
   sleep(min(.01,end-monotonic()))
  return True
 def _run(self):
  while not self.quit.is_set():
   m=self.get()
   if args.dry_run or m=='OFF': motors_off(); sleep(.01); continue
   if m=='TURN_LEFT': motors_off(); R1.on(); sleep(.01); continue
   if m=='TURN_RIGHT': motors_off(); L1.on(); sleep(.01); continue
   if m=='FORWARD':
    motors_off(); L1.on()
    if not self.wait(args.forward_left_sec,m): motors_off(); continue
    motors_off()
    if args.forward_gap_sec and not self.wait(args.forward_gap_sec,m): continue
    R1.on()
    if not self.wait(args.forward_right_sec,m): motors_off(); continue
    motors_off()
    if args.forward_gap_sec:self.wait(args.forward_gap_sec,m)
  motors_off()
 def close(self): self.set('OFF','shutdown'); self.quit.set(); self.thread.join(timeout=1); motors_off()
motor=Motor()

class LatestFrame:
 def __init__(self): self.c=threading.Condition(); self.id=0; self.t=0.; self.img=None; self.closed=False
 def publish(self,i,t,img):
  with self.c:self.id=i;self.t=t;self.img=img;self.c.notify_all()
 def wait(self,last,timeout=1):
  with self.c:
   self.c.wait_for(lambda:self.closed or self.id>last,timeout)
   return None if self.closed or self.id<=last or self.img is None else (self.id,self.t,self.img)
 def close(self):
  with self.c:self.closed=True;self.c.notify_all()
latest=LatestFrame()
class Camera:
 def __init__(self): self.quit=threading.Event(); self.proc=None; self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
 def _run(self):
  cmd=['rpicam-vid','--codec','mjpeg','--width',str(args.width),'--height',str(args.height),'--framerate',str(args.fps),'--timeout','0','--nopreview','-o','-']
  try:
   self.proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0); buf=bytearray(); fid=0
   while not self.quit.is_set():
    ch=self.proc.stdout.read(8192)
    if not ch:break
    buf.extend(ch)
    while True:
     s=buf.find(b'\xff\xd8'); e=buf.find(b'\xff\xd9',s+2) if s>=0 else -1
     if s<0 or e<0:break
     jpg=bytes(buf[s:e+2]); del buf[:e+2]; im=cv2.imdecode(np.frombuffer(jpg,np.uint8),cv2.IMREAD_COLOR)
     if im is not None: fid+=1; latest.publish(fid,monotonic(),cv2.rotate(im,cv2.ROTATE_180))
  finally: latest.close()
 def close(self):
  self.quit.set()
  if self.proc:
   try:self.proc.terminate()
   except:pass
  self.thread.join(timeout=1)

def safety_metrics(img):
 g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); h,w=g.shape; roi=g[int(.25*h):int(.80*h),int(.25*w):int(.75*w)]
 ed=cv2.Canny(roi,50,120); E=float(np.count_nonzero(ed))/ed.size; inv=(ed==0).astype(np.uint8); d=cv2.distanceTransform(inv,cv2.DIST_L2,5); S=float(np.percentile(d,95)) if d.size else 0.
 if E<=1e-6:S=65534.
 return E,S

def parse_sequences(s):
 bank=[]
 for seq in s.split(';'):
  steps=[]
  for z in seq.split(','):
   if not z.strip():continue
   a,d=z.strip().split(':',1); a=a.upper(); d=float(d)
   if a not in 'FLRO':raise ValueError(z)
   if a in 'LR':d=min(d,args.sequence_max_turn_sec)
   steps.append((a,d))
  if steps:bank.append(steps)
 if not bank:raise ValueError('empty sequence bank')
 return bank
BANK=parse_sequences(args.sequences); mode_for={'F':'FORWARD','L':'TURN_LEFT','R':'TURN_RIGHT','O':'OFF'}

class Pose:
 def __init__(self): self.lock=threading.Lock(); self.x=0.;self.y=0.;self.h=0.;self.last=monotonic();self.last_mode='OFF'
 def update(self,now,mode):
  with self.lock:
   dt=max(0,min(.25,now-self.last)); hr=math.radians(self.h)
   if self.last_mode=='FORWARD': self.x+=args.display_forward_ft_s*dt*math.sin(hr); self.y+=args.display_forward_ft_s*dt*math.cos(hr)
   elif self.last_mode=='TURN_LEFT': self.h-=args.display_left_deg_s*dt
   elif self.last_mode=='TURN_RIGHT': self.h+=args.display_right_deg_s*dt
   self.h=((self.h+180)%360)-180; self.last=now; self.last_mode=mode
   return self.x,self.y,self.h
 def snap_wall(self):
  with self.lock:
   # Ray from estimated pose; choose first positive rectangle intersection in heading direction.
   dx,dy=math.sin(math.radians(self.h)),math.cos(math.radians(self.h)); xmin,xmax=-args.pool_short_ft/2,args.pool_short_ft/2; ymin,ymax=0,args.pool_long_ft
   cand=[]
   if dx>1e-6:cand.append(((xmax-self.x)/dx,'RIGHT'))
   if dx<-1e-6:cand.append(((xmin-self.x)/dx,'LEFT'))
   if dy>1e-6:cand.append(((ymax-self.y)/dy,'FAR'))
   if dy<-1e-6:cand.append(((ymin-self.y)/dy,'NEAR'))
   cand=[c for c in cand if c[0]>=0]
   if not cand:return None
   _,wall=min(cand); before=(self.x,self.y)
   c=args.wall_clearance_ft
   if wall=='RIGHT':self.x=xmax-c
   elif wall=='LEFT':self.x=xmin+c
   elif wall=='FAR':self.y=ymax-c
   else:self.y=ymin+c
   return wall,before,(self.x,self.y)
 def get(self):
  with self.lock:return self.x,self.y,self.h
pose=Pose()

class Vision:
 def __init__(self):
  self.enabled=False;self.model=None
  if args.vision_mode=='off':return
  p=Path(args.model).expanduser()
  if not p.exists():event(f'VISION unavailable model={p}');return
  try:
   from ultralytics import YOLO
   self.model=YOLO(str(p));self.enabled=True;event(f'VISION {args.vision_mode}: {p}')
  except Exception as e:event(f'VISION disabled: {e}')
 def detect(self,img,t):
  if not self.enabled:return None
  try:
   r=self.model.predict(img,conf=args.vision_conf,imgsz=args.vision_imgsz,verbose=False)[0]
   if r.boxes is None or len(r.boxes)==0:return None
   i=int(np.argmax(r.boxes.conf.cpu().numpy())); box=r.boxes.xyxy.cpu().numpy()[i]; conf=float(r.boxes.conf.cpu().numpy()[i]);x1,y1,x2,y2=map(float,box);h,w=img.shape[:2]
   cx=(x1+x2)/2;cy=(y1+y2)/2
   return {'t':t,'conf':conf,'error':(cx-w/2)/(w/2),'area':max(0,x2-x1)*max(0,y2-y1)/(w*h),'cx':cx,'cy':cy,'box':[x1,y1,x2,y2]}
  except Exception as e:event(f'VISION inference error {type(e).__name__}: {e}');return None

# Compact thermal analyzer derived from pool thermal_anomaly_test_v2.
def _local_bg(a,r=4):
 p=np.pad(a,((r,r),(r,r)),mode='edge');o=np.empty_like(a)
 for y in range(24):
  for x in range(32):o[y,x]=np.median(p[y:y+2*r+1,x:x+2*r+1])
 return o
def _components(mask):
 seen=np.zeros_like(mask,bool);out=[]
 for y in range(24):
  for x in range(32):
   if not mask[y,x] or seen[y,x]:continue
   q=[(y,x)];seen[y,x]=1;pix=[]
   while q:
    cy,cx=q.pop();pix.append((cy,cx))
    for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
     ny,nx=cy+dy,cx+dx
     if 0<=ny<24 and 0<=nx<32 and mask[ny,nx] and not seen[ny,nx]:seen[ny,nx]=1;q.append((ny,nx))
   out.append(pix)
 return out
def analyze_thermal(a):
 a=np.asarray(a,float); bg=_local_bg(a); choices=[]
 for pol,score in [('warm',a-bg),('cold',bg-a)]:
  if args.thermal_polarity not in (pol,'both'):continue
  for pix in _components(score>=args.thermal_threshold_c):
   if len(pix)<args.thermal_min_blob_pixels or len(pix)>0.18*768:continue
   ys=np.array([p[0] for p in pix]);xs=np.array([p[1] for p in pix]);edge=xs.min()<=1 or xs.max()>=30 or ys.min()<=1 or ys.max()>=22
   vals=np.array([score[y,x] for y,x in pix]);q=float(vals.mean()*math.sqrt(len(pix)))*(.25 if edge else 1)
   choices.append((q,pol,pix,score))
 if not choices:return None
 q,pol,pix,score=max(choices,key=lambda z:z[0]);ys=np.array([p[0] for p in pix],float);xs=np.array([p[1] for p in pix],float);wt=np.maximum(np.array([score[int(y),int(x)] for y,x in zip(ys,xs)]),.001);cx=float(np.average(xs,weights=wt));cy=float(np.average(ys,weights=wt));delta=float(max(score[y,x] for y,x in pix));bearing=((cx-15.5)/15.5)*(args.thermal_fov_x_deg/2)
 return {'polarity':pol,'bearing':bearing,'delta_c':delta,'blob_px':len(pix),'cx':cx,'cy':cy,'quality':q}

class ThermalWorker:
 def __init__(self): self.lock=threading.Lock();self.obs=deque(maxlen=50);self.quit=threading.Event();self.thread=None
 def start(self):
  if args.thermal_mode=='off':return
  self.thread=threading.Thread(target=self._run,daemon=True,name='thermal');self.thread.start()
 def _run(self):
  try:
   import board,busio,adafruit_mlx90640
   i2c=busio.I2C(board.SCL,board.SDA,frequency=400000);mlx=adafruit_mlx90640.MLX90640(i2c)
   try:mlx.refresh_rate=adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
   except:pass
   event(f'THERMAL {args.thermal_mode} polarity={args.thermal_polarity}')
   period=1/max(.2,args.thermal_rate_hz);frame=[0.]*768
   while not self.quit.is_set():
    t0=monotonic()
    try:
     mlx.getFrame(frame);a=np.array(frame,float).reshape(24,32);z=analyze_thermal(a);t=monotonic()
     if z:
      z['t']=t
      with self.lock:self.obs.append(z)
      np.save(OUT/'thermal'/f'thermal_{rel(t):09.3f}.npy',a)
    except Exception as e: pass
    sleep(max(.01,period-(monotonic()-t0)))
  except Exception as e:event(f'THERMAL unavailable: {type(e).__name__}: {e}')
 def recent(self,now):
  with self.lock:return [z.copy() for z in self.obs if now-z['t']<=args.thermal_window_sec]
 def close(self):self.quit.set(); self.thread and self.thread.join(timeout=1)
thermal=ThermalWorker()

def thermal_candidate(now):
 r=thermal.recent(now)
 if len(r)<args.thermal_candidate_hits:return None
 b=np.array([z['bearing'] for z in r[-args.thermal_candidate_hits:]])
 if np.ptp(b)>args.thermal_bearing_spread_deg:return None
 z=max(r[-args.thermal_candidate_hits:],key=lambda q:q['quality']).copy();z['bearing']=float(np.mean(b));return z

def approach_burst(area,area_hist):
 # Shrink with absolute apparent size and with positive recent growth.
 frac=min(1,max(0,area/max(args.approach_area_near,1e-6))); base=args.approach_forward_max_sec-(args.approach_forward_max_sec-args.approach_forward_min_sec)*frac
 growth=0.
 if len(area_hist)>=2:growth=max(0,area_hist[-1]-area_hist[-2])
 return max(args.approach_forward_min_sec,min(args.approach_forward_max_sec,base/(1+args.approach_growth_gain*growth*100)))

def main():
 global RUN_T0,_last_event
 RUN_T0=monotonic();_last_event=RUN_T0;pose.last=RUN_T0
 cam=Camera();vision=Vision();thermal.start();stamp=datetime.now().strftime('%Y%m%d_%H%M%S');csvp=OUT/f'telemetry_{stamp}.csv';f=open(csvp,'w',newline='',buffering=1);w=csv.writer(f);w.writerow(['t','frame','state','motor','x_ft','y_ft','heading_deg','E','S','seq','step','rgb_conf','rgb_err','rgb_area','thermal_bearing','thermal_delta_c','event'])
 state='SEARCH';lastid=0;seq_i=0;step_i=0;step_t=monotonic();hard_n=0;open_n=0;recovery_pulses=0;recovery_dir='TURN_LEFT' if args.default_recovery_turn=='left' else 'TURN_RIGHT';deadline=0.;last_rgb=None;last_rgb_t=-1e9;rgb_hits=deque(maxlen=max(1,args.vision_candidate_hits));area_hist=deque(maxlen=5);done_n=0;reacq_n=0;last_save=0.;last_status=0.;last_infer=-10**9;candidate_since=None;note=''
 if args.start_delay_sec>0:event(f'Starting in {args.start_delay_sec:.1f}s');sleep(args.start_delay_sec)
 event(f'START v36 sequences={len(BANK)} vision={args.vision_mode} thermal={args.thermal_mode}')
 try:
  while True:
   got=latest.wait(lastid,1)
   if got is None:
    if latest.closed:raise RuntimeError('camera closed')
    continue
   fid,cap_t,img=got;lastid=fid;now=monotonic();
   if now-cap_t>args.max_frame_age:continue
   x,y,h=pose.update(now,motor.get());E,S=safety_metrics(img);hard=(E<=args.collision_edge_max and S>=args.collision_scale_min);hard_n=hard_n+1 if hard else 0;open_n=open_n+1 if not hard else 0
   det=None
   if vision.enabled and fid-last_infer>=max(1,args.vision_every_frames):
    last_infer=fid;det=vision.detect(img,cap_t);rgb_hits.append(1 if det else 0)
    if det:last_rgb=det;last_rgb_t=now;area_hist.append(det['area'])
   elif last_rgb and now-last_rgb_t<.45:det=last_rgb
   th=thermal_candidate(now); note=''
   # Confirmed physical wall contact always wins.
   if hard_n>=args.collision_confirm_frames and state not in ('RECOVERY_PAUSE','RECOVERY_TURN'):
    motor.set('OFF','confirmed wall contact');state='RECOVERY_PAUSE';deadline=now+args.recovery_pause_sec;hard_n=0;open_n=0;recovery_pulses=0;candidate_since=None
    if args.wall_anchor:
     snap=pose.snap_wall()
     if snap:
      wall,b,a=snap;event(f'POSE WALL_ANCHOR {wall}: ({b[0]:.2f},{b[1]:.2f}) -> ({a[0]:.2f},{a[1]:.2f})');note=f'WALL_ANCHOR_{wall}'
    # choose turn away from the side toward which reconstructed heading points; alternate ambiguous end-wall cases
    _,_,hh=pose.get();recovery_dir='TURN_LEFT' if hh>0 else 'TURN_RIGHT'
    seq_i=(seq_i+1)%len(BANK);step_i=0;step_t=now
   if state=='RECOVERY_PAUSE' and now>=deadline:
    motor.set(recovery_dir,'recovery turn / seek open view');state='RECOVERY_TURN';deadline=now+args.recovery_turn_sec;open_n=0
   elif state=='RECOVERY_TURN':
    if open_n>=args.open_confirm_frames and now>=deadline-.15:
     motor.set('FORWARD','open view; leave wall');state='RECOVERY_FORWARD';deadline=now+args.recovery_forward_sec;step_t=now
    elif now>=deadline:
     recovery_pulses+=1
     if recovery_pulses>=args.recovery_max_pulses:
      motor.set('FORWARD','recovery max pulses; force translation');state='RECOVERY_FORWARD';deadline=now+args.recovery_forward_sec
     else:
      motor.set('OFF','recovery observe');sleep(args.recovery_pause_sec);motor.set(recovery_dir,'continue recovery turn');deadline=monotonic()+args.recovery_turn_sec;open_n=0
   elif state=='RECOVERY_FORWARD' and now>=deadline:
    state='SEARCH';step_t=now;motor.set('OFF','resume search')
   # Candidate acquisition: ACTIVE sensors only. MONITOR can never alter state.
   rgb_trigger=(args.vision_mode=='active' and sum(rgb_hits)>=args.vision_candidate_hits and now-last_rgb_t<.7)
   th_trigger=(args.thermal_mode=='active' and th is not None)
   if state=='SEARCH' and (rgb_trigger or th_trigger):
    state='CANDIDATE';candidate_since=now;motor.set('OFF','sensor candidate takeover');src='RGB' if rgb_trigger else 'THERMAL';event(f'CANDIDATE source={src}');deadline=now+args.target_observe_sec
   # Candidate: if RGB exists, immediately lock. Thermal-only: orient cautiously toward bearing and keep seeking RGB/thermal affirmations.
   if state=='CANDIDATE':
    if motor.get()!='OFF' and now>=deadline:
     motor.set('OFF','candidate observe');deadline=now+args.target_observe_sec
    if args.vision_mode=='active' and det is not None:
     state='TARGET';motor.set('OFF','RGB lock');deadline=now+args.target_observe_sec;event(f'RGB LOCK err={det["error"]:+.3f} area={det["area"]:.4f}')
    elif th_trigger and now>=deadline:
     b=th['bearing'];turn='TURN_RIGHT' if b>0 else 'TURN_LEFT';dur=max(args.target_turn_min_sec,min(args.target_turn_max_sec,abs(b)/30*.25));motor.set(turn,f'thermal investigate bearing={b:+.1f}');deadline=now+dur
    elif candidate_since and now-candidate_since>args.candidate_timeout_sec:
     motor.set('OFF','candidate expired');state='SEARCH';rgb_hits.clear();step_t=now
   # RGB primary visual servo. Pulses end in OFF/observe; next fresh detection decides again.
   if state=='TARGET':
    # Every steering/forward pulse ends with an OFF observation interval.
    if motor.get()!='OFF' and now>=deadline:
     motor.set('OFF','target observe');deadline=now+args.target_observe_sec
    if det is not None:
     e=float(det['error']);a=float(det['area']);done_n=done_n+1 if a>=args.done_area else 0
     if done_n>=args.done_hits:
      motor.set('OFF',f'DONE RGB area={a:.3f}');event(f'TARGET DONE area={a:.3f} thermal_delta={(th or {}).get("delta_c",0):.2f}C');state='DONE' if args.target_done=='hold' else 'SEARCH';rgb_hits.clear();step_t=now
     elif now>=deadline:
      if abs(e)>args.target_deadband:
       turn='TURN_RIGHT' if e>0 else 'TURN_LEFT';scale=min(1,abs(e)/max(args.target_hard_error,1e-6));dur=args.target_turn_min_sec+(args.target_turn_max_sec-args.target_turn_min_sec)*scale;motor.set(turn,f'RGB center e={e:+.3f}');deadline=now+dur
      else:
       dur=approach_burst(a,list(area_hist));motor.set('FORWARD',f'RGB approach area={a:.4f} burst={dur:.2f}s');deadline=now+dur
    elif now-last_rgb_t>=args.target_lost_sec:
     # Large recent target disappearing after approach can mean arrival; otherwise bounded reacquisition.
     if last_rgb and last_rgb['area']>=args.done_area*.8:
      motor.set('OFF','large target vanished close; possible arrival');state='DONE';event('TARGET DONE possible close-FOV disappearance')
     else:
      motor.set('OFF','RGB lost; reacquire');state='REACQUIRE';reacq_n=0;deadline=now+args.target_observe_sec
   if state=='REACQUIRE':
    if det is not None:state='TARGET';motor.set('OFF','RGB reacquired');deadline=now+args.target_observe_sec;event('RGB REACQUIRED')
    elif now>=deadline:
     if reacq_n>=args.reacquire_attempts:state='SEARCH';motor.set('OFF','reacquire exhausted');rgb_hits.clear();step_t=now
     else:
      # first search toward last RGB side; then alternate around it
      first='TURN_RIGHT' if last_rgb and last_rgb['error']>0 else 'TURN_LEFT';turn=first if reacq_n%2==0 else ('TURN_LEFT' if first=='TURN_RIGHT' else 'TURN_RIGHT');motor.set(turn,f'reacquire {reacq_n+1}');sleep(args.reacquire_turn_sec);motor.set('OFF','reacquire observe');reacq_n+=1;deadline=monotonic()+args.target_observe_sec
   if state=='DONE':motor.set('OFF')
   # SEARCH bank. Sequence changes each complete cycle and after every wall recovery.
   if state=='SEARCH':
    seq=BANK[seq_i];act,dur=seq[step_i]
    if now-step_t>=dur:
     step_i+=1;step_t=now
     if step_i>=len(seq):
      step_i=0
      if args.sequence_switch_on_cycle:seq_i=(seq_i+1)%len(BANK)
     act,dur=BANK[seq_i][step_i]
    motor.set(mode_for[act],f'SEARCH seq={seq_i} step={step_i} {act}:{dur:.2f}')
   # Save annotated RGB + telemetry.
   if now-last_save>=args.save_rgb_sec:
    vis=img.copy();cv2.line(vis,(args.width//2,0),(args.width//2,args.height),(255,255,255),1)
    if det:
     x1,y1,x2,y2=map(int,det['box']);cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.putText(vis,f'{state} x={x:.1f} y={y:.1f} h={h:.0f}',(10,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2);cv2.imwrite(str(OUT/'rgb'/f'rgb_{rel(cap_t):09.3f}.jpg'),vis);last_save=now
   x,y,h=pose.get();td=det or {};tz=th or {};w.writerow([f'{rel(now):.3f}',fid,state,motor.get(),f'{x:.4f}',f'{y:.4f}',f'{h:.2f}',f'{E:.6f}',f'{S:.2f}',seq_i,step_i,td.get('conf',''),td.get('error',''),td.get('area',''),tz.get('bearing',''),tz.get('delta_c',''),note])
   if now-last_status>=args.status_sec:
    print(f'[{rel(now):8.3f}s] {state:16s} motor={motor.get():10s} pose=({x:+.2f},{y:+.2f},{h:+.0f}) wall={hard_n} RGB={td.get("conf",0):.2f}/{td.get("error",0):+.2f}/{td.get("area",0):.3f} TH={tz.get("bearing",0):+.1f}/{tz.get("delta_c",0):.2f} seq={seq_i}:{step_i}',flush=True);last_status=now
 except KeyboardInterrupt:print('\n*** CTRL-C: MOTORS OFF ***',flush=True)
 finally:
  motor.set('OFF','final shutdown');motors_off();thermal.close();cam.close();motor.close();f.close();(OUT/f'run_{stamp}_config.json').write_text(json.dumps(vars(args),indent=2));event(f'LOG saved {csvp}')
if __name__=='__main__':main()
