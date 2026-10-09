#!/usr/bin/env python3
"""Ocean Voyager v44 pool autonomy with auditable RGB/thermal fusion.

SEARCH: configurable sequence bank; switches sequence after wall events.
SAFETY: confirmed wall contact -> stop -> bounded turn until camera view clears -> resume.
SENSORS: RGB and MLX90640 are independent OFF/MONITOR/ACTIVE producers.
TARGET: only corroborated RGB + thermal evidence may interrupt SEARCH.
MISSION: continuous movement is primary; target pursuit is a short diversion, never a terminal state.
POSE: display-only dead reckoning (x,y,heading) with wall-anchor corrections for replay.

Coordinate convention: start at midpoint of near short edge: x=0, y=0, heading=0 deg (+Y).
GPIO preserved from v34/v35: left 17/27, right 22/23.
"""
from gpiozero import DigitalOutputDevice
from time import sleep, monotonic
from pathlib import Path
from datetime import datetime
from collections import deque
import argparse, csv, json, math, subprocess, threading, random
import cv2, numpy as np

ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Motors / start
ap.add_argument('--forward-left-sec',type=float,default=1.7); ap.add_argument('--forward-right-sec',type=float,default=2.0)
ap.add_argument('--forward-gap-sec',type=float,default=0.0); ap.add_argument('--start-delay-sec',type=float,default=0.0)
# v45 stochastic navigation: ACTION probability,min_sec,max_sec
ap.add_argument('--normal-forward',default='0.55,1.0,8.0')
ap.add_argument('--normal-left',default='0.18,0.3,3.0')
ap.add_argument('--normal-right',default='0.18,0.3,3.0')
ap.add_argument('--normal-pause',default='0.09,0.3,2.0')
ap.add_argument('--collision-left',default='0.40,0.4,3.0')
ap.add_argument('--collision-right',default='0.40,0.4,3.0')
ap.add_argument('--collision-pause',default='0.20,0.3,1.5')
ap.add_argument('--random-seed',type=int,default=None,help='fixed seed reproduces the random action sequence')
ap.add_argument('--archive-rgb-every-frames',type=int,default=2,help='archive raw RGB every N camera frames for offline trajectory reconstruction; 1 saves every frame')
# Independent forward trim controls
ap.add_argument('--forward-schedule',default='',help='SEARCH-forward-time ranges start-end:left:right, comma separated; e.g. 0-5:1.5:2.0,5-25:1.4:2.0')
ap.add_argument('--visual-drift-mode',choices=['off','monitor','adaptive'],default='off')
ap.add_argument('--visual-drift-window-sec',type=float,default=0.7); ap.add_argument('--visual-drift-min-features',type=int,default=12)
ap.add_argument('--visual-drift-threshold-px',type=float,default=3.0); ap.add_argument('--visual-drift-confirmations',type=int,default=3)
ap.add_argument('--visual-adapt-step-sec',type=float,default=.05); ap.add_argument('--visual-adapt-max-sec',type=float,default=.10)
ap.add_argument('--visual-adapt-cooldown-sec',type=float,default=2.0)
# Independent periodic counter-steer during SEARCH forward travel.
ap.add_argument('--counter-steer-mode',choices=['off','timed'],default='off')
ap.add_argument('--counter-steer-direction',choices=['left','right'],default='right')
ap.add_argument('--counter-steer-every-sec',type=float,default=4.0)
ap.add_argument('--counter-steer-pulse-sec',type=float,default=.20)
ap.add_argument('--counter-steer-schedule',default='',help='optional forward-time ranges start-end:pulse_sec; e.g. 0-15:.15,15-30:.20')
# Camera
ap.add_argument('--width',type=int,default=640); ap.add_argument('--height',type=int,default=480); ap.add_argument('--fps',type=int,default=10)
ap.add_argument('--max-frame-age',type=float,default=.45)
# Search sequence bank. F is calibrated sequential forward cycle; L/R are continuous one-pump pulses; O=off.
ap.add_argument('--sequences',default='F:24;F:20,L:0.20,F:16;F:20,R:0.20,F:16;F:18,L:0.15,F:12,R:0.15,F:12')
ap.add_argument('--sequence-max-turn-sec',type=float,default=1.0); ap.add_argument('--sequence-switch-on-cycle',action=argparse.BooleanOptionalAction,default=True)
# Wall contact + recovery
ap.add_argument('--collision-edge-max',type=float,default=.002); ap.add_argument('--collision-scale-min',type=float,default=1000.)
ap.add_argument('--collision-confirm-frames',type=int,default=3); ap.add_argument('--recovery-pause-sec',type=float,default=.35)
ap.add_argument('--recovery-turn-sec',type=float,default=3.0); ap.add_argument('--recovery-max-pulses',type=int,default=4)
ap.add_argument('--open-confirm-frames',type=int,default=6); ap.add_argument('--recovery-forward-sec',type=float,default=4.0)
ap.add_argument('--default-recovery-turn',choices=['left','right'],default='left')
# Sensor modes
ap.add_argument('--vision-mode',choices=['off','monitor','active'],default='active')
ap.add_argument('--thermal-mode',choices=['off','monitor','active'],default='monitor')
# RGB
ap.add_argument('--model',default='best.pt'); ap.add_argument('--vision-conf',type=float,default=.25,help='YOLO inference/logging floor'); ap.add_argument('--vision-strong-conf',type=float,default=.35,help='strong RGB threshold; 0.25-0.35 is weak and requires thermal corroboration'); ap.add_argument('--vision-imgsz',type=int,default=640)
ap.add_argument('--vision-every-frames',type=int,default=3); ap.add_argument('--vision-candidate-hits',type=int,default=1)
ap.add_argument('--vision-max-area',type=float,default=.12,help='reject implausibly large YOLO boxes')
ap.add_argument('--vision-max-width-frac',type=float,default=.55,help='reject boxes spanning too much image width')
ap.add_argument('--vision-max-height-frac',type=float,default=.55,help='reject boxes spanning too much image height')
ap.add_argument('--sensor-bearing-agree-deg',type=float,default=18.0,help='RGB/thermal bearing agreement for strong combined evidence')
# Thermal
ap.add_argument('--thermal-polarity',choices=['warm','cold','both'],default='warm'); ap.add_argument('--thermal-threshold-c',type=float,default=.8)
ap.add_argument('--thermal-min-blob-pixels',type=int,default=4); ap.add_argument('--thermal-fov-x-deg',type=float,default=55.)
ap.add_argument('--thermal-rate-hz',type=float,default=2.0); ap.add_argument('--thermal-candidate-hits',type=int,default=2)
ap.add_argument('--thermal-window-sec',type=float,default=1.8); ap.add_argument('--thermal-bearing-spread-deg',type=float,default=15.)
ap.add_argument('--thermal-save-png',action=argparse.BooleanOptionalAction,default=True,help='save viewable thermal PNG for every valid thermal frame')
ap.add_argument('--thermal-min-delta-active-c',type=float,default=2.5,help='minimum thermal contrast for takeover')
ap.add_argument('--candidate-forward',action=argparse.BooleanOptionalAction,default=True,help='keep moving during unconfirmed candidate tracking')
# Candidate / target approach
ap.add_argument('--candidate-timeout-sec',type=float,default=3.0); ap.add_argument('--target-deadband',type=float,default=.14)
ap.add_argument('--target-hard-error',type=float,default=.35); ap.add_argument('--target-turn-min-sec',type=float,default=.16); ap.add_argument('--target-turn-max-sec',type=float,default=.55)
ap.add_argument('--approach-forward-min-sec',type=float,default=.30); ap.add_argument('--approach-forward-max-sec',type=float,default=1.60)
ap.add_argument('--approach-area-near',type=float,default=.07); ap.add_argument('--approach-growth-gain',type=float,default=4.0)
ap.add_argument('--target-observe-sec',type=float,default=.18); ap.add_argument('--target-lost-sec',type=float,default=1.5)
ap.add_argument('--reacquire-turn-sec',type=float,default=.30); ap.add_argument('--reacquire-attempts',type=int,default=6)
ap.add_argument('--done-area',type=float,default=.10); ap.add_argument('--done-hits',type=int,default=3)
ap.add_argument('--done-thermal-delta-c',type=float,default=2.5,help='supporting evidence only; RGB area remains primary reached rule')
ap.add_argument('--done-center-error',type=float,default=.20,help='target must also be centered before TARGET_REACHED')
ap.add_argument('--object-avoid-area',type=float,default=.085,help='end target approach at this RGB area fraction and perform one stochastic collision-bank escape action')
ap.add_argument('--target-cooldown-sec',type=float,default=30.0,help='mandatory random-navigation interval after ANY target approach ends')
ap.add_argument('--target-pursuit-max-sec',type=float,default=6.0,help='hard cap on a target diversion before resuming SEARCH')
ap.add_argument('--target-done',choices=['hold','resume'],default='resume',help='compatibility option; continuous mission resumes SEARCH')
# Display pose / replay
ap.add_argument('--pool-short-ft',type=float,default=5.0); ap.add_argument('--pool-long-ft',type=float,default=7.0)
ap.add_argument('--display-forward-ft-s',type=float,default=.175,help='display-only; ~7 ft / 40 s initial estimate')
ap.add_argument('--display-left-deg-s',type=float,default=30.0); ap.add_argument('--display-right-deg-s',type=float,default=45.0)
ap.add_argument('--wall-anchor',action=argparse.BooleanOptionalAction,default=True); ap.add_argument('--wall-clearance-ft',type=float,default=.15)
# Output
ap.add_argument('--output',default='outputs/pool_v45_random'); ap.add_argument('--save-rgb-sec',type=float,default=2.0); ap.add_argument('--status-sec',type=float,default=1.0)
ap.add_argument('--dry-run',action='store_true')
args=ap.parse_args(); OUT=Path(args.output); OUT.mkdir(parents=True,exist_ok=True); (OUT/'rgb').mkdir(exist_ok=True); (OUT/'rgb_raw').mkdir(exist_ok=True); (OUT/'thermal').mkdir(exist_ok=True)


def parse_forward_schedule(spec):
 ranges=[]
 if not spec.strip(): return ranges
 for item in spec.split(','):
  span,l,r=item.strip().split(':')
  a,b=span.split('-',1); a=float(a); b=float(b); l=float(l); r=float(r)
  if b<=a or min(l,r)<=0: raise ValueError(f'bad forward schedule item: {item}')
  ranges.append((a,b,l,r))
 ranges.sort()
 for i,z in enumerate(ranges):
  if i and z[0] < ranges[i-1][1]-1e-9: raise ValueError('overlapping forward schedule ranges')
 return ranges
FORWARD_SCHEDULE=parse_forward_schedule(args.forward_schedule)
def parse_counter_schedule(spec):
 ranges=[]
 if not spec.strip(): return ranges
 for item in spec.split(','):
  span,p=item.strip().split(':'); a,b=span.split('-',1); a=float(a); b=float(b); p=float(p)
  if b<=a or p<=0: raise ValueError(f'bad counter-steer schedule item: {item}')
  ranges.append((a,b,p))
 ranges.sort()
 for i,z in enumerate(ranges):
  if i and z[0] < ranges[i-1][1]-1e-9: raise ValueError('overlapping counter-steer schedule ranges')
 return ranges
COUNTER_SCHEDULE=parse_counter_schedule(args.counter_steer_schedule)
def counter_pulse_for(t):
 for a,b,p in COUNTER_SCHEDULE:
  if a<=t<b:return p,f'{a:g}-{b:g}'
 return args.counter_steer_pulse_sec,'default'

def parse_random_action(spec,name):
 parts=[x.strip() for x in spec.split(',')]
 if len(parts)!=3: raise ValueError(f'{name} must be probability,min_sec,max_sec')
 p,a,b=map(float,parts)
 if p<0 or a<0 or b<a: raise ValueError(f'bad {name}: {spec}')
 return p,a,b

def build_random_bank(items):
 bank=[]
 for action,spec in items:
  p,a,b=parse_random_action(spec,action)
  if p>0: bank.append((action,p,a,b))
 if not bank or sum(z[1] for z in bank)<=0: raise ValueError('random action bank has zero probability')
 return bank

NORMAL_BANK=build_random_bank([('FORWARD',args.normal_forward),('TURN_LEFT',args.normal_left),('TURN_RIGHT',args.normal_right),('OFF',args.normal_pause)])
COLLISION_BANK=build_random_bank([('TURN_LEFT',args.collision_left),('TURN_RIGHT',args.collision_right),('OFF',args.collision_pause)])
EFFECTIVE_RANDOM_SEED=args.random_seed if args.random_seed is not None else random.SystemRandom().randrange(0,2**63)
RNG=random.Random(EFFECTIVE_RANDOM_SEED)

def choose_random(bank):
 total=sum(z[1] for z in bank); r=RNG.random()*total; acc=0.
 for action,p,a,b in bank:
  acc+=p
  if r<=acc:
   return action,RNG.uniform(a,b),r/total
 action,p,a,b=bank[-1]; return action,RNG.uniform(a,b),r/total

class DriveTrim:
 def __init__(self):
  self.lock=threading.Lock(); self.search_forward_sec=0.; self.last_t=monotonic(); self.last_counting=False
  self.base_l=args.forward_left_sec; self.base_r=args.forward_right_sec; self.visual_l_delta=0.; self.range_label='base'
 def tick(self,now,counting):
  with self.lock:
   dt=max(0,min(.5,now-self.last_t))
   if self.last_counting:self.search_forward_sec+=dt
   self.last_t=now; self.last_counting=counting
   self.base_l,self.base_r=args.forward_left_sec,args.forward_right_sec; self.range_label='base'
   for a,b,l,r in FORWARD_SCHEDULE:
    if a<=self.search_forward_sec<b:
     self.base_l,self.base_r=l,r; self.range_label=f'{a:g}-{b:g}'; break
   return self.search_forward_sec
 def timing(self):
  with self.lock:return max(.05,self.base_l+self.visual_l_delta),max(.05,self.base_r),self.search_forward_sec,self.range_label,self.visual_l_delta
 def adapt(self,direction):
  with self.lock:
   old=self.visual_l_delta
   if direction=='LEFT': self.visual_l_delta=max(-args.visual_adapt_max_sec,self.visual_l_delta-args.visual_adapt_step_sec)
   elif direction=='RIGHT': self.visual_l_delta=min(args.visual_adapt_max_sec,self.visual_l_delta+args.visual_adapt_step_sec)
   return old,self.visual_l_delta
drive=DriveTrim()

L1=DigitalOutputDevice(17,initial_value=False); L2=DigitalOutputDevice(27,initial_value=False); R1=DigitalOutputDevice(22,initial_value=False); R2=DigitalOutputDevice(23,initial_value=False)
def motors_off(): L1.off(); L2.off(); R1.off(); R2.off()
RUN_T0=monotonic(); _elock=threading.Lock(); _last_event=RUN_T0
def rel(t=None): return (monotonic() if t is None else t)-RUN_T0
def event(msg):
 global _last_event
 now=monotonic()
 with _elock:
  line=f'[{rel(now):8.3f}s | event Δ {now-_last_event:6.3f}s] {msg}'
  print(line,flush=True)
  try:
   with open(OUT/'events.log','a',buffering=1) as ef: ef.write(line+'\n')
  except Exception: pass
  _last_event=now

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
  while True:
   if not self.same(m):return False
   remaining=end-monotonic()
   if remaining<=0:return True
   sleep(min(.01,remaining))
 def _run(self):
  while not self.quit.is_set():
   m=self.get()
   if args.dry_run or m=='OFF': motors_off(); sleep(.01); continue
   if m=='TURN_LEFT': motors_off(); R1.on(); sleep(.01); continue
   if m=='TURN_RIGHT': motors_off(); L1.on(); sleep(.01); continue
   if m=='FORWARD':
    motors_off(); L1.on()
    fl,fr,_,_,_=drive.timing()
    if not self.wait(fl,m): motors_off(); continue
    motors_off()
    if args.forward_gap_sec and not self.wait(args.forward_gap_sec,m): continue
    R1.on()
    if not self.wait(fr,m): motors_off(); continue
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


class VisualDrift:
 def __init__(self):
  self.prev=None; self.prev_t=0.; self.hist=deque(maxlen=max(3,args.visual_drift_confirmations)); self.last_adapt=-1e9
 def reset(self): self.prev=None; self.hist.clear()
 def update(self,img,t,eligible):
  if args.visual_drift_mode=='off' or not eligible:
   self.reset(); return ('UNKNOWN',0.,0,0.)
  h,w=img.shape[:2]; gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); gray=gray[:max(1,int(.55*h)),:]
  if self.prev is None or t-self.prev_t>max(1.5,args.visual_drift_window_sec*3):
   self.prev=gray;self.prev_t=t;return ('UNKNOWN',0.,0,0.)
  if t-self.prev_t<args.visual_drift_window_sec:return ('WAIT',0.,0,0.)
  p0=cv2.goodFeaturesToTrack(self.prev,maxCorners=120,qualityLevel=.02,minDistance=10,blockSize=7)
  direction='UNKNOWN'; conf=0.; n=0; dx=0.
  if p0 is not None:
   p1,st,err=cv2.calcOpticalFlowPyrLK(self.prev,gray,p0,None,winSize=(21,21),maxLevel=3)
   if p1 is not None:
    good0=p0[st.ravel()==1].reshape(-1,2); good1=p1[st.ravel()==1].reshape(-1,2); n=len(good0)
    if n>=args.visual_drift_min_features:
     M,inl=cv2.estimateAffinePartial2D(good0,good1,method=cv2.RANSAC,ransacReprojThreshold=3.0)
     if M is not None:
      dx=float(M[0,2]); nin=int(inl.sum()) if inl is not None else n; conf=min(1.,nin/max(args.visual_drift_min_features*2,1))
      if abs(dx)<args.visual_drift_threshold_px: direction='STRAIGHT'
      else: direction='LEFT' if dx>0 else 'RIGHT'
  self.prev=gray;self.prev_t=t
  if direction in ('LEFT','RIGHT','STRAIGHT'):self.hist.append(direction)
  return direction,conf,n,dx
visual_drift=VisualDrift()

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
   boxes=r.boxes.xyxy.cpu().numpy(); confs=r.boxes.conf.cpu().numpy(); h,w=img.shape[:2]; plausible=[]
   for i,box in enumerate(boxes):
    x1,y1,x2,y2=map(float,box); bw=max(0,x2-x1); bh=max(0,y2-y1); area=bw*bh/(w*h)
    if area>args.vision_max_area or bw/w>args.vision_max_width_frac or bh/h>args.vision_max_height_frac: continue
    plausible.append((float(confs[i]),x1,y1,x2,y2,area))
   if not plausible:return None
   conf,x1,y1,x2,y2,area=max(plausible,key=lambda z:z[0]); cx=(x1+x2)/2;cy=(y1+y2)/2
   return {'t':t,'conf':conf,'strength':('STRONG' if conf>=args.vision_strong_conf else 'WEAK'),'error':(cx-w/2)/(w/2),'bearing':((cx-w/2)/(w/2))*(args.thermal_fov_x_deg/2),'area':area,'cx':cx,'cy':cy,'box':[x1,y1,x2,y2]}
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

def save_thermal_png(a,path):
 finite=a[np.isfinite(a)]
 if finite.size==0:return
 lo=float(np.percentile(finite,2));hi=float(np.percentile(finite,98))
 if hi<=lo:hi=lo+1e-6
 u=np.clip((a-lo)/(hi-lo)*255.0,0,255).astype(np.uint8)
 u=cv2.resize(u,(320,240),interpolation=cv2.INTER_NEAREST)
 u=cv2.applyColorMap(u,cv2.COLORMAP_INFERNO)
 cv2.imwrite(str(path),u)

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
     mlx.getFrame(frame);a=np.array(frame,float).reshape(24,32);t=monotonic()
     ts=rel(t)
     np.save(OUT/'thermal'/f'thermal_{ts:09.3f}.npy',a)
     if args.thermal_save_png:save_thermal_png(a,OUT/'thermal'/f'thermal_{ts:09.3f}.png')
     z=analyze_thermal(a)
     if z:
      z['t']=t; z['file_png']=f'thermal_{ts:09.3f}.png' if args.thermal_save_png else ''; z['file_npy']=f'thermal_{ts:09.3f}.npy'; z['rel_t']=ts
      with self.lock:self.obs.append(z)
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
 cam=Camera();vision=Vision()
 if args.vision_mode=='active' and not vision.enabled:
  cam.close(); motors_off(); raise RuntimeError('vision-mode active requested but YOLO model failed to load')
 thermal.start();stamp=datetime.now().strftime('%Y%m%d_%H%M%S');csvp=OUT/f'telemetry_{stamp}.csv';f=open(csvp,'w',newline='',buffering=1);w=csv.writer(f)
 w.writerow(['t','frame','state','motor','x_ft','y_ft','heading_deg','E','S','hard_frames','random_bank','random_action','random_requested_sec','random_remaining_sec','rgb_conf','rgb_err','rgb_area','thermal_bearing','thermal_delta_c','fused','cooldown_remaining_sec','event'])
 state='RANDOM';lastid=0;hard_n=0;last_rgb=None;last_rgb_t=-1e9;rgb_hits=deque(maxlen=max(1,args.vision_candidate_hits));area_hist=deque(maxlen=5);done_n=0;last_save=0.;last_status=0.;last_infer=-10**9;target_since=None;target_cooldown_until=-1e9;deadline=0.;note='';random_action='OFF';random_requested=0.;random_bank='NORMAL'
 def start_random(bank_name,why=''):
  nonlocal deadline,random_action,random_requested,random_bank,state
  bank=NORMAL_BANK if bank_name=='NORMAL' else COLLISION_BANK
  action,dur,draw=choose_random(bank); random_action=action;random_requested=dur;random_bank=bank_name;deadline=monotonic()+dur;state='RANDOM' if bank_name=='NORMAL' else 'COLLISION_RANDOM'
  motor.set(action,f'{bank_name}_RANDOM requested={dur:.3f}s draw={draw:.6f}'+(f' | {why}' if why else ''))
  event(f'RANDOM_CHOICE bank={bank_name} action={action} requested={dur:.3f}s draw={draw:.6f}')
 if args.start_delay_sec>0:event(f'Starting in {args.start_delay_sec:.1f}s');sleep(args.start_delay_sec)
 event(f'START v45 stochastic navigation seed={EFFECTIVE_RANDOM_SEED} vision={args.vision_mode} thermal={args.thermal_mode} target_max={args.target_pursuit_max_sec:.1f}s cooldown={args.target_cooldown_sec:.1f}s')
 event(f'NORMAL_BANK F={args.normal_forward} L={args.normal_left} R={args.normal_right} P={args.normal_pause}')
 event(f'COLLISION_BANK L={args.collision_left} R={args.collision_right} P={args.collision_pause}')
 start_random('NORMAL','mission start')
 try:
  while True:
   got=latest.wait(lastid,1)
   if got is None:
    if latest.closed:raise RuntimeError('camera closed')
    continue
   fid,cap_t,img=got;lastid=fid;now=monotonic()
   if now-cap_t>args.max_frame_age:continue
   if args.archive_rgb_every_frames>0 and fid % args.archive_rgb_every_frames==0:
    cv2.imwrite(str(OUT/'rgb_raw'/f'rgb_raw_{rel(cap_t):09.3f}_f{fid:07d}.jpg'),img)
   x,y,h=pose.update(now,motor.get()); E,S=safety_metrics(img);hard=(E<=args.collision_edge_max and S>=args.collision_scale_min);hard_n=hard_n+1 if hard else 0
   det=None;new_rgb_obs=False;rgb_ts=None;rgb_file=''
   if vision.enabled and fid-last_infer>=max(1,args.vision_every_frames):
    last_infer=fid;new_rgb_obs=True;rgb_ts=rel(cap_t);rgb_file=f'rgb_obs_{rgb_ts:09.3f}.jpg';cv2.imwrite(str(OUT/'rgb'/rgb_file),img);det=vision.detect(img,cap_t);rgb_hits.append(1 if det else 0)
    if det:det['file']=rgb_file;det['rel_t']=rgb_ts;last_rgb=det;last_rgb_t=now;area_hist.append(det['area'])
   elif last_rgb and now-last_rgb_t<.45:det=last_rgb
   th=thermal_candidate(now);note=''
   if new_rgb_obs:
    if det:event(f'RGB_OBS t={det.get("rel_t",rel(det["t"])):.3f} file={det.get("file","")} strength={det.get("strength","")} conf={det["conf"]:.3f} box={[round(v,1) for v in det["box"]]} bearing={det["bearing"]:+.1f} area={det["area"]:.4f} | TH='+(f't={th.get("rel_t",rel(th["t"])):.3f} file={th.get("file_png","")} raw={th.get("file_npy","")} delta={th.get("delta_c",0):.2f}C bearing={th.get("bearing",0):+.1f}' if th else 'NONE'))
    else:event(f'RGB_OBS t={rgb_ts:.3f} file={rgb_file} hit=NONE | TH='+(f't={th.get("rel_t",rel(th["t"])):.3f} file={th.get("file_png","")} raw={th.get("file_npy","")} delta={th.get("delta_c",0):.2f}C bearing={th.get("bearing",0):+.1f}' if th else 'NONE'))
   if th:event(f'THERMAL_OBS t={th.get("rel_t",rel(th["t"])):.3f} file={th.get("file_png","")} raw={th.get("file_npy","")} delta={th.get("delta_c",0):.2f}C bearing={th.get("bearing",0):+.1f} | RGB='+(f't={det.get("rel_t",rel(det["t"])):.3f} file={det.get("file","")} conf={det.get("conf",0):.3f} bearing={det.get("bearing",0):+.1f}' if det else 'NONE'))
   rgb_trigger=(args.vision_mode=='active' and sum(rgb_hits)>=args.vision_candidate_hits and now-last_rgb_t<.7)
   th_trigger=(args.thermal_mode=='active' and th is not None and th.get('delta_c',0)>=args.thermal_min_delta_active_c)
   agree=bool(det is not None and th is not None and abs(float(det.get('bearing',0))-float(th.get('bearing',0)))<=args.sensor_bearing_agree_deg)
   fused=bool(rgb_trigger and th_trigger and det is not None and det.get('conf',0)>=args.vision_conf and agree)
   # Wall contact preempts everything, including target approach. Recovery is stochastic L/R/PAUSE only.
   if hard_n>=args.collision_confirm_frames and state!='COLLISION_RANDOM':
    if state=='TARGET':
     event('TARGET_ABORT wall contact -> collision random recovery');target_since=None;target_cooldown_until=now+args.target_cooldown_sec;rgb_hits.clear();area_hist.clear();done_n=0
    motor.set('OFF','confirmed wall contact');event(f'COLLISION_CONFIRMED E={E:.6f} S={S:.2f} -> stochastic recovery');hard_n=0;start_random('COLLISION','wall contact')
   elif state=='COLLISION_RANDOM':
    if now>=deadline:
     # If current view is still a confirmed wall, draw another collision action; otherwise resume normal randomness.
     if hard:
      start_random('COLLISION','wall still present')
     else:
      event('COLLISION_CLEAR -> normal random navigation');hard_n=0;start_random('NORMAL','collision clear')
   elif state=='TARGET':
    if target_since is not None and now-target_since>=args.target_pursuit_max_sec:
     event(f'TARGET_TIMEOUT {now-target_since:.1f}s -> mandatory cooldown/random navigation');target_since=None;target_cooldown_until=now+args.target_cooldown_sec;rgb_hits.clear();area_hist.clear();done_n=0;start_random('NORMAL','target timeout')
    elif det is not None and th_trigger and agree:
     e=float(det['error']);a=float(det['area']);close_and_centered=(a>=args.done_area and abs(e)<=args.done_center_error);done_n=done_n+1 if close_and_centered else 0
     if a>=args.object_avoid_area:
      event(f'OBJECT_NEAR area={a:.3f} -> end approach, cooldown, stochastic escape');target_since=None;target_cooldown_until=now+args.target_cooldown_sec;rgb_hits.clear();area_hist.clear();done_n=0;start_random('COLLISION','object proximity')
     elif done_n>=args.done_hits:
      ev=img.copy();x1,y1,x2,y2=map(int,det['box']);cv2.rectangle(ev,(x1,y1),(x2,y2),(0,255,0),3);cv2.putText(ev,f'TARGET_CONFIRMED conf={det["conf"]:.2f} err={e:+.2f} area={a:.3f}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2);cv2.imwrite(str(OUT/'rgb'/f'target_confirmed_{rel(cap_t):09.3f}.jpg'),ev)
      event(f'TARGET_CONFIRMED area={a:.3f} err={e:+.3f} -> mandatory cooldown/random navigation');target_since=None;target_cooldown_until=now+args.target_cooldown_sec;rgb_hits.clear();area_hist.clear();done_n=0;start_random('NORMAL','target confirmed')
     elif now>=deadline:
      if abs(e)>args.target_deadband:
       turn='TURN_RIGHT' if e>0 else 'TURN_LEFT';scale=min(1,abs(e)/max(args.target_hard_error,1e-6));dur=args.target_turn_min_sec+(args.target_turn_max_sec-args.target_turn_min_sec)*scale;motor.set(turn,f'target steer e={e:+.3f}');deadline=now+dur
      else:
       dur=approach_burst(a,list(area_hist));motor.set('FORWARD',f'target approach area={a:.4f} burst={dur:.2f}s');deadline=now+dur
    elif now-last_rgb_t>=args.target_lost_sec:
     event('TARGET_LOST -> mandatory cooldown/random navigation');target_since=None;target_cooldown_until=now+args.target_cooldown_sec;rgb_hits.clear();area_hist.clear();done_n=0;start_random('NORMAL','target lost')
    elif now>=deadline: motor.set('FORWARD','target evidence gap');deadline=now+.25
   else:
    # RANDOM normal state. Fused observations remain fully logged during cooldown but cannot take over.
    if fused and now<target_cooldown_until:
     event(f'FUSED_SUPPRESSED cooldown_remaining={target_cooldown_until-now:.2f}s RGB={det.get("conf",0):.3f} TH={th.get("delta_c",0):.2f}C')
    elif fused and now>=target_cooldown_until:
     state='TARGET';target_since=now;done_n=0;area_hist.clear();deadline=now
     event(f'FUSED_CANDIDATE RGB={{t={det.get("rel_t",rel(det["t"])):.3f},file={det.get("file","")},conf={det["conf"]:.3f},bearing={det["bearing"]:+.1f},area={det["area"]:.4f}}} TH={{t={th.get("rel_t",rel(th["t"])):.3f},file={th.get("file_png","")},delta={th["delta_c"]:.2f}C,bearing={th["bearing"]:+.1f}}} decision=PURSUIT')
     motor.set('OFF','target takeover');note='TARGET_LOCK'
    elif now>=deadline:start_random('NORMAL','random action complete')
   if now-last_save>=args.save_rgb_sec:
    vis=img.copy();cv2.line(vis,(args.width//2,0),(args.width//2,args.height),(255,255,255),1)
    if det:
     x1,y1,x2,y2=map(int,det['box']);cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.putText(vis,f'{state} {motor.get()} rnd={random_bank}:{random_action}',(10,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2);cv2.imwrite(str(OUT/'rgb'/f'rgb_{rel(cap_t):09.3f}.jpg'),vis);last_save=now
   x,y,h=pose.get();td=det or {};tz=th or {};cool=max(0.,target_cooldown_until-now);remain=max(0.,deadline-now)
   w.writerow([f'{rel(now):.3f}',fid,state,motor.get(),f'{x:.4f}',f'{y:.4f}',f'{h:.2f}',f'{E:.6f}',f'{S:.2f}',hard_n,random_bank,random_action,f'{random_requested:.3f}',f'{remain:.3f}',td.get('conf',''),td.get('error',''),td.get('area',''),tz.get('bearing',''),tz.get('delta_c',''),int(fused),f'{cool:.3f}',note])
   if now-last_status>=args.status_sec:
    print(f'[{rel(now):8.3f}s] {state:18s} motor={motor.get():10s} random={random_bank}:{random_action} rem={remain:.1f}s wall={hard_n} RGB={td.get("conf",0):.2f}/{td.get("area",0):.3f} TH={tz.get("delta_c",0):.2f} fused={int(fused)} cooldown={cool:.1f}s',flush=True);last_status=now
 except KeyboardInterrupt:print('\n*** CTRL-C: MOTORS OFF ***',flush=True)
 finally:
  motor.set('OFF','final shutdown');motors_off();thermal.close();cam.close();motor.close();f.close();(OUT/f'run_{stamp}_config.json').write_text(json.dumps({**vars(args),'effective_random_seed':EFFECTIVE_RANDOM_SEED,'normal_bank':NORMAL_BANK,'collision_bank':COLLISION_BANK},indent=2));event(f'LOG saved {csvp}')
if __name__=='__main__':main()
