import os, math, time, glob
import cv2
import numpy as np
import mediapipe as mp


FRAME_W, FRAME_H = 640, 480
FPS = 20.0

MAX_FACES = 5
USE_DNN   = True
DNN_PROTO = "deploy.prototxt"
DNN_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
DNN_CONF  = 0.60

RENDER_COMBINED   = True
SHOW_SEGMENTATION = False


SAVE_VIDEO = True
OUTPUT_DIR = "output"
TRY_CODECS = ["MJPG", "XVID", "mp4v"]


EAR_BLINK_THRESH          = 0.23
BLINK_CONSEC_FRAMES_DOWN  = 2
BLINK_CONSEC_FRAMES_UP    = 2
BLINK_REFRACTORY_SEC      = 0.22
BLINK_MIN_CLOSED_MS       = 40

EMA_A_FEATURES    = 0.25
EMA_A_POS_BASE    = 0.22
EMA_A_SCALE_BASE  = 0.22
DEAD_EAR          = 0.004
DEAD_MAR          = 0.003
DEAD_POS          = 1.6
DEAD_RAD          = 1.0
MAX_STEP_POS_BASE = 6.0
MAX_STEP_RAD_BASE = 3.0

EMA_A_POS_BOOST    = 0.45
EMA_A_SCALE_BOOST  = 0.45
MAX_STEP_POS_BOOST = 12.0
MAX_STEP_RAD_BOOST = 6.0
MOVE_THRESH_DXDY   = 3.0
MOVE_THRESH_DR     = 2.0


BASELINE_INIT_GUESS    = 0.26
BASELINE_UPDATE_MARGIN = 0.02
BASELINE_EMA_ALPHA     = 0.06
BASELINE_SLOW_DRIFT    = 0.002
DELTA_HARD_CLOSE       = 0.015
DELTA_OPEN_START       = 0.055


SMILE_WIDTH_T          = 1.05
SMILE_OPEN_GUARD_EXTRA = 0.02


RADIUS_MIN = 60
RADIUS_MAX = 160
IOD_TO_RADIUS = 0.95
EMOJI_Y_OFFSET = -10

YELLOW=(0,255,255); BLACK=(0,0,0); WHITE=(255,255,255); GRAYBG=(240,240,240)
BLUE=(255,0,0); GREEN=(0,200,0)
AA = cv2.LINE_AA


def dist(p,q): return math.hypot(p[0]-q[0], p[1]-q[1])

def ema(prev,new,a): return new if prev is None else a*new + (1-a)*prev

def dz(prev,new,d):
    """Zone morte : si la variation est petite, conserve l'ancienne valeur."""
    if prev is not None and abs(new - prev) < d: return prev
    return new

def clamp_step(prev,new,max_step):
    """Borne la variation max par frame (évite les sauts brusques)."""
    if prev is None: return new
    delta = new - prev
    if abs(delta) <= max_step: return new
    return prev + np.sign(delta) * max_step


def compute_ear(pts):
    """EAR moyen des 2 yeux, à partir des indices FaceMesh."""
    try:
        hL=dist(pts[33],pts[133]); vL=(dist(pts[160],pts[144])+dist(pts[158],pts[153]))/2.0
        hR=dist(pts[263],pts[362]); vR=(dist(pts[387],pts[373])+dist(pts[385],pts[380]))/2.0
        return ((vL/(hL+1e-6)) + (vR/(hR+1e-6)))/2.0
    except: return None

def compute_mar(pts):
    """MAR = ouverture verticale / largeur de bouche."""
    try: return dist(pts[13],pts[14]) / (dist(pts[78],pts[308])+1e-6)
    except: return None

def compute_smile(pts, iod, mar, mar_open_thresh):
    """Sourire si bouche 'large' (width/iod) et pas 'trop ouverte' (garde MAR)."""
    try:
        mouth_w = dist(pts[78], pts[308])
        width_ratio = mouth_w / (iod + 1e-6)
        not_too_open = (mar is None) or (mar < (mar_open_thresh + SMILE_OPEN_GUARD_EXTRA))
        smiling = (width_ratio >= SMILE_WIDTH_T) and not_too_open
        return smiling, width_ratio
    except:
        return False, 0.0


def load_background():
    """Charge un fond si présent, sinon un fond gris clair."""
    for p in ["background_fixed.png","background.png",*sorted(glob.glob("background*.png"))]:
        if os.path.exists(p):
            img=cv2.imread(p)
            if img is not None:
                return cv2.resize(img,(FRAME_W,FRAME_H))
    return np.full((FRAME_H,FRAME_W,3), GRAYBG, dtype=np.uint8)


def draw_eyes(canvas, cx, cy, r, ear, eyes_fully_closed):
    """Dessine les yeux (ellipses + pupille) ou un trait si fermés."""
    sclera_rx=int(0.22*r); sclera_ry_open=int(0.13*r); sclera_ry_close=max(2,int(0.05*r))
    eye_dx=int(0.38*r); eye_dy=int(0.25*r); stroke=max(2,int(0.025*r))

    if eyes_fully_closed:
        for ex,ey in [(cx-eye_dx,cy-eye_dy),(cx+eye_dx,cy-eye_dy)]:
            ex,ey=int(ex),int(ey)
            cv2.line(canvas,(ex-sclera_rx,ey),(ex+sclera_rx,ey),BLACK,stroke,AA)
        return

    if ear is None:
        sclera_ry = sclera_ry_open
    else:
        t = np.clip((ear-(EAR_BLINK_THRESH-0.08))/0.18, 0, 1)
        sclera_ry = int(sclera_ry_close + t*(sclera_ry_open - sclera_ry_close))

    for ex,ey in [(cx-eye_dx,cy-eye_dy),(cx+eye_dx,cy-eye_dy)]:
        ex,ey=int(ex),int(ey)
        cv2.ellipse(canvas,(ex,ey),(sclera_rx,sclera_ry),0,0,360,WHITE,-1,AA)
        cv2.ellipse(canvas,(ex,ey),(sclera_rx,sclera_ry),0,0,360,BLACK,stroke,AA)
        if sclera_ry>3:
            prx=max(3,int(0.40*sclera_rx)); pry=max(2,int(0.55*sclera_ry))
            cv2.ellipse(canvas,(ex,ey),(prx,pry),0,0,360,BLACK,-1,AA)

def draw_mouth(canvas, cx, cy, r, mar, mouth_fully_closed, mar_open_thresh):
    """Bouche elliptique (fond jaune + contour noir) ou petit arc si fermée."""
    mouth_rx=int(0.45*r)
    mouth_ry_min=max(8,int(0.08*r))
    mouth_ry_open=int(0.45*r*0.60)
    mouth_dy=int(0.33*r)
    stroke=max(3,int(0.03*r))
    center=(int(cx),int(cy+mouth_dy))

    if mouth_fully_closed:

        n=24; pts=[]
        for i in range(n+1):
            t=i/n; x=int(center[0]-mouth_rx+2*mouth_rx*t); w=(t-0.5)
            y=int(center[1] + (-2)*(w*w))
            pts.append([x,y])
        cv2.polylines(canvas,[np.array(pts,np.int32)],False,BLACK,stroke,AA)
        return

    mar_v=0.0 if mar is None else mar
    open_t=np.clip((mar_v - mar_open_thresh)/0.10, 0.0, 1.0)
    mouth_ry=int(mouth_ry_min + open_t*(mouth_ry_open - mouth_ry_min))

    cv2.ellipse(canvas, center, (mouth_rx, mouth_ry), 0, 0, 360, YELLOW, -1, AA)
    cv2.ellipse(canvas, center, (mouth_rx, mouth_ry), 0, 0, 360, BLACK,  stroke, AA)

def draw_emoji(canvas, cx, cy, r, ear, mar, eyes_closed, mouth_closed, mar_open_thresh):
    """Disque jaune + yeux + bouche."""
    cx=int(np.clip(cx, r, FRAME_W-r)); cy=int(np.clip(cy, r, FRAME_H-r))
    cv2.circle(canvas,(cx,cy),int(r),YELLOW,-1,AA)
    draw_eyes(canvas,cx,cy,r,ear,eyes_closed)
    draw_mouth(canvas,cx,cy,r,mar,mouth_closed,mar_open_thresh)

def draw_face_status(canvas, cx, cy, r, eyes_closed, mouth_closed, blinks, smiling):
    """Texte d'état en anglais, sous l'emoji (demandé)."""
    txt = f"Eyes: {'closed' if eyes_closed else 'open'} | Mouth: {'closed' if mouth_closed else 'open'} | Blink: {blinks} | Smile: {'yes' if smiling else 'no'}"
    x = int(np.clip(cx - r, 10, FRAME_W - 10))
    y = int(np.clip(cy + r + 22, 20, FRAME_H - 5))
    cv2.putText(canvas, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 2, AA)


OVAL_IDX=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10]
def face_mask_from_pts(shape_hw, pts):
    mask=np.zeros(shape_hw[:2],dtype=np.uint8)
    hull=np.array([pts[i] for i in OVAL_IDX if i < len(pts)],np.int32)
    if len(hull)>=3: cv2.fillConvexPoly(mask,hull,255)
    return mask


def init_dnn():
    if not USE_DNN: return None
    if not (os.path.exists(DNN_PROTO) and os.path.exists(DNN_MODEL)):
        print("[DNN] Fichiers absents -> DNN désactivé (fallback)."); return None
    try:
        net=cv2.dnn.readNetFromCaffe(DNN_PROTO,DNN_MODEL)
        print("[DNN] Modèle chargé.")
        return net
    except Exception as e:
        print("[DNN] Échec de chargement :", e); return None

def dnn_faces(net, frame_bgr):
    if net is None: return []
    h,w=frame_bgr.shape[:2]
    blob=cv2.dnn.blobFromImage(cv2.resize(frame_bgr,(300,300)),1.0,(300,300),(104,177,123),False,False)
    net.setInput(blob); det=net.forward(); boxes=[]
    for i in range(det.shape[2]):
        conf=float(det[0,0,i,2])
        if conf>=DNN_CONF:
            x1,y1,x2,y2=(det[0,0,i,3:7]*np.array([w,h,w,h])).astype(int)
            x1,y1=max(0,x1),max(0,y1); x2,y2=min(w-1,x2),min(h-1,y2)
            if x2>x1 and y2>y1: boxes.append([x1,y1,x2,y2,conf])
    return boxes

def point_in_any_box(pt, boxes, margin=6):
    x,y=pt
    for (x1,y1,x2,y2,_) in boxes:
        if (x1-margin)<=x<=(x2+margin) and (y1-margin)<=y<=(y2+margin): return True
    return False


class FaceState:
    def __init__(self):
        self.ear_e=None; self.mar_e=None
        self.cx=None; self.cy=None; self.r=120.0

        self.blinks=0; self._down_cnt=0; self._up_cnt=0
        self._is_closed=False; self._last_blink_ts=0.0; self._closed_start_ts=None

        self.eyes_closed=False; self.mouth_closed=False

        self.mar_baseline=None
        self.last_seen=time.time()

    def update_metrics(self, ear, mar):

        if ear is not None:
            ear=dz(self.ear_e,ear,DEAD_EAR); self.ear_e=ema(self.ear_e,ear,EMA_A_FEATURES)
        if mar is not None:
            mar=dz(self.mar_e,mar,DEAD_MAR); self.mar_e=ema(self.mar_e,mar,EMA_A_FEATURES)


        now = time.time()
        if self.ear_e is None: return
        if self.ear_e < EAR_BLINK_THRESH:
            self._down_cnt += 1; self._up_cnt = 0
            if not self._is_closed and self._down_cnt >= BLINK_CONSEC_FRAMES_DOWN:
                self._is_closed = True; self._closed_start_ts = now
        else:
            self._up_cnt += 1; self._down_cnt = 0
            if self._is_closed and self._up_cnt >= BLINK_CONSEC_FRAMES_UP:
                dur_ok = True
                if self._closed_start_ts is not None:
                    dur_ok = (now - self._closed_start_ts)*1000.0 >= BLINK_MIN_CLOSED_MS
                refractory_ok = (now - self._last_blink_ts) >= BLINK_REFRACTORY_SEC
                if dur_ok and refractory_ok:
                    self.blinks += 1; self._last_blink_ts = now
                self._is_closed = False; self._closed_start_ts = None

    def update_baseline(self):
        if self.mar_e is None: return
        if self.mar_baseline is None:
            self.mar_baseline = self.mar_e if self.mar_e < 0.5 else BASELINE_INIT_GUESS
        else:
            if self.mar_e <= self.mar_baseline + BASELINE_UPDATE_MARGIN:
                self.mar_baseline = (1-BASELINE_EMA_ALPHA)*self.mar_baseline + BASELINE_EMA_ALPHA*self.mar_e
            else:
                self.mar_baseline = (1-BASELINE_SLOW_DRIFT)*self.mar_baseline + BASELINE_SLOW_DRIFT*self.mar_e

    def thresholds(self):
        mb = self.mar_baseline if self.mar_baseline is not None else BASELINE_INIT_GUESS
        return (mb + DELTA_HARD_CLOSE, mb + DELTA_OPEN_START)


FRAME_SIZE=(FRAME_W,FRAME_H)
bg=load_background()


cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,FRAME_H)
if not cap.isOpened(): raise RuntimeError("Webcam indisponible.")


mp_face_mesh=mp.solutions.face_mesh
mesh=mp_face_mesh.FaceMesh(max_num_faces=MAX_FACES, refine_landmarks=True,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)


dnn_net=init_dnn()


def _open_writer(path, size, fps, codec):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    w = cv2.VideoWriter(path, fourcc, fps, size)
    print(f"[VIDEO] Open {os.path.basename(path)} codec={codec}: {'OK' if w.isOpened() else 'FAIL'}")
    return w

writers = {}
if SAVE_VIDEO:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    AVATAR_SIZE   = (FRAME_W, FRAME_H)
    WEBCAM_SIZE   = (FRAME_W, FRAME_H)
    COMBINED_SIZE = (FRAME_W*2, FRAME_H)

    out_avatar   = os.path.join(OUTPUT_DIR, "avatar.avi")
    out_webcam   = os.path.join(OUTPUT_DIR, "webcam.avi")
    out_combined = os.path.join(OUTPUT_DIR, "combined.avi")

    avatar_w = webcam_w = combined_w = None
    for codec in TRY_CODECS:
        if avatar_w is None or not avatar_w.isOpened():
            avatar_w = _open_writer(out_avatar, AVATAR_SIZE, FPS, codec)
        if webcam_w is None or not webcam_w.isOpened():
            webcam_w = _open_writer(out_webcam, WEBCAM_SIZE, FPS, codec)
        if combined_w is None or not combined_w.isOpened():
            combined_w = _open_writer(out_combined, COMBINED_SIZE, FPS, codec)

        if avatar_w.isOpened() and webcam_w.isOpened() and combined_w.isOpened():
            print(f"[VIDEO] ✅ Tous les writers ouverts avec codec={codec}")
            break

    writers = {"avatar": avatar_w, "webcam": webcam_w, "combined": combined_w}
    for name, w in writers.items():
        if not w.isOpened():
            print(f"[VIDEO][WARN] Writer '{name}' non ouvert. (Essayés: {TRY_CODECS})")
            print("               ➤ Astuce: garder .avi et préférer 'MJPG' ou 'XVID'.")


faces={}; next_id=0; prev_centers={}; MAX_IDLE=3.0
print("[INFO] Exécution... (ESC pour quitter)")
while True:
    ok, frame = cap.read()
    if not ok: break
    frame=cv2.resize(frame,FRAME_SIZE)
    frame=cv2.flip(frame,1)


    boxes=dnn_faces(dnn_net, frame) if dnn_net is not None else []
    for (x1,y1,x2,y2,conf) in boxes:
        cv2.rectangle(frame,(x1,y1),(x2,y2),BLUE,1,AA)
        cv2.putText(frame,f"{conf:.2f}",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,BLUE,1,AA)


    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res=mesh.process(rgb)
    canvas=bg.copy()
    now=time.time()


    for fid in list(faces.keys()):
        if now - faces[fid].last_seen > MAX_IDLE:
            faces.pop(fid,None); prev_centers.pop(fid,None)


    centers=[]; bundle=[]
    if res.multi_face_landmarks:
        h,w=frame.shape[:2]
        for fm in res.multi_face_landmarks:
            pts=[(int(p.x*w),int(p.y*h)) for p in fm.landmark]
            L,R,nose=pts[33],pts[263],pts[1]
            midx=(L[0]+R[0])/2.0; midy=(L[1]+R[1])/2.0
            a=0.35
            cx=(1-a)*midx + a*nose[0]
            cy=(1-a)*midy + a*(nose[1]-8) + EMOJI_Y_OFFSET
            iod=dist(L,R)
            r=np.clip(IOD_TO_RADIUS*iod, RADIUS_MIN, RADIUS_MAX)

            if dnn_net is not None and len(boxes)>0:
                if not point_in_any_box((int(cx),int(cy)), boxes):
                    continue

            centers.append((cx,cy)); bundle.append((pts,cx,cy,r,iod))


    used_prev=set(); mapping={}
    for i,(cx,cy) in enumerate(centers):
        best=None;bestd=1e9
        for fid,(px,py) in prev_centers.items():
            if fid in used_prev: continue
            d=dist((cx,cy),(px,py))
            if d<bestd: bestd=d; best=fid
        if best is not None and bestd<=80:
            mapping[i]=best; used_prev.add(best)
        else:
            mapping[i]=None


    seen=set()
    for i,(pts,cx_raw,cy_raw,rad_raw,iod) in enumerate(bundle):
        fid=mapping[i]
        if fid is None:
            fid=next_id; next_id+=1; faces[fid]=FaceState()
        st=faces[fid]; seen.add(fid)
        prev_centers[fid]=(cx_raw,cy_raw); st.last_seen=now


        ear_raw=compute_ear(pts); mar_raw=compute_mar(pts)
        st.update_metrics(ear_raw,mar_raw)


        cx_raw=dz(st.cx,cx_raw,DEAD_POS); cy_raw=dz(st.cy,cy_raw,DEAD_POS); rad_raw=dz(st.r,rad_raw,DEAD_RAD)
        dx=abs(cx_raw-(st.cx if st.cx is not None else cx_raw))
        dy=abs(cy_raw-(st.cy if st.cy is not None else cy_raw))
        dr=abs(rad_raw-(st.r  if st.r  is not None else rad_raw))
        a_pos=EMA_A_POS_BASE; a_scl=EMA_A_SCALE_BASE
        step_pos=MAX_STEP_POS_BASE; step_rad=MAX_STEP_RAD_BASE
        if dx>MOVE_THRESH_DXDY or dy>MOVE_THRESH_DXDY or dr>MOVE_THRESH_DR:
            a_pos=max(a_pos,EMA_A_POS_BOOST); a_scl=max(a_scl,EMA_A_SCALE_BOOST)
            step_pos=max(step_pos,MAX_STEP_POS_BOOST); step_rad=max(step_rad,MAX_STEP_RAD_BOOST)
        cx_tmp=ema(st.cx,cx_raw,a_pos); cy_tmp=ema(st.cy,cy_raw,a_pos); r_tmp=ema(st.r,rad_raw,a_scl)
        st.cx=clamp_step(st.cx,cx_tmp,step_pos); st.cy=clamp_step(st.cy,cy_tmp,step_pos); st.r=clamp_step(st.r,r_tmp,step_rad)


        st.update_baseline()
        mouth_hard_close, mar_open_thresh = st.thresholds()


        if st.ear_e is not None:
            if not st.eyes_closed and st.ear_e < 0.18: st.eyes_closed=True
            elif st.eyes_closed and st.ear_e > 0.20:   st.eyes_closed=False
        if st.mar_e is not None:
            if not st.mouth_closed and st.mar_e < mouth_hard_close: st.mouth_closed=True
            elif st.mouth_closed and st.mar_e > (mouth_hard_close+0.03): st.mouth_closed=False


        smiling, _ = compute_smile(pts, iod, st.mar_e, mar_open_thresh)


        if SHOW_SEGMENTATION:
            mask=face_mask_from_pts(frame.shape, pts)
            seg=cv2.bitwise_and(frame,frame,mask=mask)
            canvas=cv2.addWeighted(canvas,1.0,seg,0.5,0)


        draw_emoji(canvas, st.cx, st.cy, st.r, st.ear_e, st.mar_e,
                   st.eyes_closed, st.mouth_closed, mar_open_thresh)
        draw_face_status(canvas, st.cx, st.cy, st.r, st.eyes_closed, st.mouth_closed, st.blinks, smiling)


        cv2.circle(frame,(int(st.cx),int(st.cy)),3,GREEN,-1,AA)


    faces_count=len(bundle)
    cv2.putText(canvas, f"Faces: {faces_count}", (15, FRAME_H-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 2, AA)

    cv2.imshow("avatar", canvas)
    if RENDER_COMBINED:
        combined=np.hstack((frame,canvas))
        cv2.imshow("combined", combined)
    else:
        cv2.imshow("webcam", frame)


    if SAVE_VIDEO and writers:
        if writers["avatar"].isOpened():
            writers["avatar"].write(canvas)
        if RENDER_COMBINED and writers["combined"].isOpened():

            combined_for_write = np.hstack((frame, canvas))
            writers["combined"].write(combined_for_write)
        elif writers["webcam"].isOpened():
            writers["webcam"].write(frame)


    if cv2.waitKey(1)&0xFF==27: break


cap.release()
cv2.destroyAllWindows()
if SAVE_VIDEO and writers:
    for name,w in writers.items():
        try: w.release()
        except: pass
