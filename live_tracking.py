# live_tracking.py
# Two-eye inference (50x200), cached cubic calibration (5x5 dots),
# median→EMA smoothing + dead-zone, optional dwell zoom,
# blink-to-click with freeze-at-stable.

# --- Windows DPI awareness (ensures real pixel coords) ---
import sys
if sys.platform.startswith("win"):
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import os
import torch, pyautogui, cv2 as cv, mediapipe as mp
import numpy as np
from torchvision import transforms
from PIL import Image
from collections import deque
import math

# =========================
# Screen & safety
# =========================
W, H = pyautogui.size()
SAFE_MARGIN = 10
def clamp(v, lo, hi): return max(lo, min(hi, v))

# =========================
# Model (must match training: 1x50x200, Sigmoid output)
# =========================
class EyeGazeCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 5, 1, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, 1, 2)
        self.pool  = torch.nn.MaxPool2d(2, 2, 0)
        # (1,50,200) -> (32,25,100) -> (64,12,50)
        self.fc1   = torch.nn.Linear(64*12*50, 512)
        self.fc2   = torch.nn.Linear(512, 128)
        self.fc3   = torch.nn.Linear(128, 2)
        self.out   = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(self.fc3(x))           # [0,1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EyeGazeCNN().to(device)
state = torch.load('best_eye_gaze_model.pth', map_location=device)
model.load_state_dict(state)
model.eval()

transform = transforms.Compose([
    transforms.Resize((50, 200)),             # two-eye width
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# =========================
# MediaPipe FaceMesh
# =========================
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
)

LEFT_IDS  = [33,133,159,145,158,144,153,154,155]
RIGHT_IDS = [362,263,386,374,385,373,380,381,382]

def _dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def _ear_from_landmarks(lm, w, h, is_left=True):
    # Simple EAR-like ratio using corners + one upper/lower
    if is_left:
        corners = (33, 133); upper = 159; lower = 145
    else:
        corners = (362, 263); upper = 386; lower = 374
    cx1, cy1 = int(lm[corners[0]].x*w), int(lm[corners[0]].y*h)
    cx2, cy2 = int(lm[corners[1]].x*w), int(lm[corners[1]].y*h)
    ux, uy   = int(lm[upper].x*w),      int(lm[upper].y*h)
    lx, ly   = int(lm[lower].x*w),      int(lm[lower].y*h)
    hlen = _dist((cx1, cy1), (cx2, cy2)) + 1e-6
    vlen = _dist((ux, uy), (lx, ly))
    return vlen / hlen  # smaller -> closed

def get_both_eyes_and_ear(frame):
    """Return (both-eye 50x200 grayscale, avg_EAR) or (None, None)."""
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None, None
    lm = res.multi_face_landmarks[0].landmark
    h, w = frame.shape[:2]

    def eye_bbox(ids, extra_bottom=0.55, extra_top=0.30, extra_x=0.25):
        xs = [int(lm[i].x*w) for i in ids]; ys = [int(lm[i].y*h) for i in ids]
        x1,x2 = max(0,min(xs)), min(w-1,max(xs))
        y1,y2 = max(0,min(ys)), min(h-1,max(ys))
        pad_x = int(extra_x*(x2-x1+1))
        pad_t = int(extra_top*(y2-y1+1))
        pad_b = int(extra_bottom*(y2-y1+1))
        x1 = max(0,x1-pad_x); x2 = min(w-1,x2+pad_x)
        y1 = max(0,y1-pad_t); y2 = min(h-1,y2+pad_b)
        return x1,y1,x2,y2

    lx1,ly1,lx2,ly2 = eye_bbox(LEFT_IDS)
    rx1,ry1,rx2,ry2 = eye_bbox(RIGHT_IDS)
    if lx2<=lx1 or ly2<=ly1 or rx2<=rx1 or ry2<=ry1:
        return None, None

    L = cv.cvtColor(frame[ly1:ly2, lx1:lx2], cv.COLOR_BGR2GRAY)
    R = cv.cvtColor(frame[ry1:ry2, rx1:rx2], cv.COLOR_BGR2GRAY)
    L = cv.resize(L, (100,50)); R = cv.resize(R, (100,50))
    both = np.hstack([L, R])

    ear_left  = _ear_from_landmarks(lm, w, h, is_left=True)
    ear_right = _ear_from_landmarks(lm, w, h, is_left=False)
    ear = 0.5*(ear_left + ear_right)
    return both, ear

def predict_norm_from_frame(frame):
    eye, _ = get_both_eyes_and_ear(frame)
    if eye is None: return None
    pil = Image.fromarray(eye)
    ten = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        nx, ny = model(ten).squeeze(0).cpu().numpy().tolist()
    return nx, ny, eye

# =========================
# Cubic calibration (degree-3 poly) + caching
# =========================
RUN_CALIBRATION = False  # don't force on every start
CALIB_DIR = ".calib"
os.makedirs(CALIB_DIR, exist_ok=True)

def calib_path():
    return os.path.join(CALIB_DIR, f"cubic_{W}x{H}.npz")

def save_calibration(wx, wy):
    np.savez(calib_path(), wx=wx, wy=wy)

def load_calibration():
    p = calib_path()
    if os.path.exists(p):
        d = np.load(p)
        return d["wx"].astype(np.float32), d["wy"].astype(np.float32)
    return None

def poly_features_cubic(nx, ny):
    x, y = nx, ny
    return np.array([1.0, x, y, x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, y*y*y], dtype=np.float32)

def solve_cubic(xs_norm, ys_pix):
    Phi = np.vstack([poly_features_cubic(nx, ny) for nx, ny in xs_norm])  # N x 10
    wx = np.linalg.lstsq(Phi, ys_pix[:, 0], rcond=None)[0]
    wy = np.linalg.lstsq(Phi, ys_pix[:, 1], rcond=None)[0]
    return wx.astype(np.float32), wy.astype(np.float32)

def apply_cubic(nx, ny, wx, wy):
    phi = poly_features_cubic(nx, ny)
    return float(phi @ wx), float(phi @ wy)

def show_calib_dot(x, y, msg="Look at the dot and press SPACE"):
    blk = np.zeros((H, W, 3), dtype=np.uint8)
    cv.circle(blk, (int(x), int(y)), 12, (0, 0, 255), -1)
    cv.putText(blk, msg, (40, H-60), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv.imshow("Calib", blk); cv.waitKey(1)

def run_quick_calibration(cam):
    m = 60
    xs = np.linspace(m, W - m, 5, dtype=int)
    ys = np.linspace(m, H - m, 5, dtype=int)
    pts = [(x,y) for y in ys for x in xs]  # 25 points

    cv.namedWindow("Calib", cv.WINDOW_NORMAL)
    cv.setWindowProperty("Calib", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    preds, pix = [], []
    for (x,y) in pts:
        show_calib_dot(x,y)
        while True:
            k = cv.waitKey(10) & 0xFF
            if k == 27: cv.destroyWindow("Calib"); return None
            if k == 32: break  # SPACE
        samples = []
        for _ in range(15):
            ok, frame = cam.read()
            if not ok: continue
            out = predict_norm_from_frame(frame)
            if out is not None:
                nx, ny, _ = out
                samples.append([nx, ny])
            cv.waitKey(10)
        if samples:
            preds.append(np.mean(np.array(samples), axis=0))
            pix.append([x, y])
    cv.destroyWindow("Calib")
    wx, wy = solve_cubic(np.array(preds), np.array(pix))
    return (wx, wy)

# =========================
# Dwell zoom (optional)
# =========================
ENABLE_DWELL_ZOOM = True
DWELL_RADIUS = 35      # px
DWELL_TIME_MS = 500    # ms
ZOOM_SIZE = 160        # capture box
ZOOM_SCALE = 3

def in_radius(ax, ay, bx, by, r):
    return (ax-bx)**2 + (ay-by)**2 <= r*r

# =========================
# Blink-to-click (+ freeze)
# =========================
ENABLE_BLINK_CLICK = True
BLINK_EAR_THR = 0.19     # tune 0.17–0.23
BLINK_MIN_MS  = 1000     # 1s hold to click
BLINK_HYST    = 0.02
FREEZE_ON_BLINK = True

# =========================
# Main loop
# =========================
def main():
    global ENABLE_DWELL_ZOOM
    cam = cv.VideoCapture(1)
    if not cam.isOpened(): cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("ERROR: Cannot open camera."); return

    # Load cached calibration (or run if requested)
    wx_wy = load_calibration()
    if wx_wy is None:
        print("No cached calibration found.")
        if RUN_CALIBRATION:
            print("Calibration... SPACE at each dot (ESC to cancel)")
            out = run_quick_calibration(cam)
            if out is None:
                print("Calibration cancelled. Using raw mapping.")
            else:
                wx_wy = out
                save_calibration(wx_wy[0], wx_wy[1])
                print("Calibration complete and saved.")
        else:
            print("Skipping calibration (RUN_CALIBRATION=False). Using raw mapping.")
    else:
        print("Loaded cached calibration.")

    hist = deque(maxlen=9)   # median window
    alpha = 0.85             # EMA factor
    sx, sy = W/2, H/2        # EMA state
    DEAD = 6                 # px dead-zone
    out_x, out_y = sx, sy    # cursor position we actually set

    # remember last stable (non-blink) point for clicking
    last_stable_x, last_stable_y = out_x, out_y

    # dwell state
    dwell_anchor = (sx, sy)
    dwell_start = None

    # blink state
    blink_start = None
    blink_fired = False
    is_blinking = False

    try:
        while True:
            ok, frame = cam.read()
            if not ok: continue

            # eyes + EAR
            eye_img, ear = get_both_eyes_and_ear(frame)
            if eye_img is None:
                cv.imshow('Both Eyes (50x200)', np.zeros((50,200), dtype=np.uint8))
                if cv.waitKey(1) & 0xFF == ord('q'): break
                continue

            # predict normalized gaze
            pil = Image.fromarray(eye_img)
            ten = transform(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                nx, ny = model(ten).squeeze(0).cpu().numpy().tolist()

            # map to pixels (use calibration if available)
            if wx_wy is not None:
                fx, fy = apply_cubic(nx, ny, wx_wy[0], wx_wy[1])
            else:
                fx, fy = nx*W, ny*H

            # ---- BLINK GATING ----
            tnow = cv.getTickCount()
            if ENABLE_BLINK_CLICK and (ear is not None):
                if ear < BLINK_EAR_THR:
                    if not is_blinking:
                        is_blinking = True
                        blink_start = tnow
                        blink_fired = False
                        click_x, click_y = last_stable_x, last_stable_y
                elif ear > (BLINK_EAR_THR + BLINK_HYST):
                    is_blinking = False
                    blink_start = None
                    blink_fired = False

            # ---- GAZE UPDATE (freeze if blinking) ----
            if not (FREEZE_ON_BLINK and is_blinking):
                hist.append((fx, fy))
                mx = np.median([p[0] for p in hist])
                my = np.median([p[1] for p in hist])

                sx = alpha*sx + (1 - alpha)*mx
                sy = alpha*sy + (1 - alpha)*my

                fx = clamp(sx, SAFE_MARGIN, W-1-SAFE_MARGIN)
                fy = clamp(sy, SAFE_MARGIN, H-1-SAFE_MARGIN)

                if np.hypot(fx - out_x, fy - out_y) >= DEAD:
                    out_x, out_y = fx, fy

                last_stable_x, last_stable_y = out_x, out_y
            else:
                # freeze during blink
                out_x, out_y = last_stable_x, last_stable_y
                if blink_start is not None and not blink_fired:
                    elapsed_ms = (tnow - blink_start) / cv.getTickFrequency() * 1000.0
                    if elapsed_ms >= BLINK_MIN_MS:
                        pyautogui.click(int(last_stable_x), int(last_stable_y))
                        blink_fired = True

            # move mouse
            print(f"EAR={ear:.3f}  norm=({nx:.3f},{ny:.3f})  px=({out_x:.0f},{out_y:.0f})  blink={is_blinking}")
            try:
                pyautogui.moveTo(int(out_x), int(out_y))
            except pyautogui.FailSafeException:
                pyautogui.moveTo(W//2, H//2)

            # dwell zoom
            if ENABLE_DWELL_ZOOM:
                if in_radius(out_x, out_y, dwell_anchor[0], dwell_anchor[1], DWELL_RADIUS):
                    if dwell_start is None:
                        dwell_start = cv.getTickCount()
                    else:
                        elapsed_ms = (cv.getTickCount()-dwell_start) / cv.getTickFrequency() * 1000.0
                        if elapsed_ms >= DWELL_TIME_MS:
                            half = ZOOM_SIZE//2
                            rx = int(clamp(out_x-half, 0, W-ZOOM_SIZE))
                            ry = int(clamp(out_y-half, 0, H-ZOOM_SIZE))
                            snap = pyautogui.screenshot(region=(rx, ry, ZOOM_SIZE, ZOOM_SIZE))
                            snap = cv.cvtColor(np.array(snap), cv.COLOR_RGB2BGR)
                            snap = cv.resize(snap, (ZOOM_SIZE*ZOOM_SCALE, ZOOM_SIZE*ZOOM_SCALE), interpolation=cv.INTER_NEAREST)
                            cv.imshow("Zoom", snap)
                else:
                    dwell_anchor = (out_x, out_y)
                    dwell_start = None
                    try:
                        if cv.getWindowProperty("Zoom", cv.WND_PROP_VISIBLE) > 0:
                            cv.destroyWindow("Zoom")
                    except:
                        pass

            # debug eye view
            dbg = cv.cvtColor(eye_img, cv.COLOR_GRAY2BGR)
            cv.putText(dbg, "BLINK" if is_blinking else "LOOK", (6, 46), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv.imshow('Both Eyes (50x200)', dbg)

            # keys
            k = cv.waitKey(1) & 0xFF
            if k == ord('q'): break
            if k == ord('r'):  # recalibrate
                out = run_quick_calibration(cam)
                if out is not None:
                    wx_wy = out
                    save_calibration(wx_wy[0], wx_wy[1])
                    print("Re-calibrated and saved.")
            if k == ord('z'):  # toggle dwell zoom
                ENABLE_DWELL_ZOOM = not ENABLE_DWELL_ZOOM
                if not ENABLE_DWELL_ZOOM:
                    try:
                        if cv.getWindowProperty("Zoom", cv.WND_PROP_VISIBLE) > 0:
                            cv.destroyWindow("Zoom")
                    except:
                        pass

    finally:
        cam.release()
        face_mesh.close()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
