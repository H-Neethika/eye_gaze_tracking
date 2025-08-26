# live_tracking.py
# Two-eye inference (50x200), cached cubic calibration (5x5 dots),
# median→EMA smoothing + dead-zone, optional dwell zoom,
# blink-to-click with freeze-at-stable, Win32 pixel-true cursor control,
# and quick bias-trim hotkeys (h/j/k/l to nudge, b to reset).

# --- Windows DPI awareness (ensures real pixel coords) ---
import sys
if sys.platform.startswith("win"):
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import os
import math
from collections import deque

import cv2 as cv
import mediapipe as mp
import numpy as np
import pyautogui
import torch
from PIL import Image
from torchvision import transforms

# =========================
# Pixel-true screen & cursor helpers
# =========================
def get_screen_size():
    """
    Returns real pixel width/height of the primary display.
    On Windows, uses Win32 GetSystemMetrics after SetProcessDPIAware.
    """
    if sys.platform.startswith("win"):
        try:
            import ctypes
            user32 = ctypes.windll.user32
            # Ensure DPI awareness (redundant call is harmless)
            user32.SetProcessDPIAware()
            return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        except Exception:
            pass
    # Fallback for other platforms
    w, h = pyautogui.size()
    return int(w), int(h)


def set_cursor_pos(x, y, W, H):
    """
    Move mouse cursor to (x,y) in real pixels without DPI scaling drift.
    """
    x = int(max(0, min(x, W - 1)))
    y = int(max(0, min(y, H - 1)))
    if sys.platform.startswith("win"):
        import ctypes
        ctypes.windll.user32.SetCursorPos(x, y)
    else:
        pyautogui.moveTo(x, y)


def click_at(x, y, W, H):
    """
    Left-click at (x,y) in real pixels.
    """
    x = int(max(0, min(x, W - 1)))
    y = int(max(0, min(y, H - 1)))
    if sys.platform.startswith("win"):
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetCursorPos(x, y)
        MOUSEEVENTF_LEFTDOWN = 0x0002
        MOUSEEVENTF_LEFTUP = 0x0004
        user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    else:
        pyautogui.click(x, y)


# =========================
# Screen & safety
# =========================
W, H = get_screen_size()
SAFE_MARGIN = 10


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# =========================
# Bias trim (optional micro-nudge)
# =========================
BIAS_TRIM = [0.0, 0.0]  # [dx, dy] in pixels


def apply_bias(x, y):
    return x + BIAS_TRIM[0], y + BIAS_TRIM[1]


# =========================
# Model (must match training: 1x50x200, Sigmoid output)
# =========================
class EyeGazeCNN(torch.nn.Module):
    def _init_(self):
        super()._init_()
        self.conv1 = torch.nn.Conv2d(1, 32, 5, 1, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, 1, 2)
        self.pool = torch.nn.MaxPool2d(2, 2, 0)
        # (1,50,200) -> (32,25,100) -> (64,12,50)
        self.fc1 = torch.nn.Linear(64 * 12 * 50, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 2)
        self.out = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(self.fc3(x))  # [0,1]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EyeGazeCNN().to(device)
state = torch.load("best_eye_gaze_model.pth", map_location=device)
model.load_state_dict(state)
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize((50, 200)),  # two-eye width
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

# =========================
# MediaPipe FaceMesh
# =========================
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
)

LEFT_IDS = [33, 133, 159, 145, 158, 144, 153, 154, 155]
RIGHT_IDS = [362, 263, 386, 374, 385, 373, 380, 381, 382]


def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _ear_from_landmarks(lm, w, h, is_left=True):
    # Simple EAR-like ratio using corners + one upper/lower
    if is_left:
        corners = (33, 133)
        upper = 159
        lower = 145
    else:
        corners = (362, 263)
        upper = 386
        lower = 374
    cx1, cy1 = int(lm[corners[0]].x * w), int(lm[corners[0]].y * h)
    cx2, cy2 = int(lm[corners[1]].x * w), int(lm[corners[1]].y * h)
    ux, uy = int(lm[upper].x * w), int(lm[upper].y * h)
    lx, ly = int(lm[lower].x * w), int(lm[lower].y * h)
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
        xs = [int(lm[i].x * w) for i in ids]
        ys = [int(lm[i].y * h) for i in ids]
        x1, x2 = max(0, min(xs)), min(w - 1, max(xs))
        y1, y2 = max(0, min(ys)), min(h - 1, max(ys))
        pad_x = int(extra_x * (x2 - x1 + 1))
        pad_t = int(extra_top * (y2 - y1 + 1))
        pad_b = int(extra_bottom * (y2 - y1 + 1))
        x1 = max(0, x1 - pad_x)
        x2 = min(w - 1, x2 + pad_x)
        y1 = max(0, y1 - pad_t)
        y2 = min(h - 1, y2 + pad_b)
        return x1, y1, x2, y2

    lx1, ly1, lx2, ly2 = eye_bbox(LEFT_IDS)
    rx1, ry1, rx2, ry2 = eye_bbox(RIGHT_IDS)
    if lx2 <= lx1 or ly2 <= ly1 or rx2 <= rx1 or ry2 <= ry1:
        return None, None

    L = cv.cvtColor(frame[ly1:ly2, lx1:lx2], cv.COLOR_BGR2GRAY)
    R = cv.cvtColor(frame[ry1:ry2, rx1:rx2], cv.COLOR_BGR2GRAY)
    L = cv.resize(L, (100, 50))
    R = cv.resize(R, (100, 50))
    both = np.hstack([L, R])

    ear_left = _ear_from_landmarks(lm, w, h, is_left=True)
    ear_right = _ear_from_landmarks(lm, w, h, is_left=False)
    ear = 0.5 * (ear_left + ear_right)
    return both, ear


def predict_norm_from_frame(frame):
    out = get_both_eyes_and_ear(frame)
    if out is None:
        return None
    eye, _ = out
    if eye is None:
        return None
    pil = Image.fromarray(eye)
    ten = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        nx, ny = model(ten).squeeze(0).cpu().numpy().tolist()
    return nx, ny, eye


# =========================
# Cubic calibration (degree-3 poly) + caching
# =========================
RUN_CALIBRATION = True  # Set True for first run; after it saves, set to False
CALIB_DIR = ".calib"
os.makedirs(CALIB_DIR, exist_ok=True)


def calib_path():
    return os.path.join(CALIB_DIR, f"cubic_{W}x{H}.npz")


def save_calibration(wx, wy, A, t):
    np.savez(calib_path(), wx=wx, wy=wy, A=A, t=t)

def load_calibration():
    p = calib_path()
    if not os.path.exists(p):
        return None
    d = np.load(p)
    wx = d["wx"].astype(np.float32)
    wy = d["wy"].astype(np.float32)
    A  = d["A"].astype(np.float32) if "A" in d else np.eye(2, dtype=np.float32)
    t  = d["t"].astype(np.float32) if "t" in d else np.zeros(2, dtype=np.float32)
    return wx, wy, A, t






def poly_features_cubic(nx, ny):
    x, y = nx, ny
    return np.array(
        [1.0, x, y, x * x, x * y, y * y, x * x * x, x * x * y, x * y * y, y * y * y],
        dtype=np.float32,
    )


def solve_cubic(xs_norm, ys_pix, lam=1e-3):
    Phi = np.vstack([poly_features_cubic(nx, ny) for nx, ny in xs_norm]).astype(np.float32)  # N x 10
    Y  = np.asarray(ys_pix, dtype=np.float32)  # N x 2
    A  = Phi.T @ Phi + lam * np.eye(Phi.shape[1], dtype=np.float32)
    wx = np.linalg.solve(A, Phi.T @ Y[:,0])
    wy = np.linalg.solve(A, Phi.T @ Y[:,1])
    return wx.astype(np.float32), wy.astype(np.float32)

def fit_affine(X_pred, Y_true):
    # X_pred, Y_true: (N,2) pixels
    N = X_pred.shape[0]
    M = np.concatenate([X_pred, np.ones((N,1), np.float32)], axis=1)  # (N,3)
    # Solve for each axis: M @ [a11 a12 tx]^T = Yx, M @ [a21 a22 ty]^T = Yy
    pa, _, _, _ = np.linalg.lstsq(M, Y_true[:,0], rcond=None)
    pb, _, _, _ = np.linalg.lstsq(M, Y_true[:,1], rcond=None)
    A = np.array([[pa[0], pa[1]],[pb[0], pb[1]]], dtype=np.float32)
    t = np.array([pa[2], pb[2]], dtype=np.float32)
    return A, t

def apply_affine(p, A, t):
    return (A @ p) + t


def apply_cubic(nx, ny, wx, wy):
    phi = poly_features_cubic(nx, ny)
    return float(phi @ wx), float(phi @ wy)


def show_calib_dot(x, y, msg="Look at the dot and press SPACE"):
    blk = np.zeros((H, W, 3), dtype=np.uint8)
    cv.circle(blk, (int(x), int(y)), 12, (0, 0, 255), -1)
    cv.putText(blk, msg, (40, H - 60), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv.imshow("Calib", blk)
    cv.waitKey(1)


def run_quick_calibration(cam):
    m = 60
    xs = np.linspace(m, W - m, 5, dtype=int)
    ys = np.linspace(m, H - m, 5, dtype=int)
    pts = [(x, y) for y in ys for x in xs]  # 25 points

    cv.namedWindow("Calib", cv.WINDOW_NORMAL)
    cv.setWindowProperty("Calib", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    preds, pix = [], []
    for (x, y) in pts:
        show_calib_dot(x, y)
        # Wait for SPACE at each point (ESC to cancel)
        while True:
            k = cv.waitKey(10) & 0xFF
            if k == 27:
                cv.destroyWindow("Calib")
                return None
            if k == 32:
                break  # SPACE
        samples = []
        for _ in range(15):
            ok, frame = cam.read()
            if not ok:
                continue
            out = predict_norm_from_frame(frame)
            if out is not None:
                nx, ny, _ = out
                samples.append([nx, ny])
            cv.waitKey(10)
        if samples:
            preds.append(np.mean(np.array(samples), axis=0))
            pix.append([x, y])
    cv.destroyWindow("Calib")
    # after collecting preds (norm) and pix (true)…
    wx, wy = solve_cubic(np.array(preds), np.array(pix))
    pix_hat = np.array([apply_cubic(nx, ny, wx, wy) for (nx, ny) in preds], dtype=np.float32)
    A, t = fit_affine(pix_hat, np.array(pix, dtype=np.float32))
    # Save inside here OR just return and let main save. I recommend: return.
    return wx, wy, A, t


# =========================
# Dwell zoom (optional)
# =========================
ENABLE_DWELL_ZOOM = True
DWELL_RADIUS = 35  # px
DWELL_TIME_MS = 500  # ms
ZOOM_SIZE = 160  # capture box
ZOOM_SCALE = 3


def in_radius(ax, ay, bx, by, r):
    return (ax - bx) ** 2 + (ay - by) ** 2 <= r * r


# =========================
# Blink-to-click (+ freeze)
# =========================
ENABLE_BLINK_CLICK = True
BLINK_EAR_THR = 0.19  # tune 0.17–0.23
BLINK_MIN_MS = 1000  # 1s hold to click
BLINK_HYST = 0.02
FREEZE_ON_BLINK = True


# =========================
# Main loop
# =========================
def main():
    global ENABLE_DWELL_ZOOM

    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("ERROR: Cannot open camera.")
        return

    # Load cached calibration (or run if requested)
    calib = load_calibration()
    if calib is None:
        if RUN_CALIBRATION:
            out = run_quick_calibration(cam)
            if out is not None:
                wx, wy, A, t = out
                save_calibration(wx, wy, A, t)    # ✅ save all four
                print("Calibration complete and saved.")
            else:
                print("Calibration cancelled. Using raw mapping.")
                wx = wy = A = t = None
        else:
            print("Skipping calibration. Using raw mapping.")
            wx = wy = A = t = None
    else:
        wx, wy, A, t = calib
        print("Loaded cached calibration.")


    hist = deque(maxlen=9)  # median window
    alpha = 0.85  # EMA factor
    sx, sy = W / 2, H / 2  # EMA state
    DEAD = 6  # px dead-zone
    out_x, out_y = sx, sy  # cursor position we actually set

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
            if not ok:
                continue

            # eyes + EAR
            eye_img, ear = get_both_eyes_and_ear(frame)
            if eye_img is None:
                cv.imshow("Both Eyes (50x200)", np.zeros((50, 200), dtype=np.uint8))
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # predict normalized gaze
            pil = Image.fromarray(eye_img)
            ten = transform(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                nx, ny = model(ten).squeeze(0).cpu().numpy().tolist()

            # map to pixels (use calibration if available)
            # map to pixels (use calibration if available)
            if (wx is not None) and (wy is not None):
                fx, fy = apply_cubic(nx, ny, wx, wy)
                if (A is not None) and (t is not None):
                    fx, fy = apply_affine(np.array([fx, fy], dtype=np.float32), A, t)
            else:
                fx, fy = nx * W, ny * H


            # bias trim
            fx, fy = apply_bias(fx, fy)

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

                sx = alpha * sx + (1 - alpha) * mx
                sy = alpha * sy + (1 - alpha) * my

                fx = clamp(sx, SAFE_MARGIN, W - 1 - SAFE_MARGIN)
                fy = clamp(sy, SAFE_MARGIN, H - 1 - SAFE_MARGIN)

                if np.hypot(fx - out_x, fy - out_y) >= DEAD:
                    out_x, out_y = fx, fy

                last_stable_x, last_stable_y = out_x, out_y
            else:
                # freeze during blink
                out_x, out_y = last_stable_x, last_stable_y
                if blink_start is not None and not blink_fired:
                    elapsed_ms = (tnow - blink_start) / cv.getTickFrequency() * 1000.0
                    if elapsed_ms >= BLINK_MIN_MS:
                        click_at(last_stable_x, last_stable_y, W, H)
                        blink_fired = True

            # move mouse (pixel-true)
            try:
                set_cursor_pos(out_x, out_y, W, H)
            except Exception:
                # failsafe to center
                set_cursor_pos(W // 2, H // 2, W, H)

            # dwell zoom (optional)
            if ENABLE_DWELL_ZOOM:
                if in_radius(out_x, out_y, dwell_anchor[0], dwell_anchor[1], DWELL_RADIUS):
                    if dwell_start is None:
                        dwell_start = cv.getTickCount()
                    else:
                        elapsed_ms = (cv.getTickCount() - dwell_start) / cv.getTickFrequency() * 1000.0
                        if elapsed_ms >= DWELL_TIME_MS:
                            half = ZOOM_SIZE // 2
                            rx = int(clamp(out_x - half, 0, W - ZOOM_SIZE))
                            ry = int(clamp(out_y - half, 0, H - ZOOM_SIZE))
                            snap = pyautogui.screenshot(region=(rx, ry, ZOOM_SIZE, ZOOM_SIZE))
                            snap = cv.cvtColor(np.array(snap), cv.COLOR_RGB2BGR)
                            snap = cv.resize(
                                snap, (ZOOM_SIZE * ZOOM_SCALE, ZOOM_SIZE * ZOOM_SCALE), interpolation=cv.INTER_NEAREST
                            )
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
            cv.putText(dbg, "BLINK" if is_blinking else "LOOK", (6, 46), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            status = f"EAR={ear:.3f}  norm=({nx:.3f},{ny:.3f})  px=({out_x:.0f},{out_y:.0f})  bias=({BIAS_TRIM[0]:.1f},{BIAS_TRIM[1]:.1f})"
            cv.putText(dbg, status, (6, 20), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv.imshow("Both Eyes (50x200)", dbg)

            # keys
            k = cv.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("r"):
                out = run_quick_calibration(cam)
                if out is not None:
                    wx, wy, A, t = out
                    save_calibration(wx, wy, A, t)
                    print("Re-calibrated and saved.")

            if k == ord("z"):  # toggle dwell zoom
                ENABLE_DWELL_ZOOM = not ENABLE_DWELL_ZOOM
                if not ENABLE_DWELL_ZOOM:
                    try:
                        if cv.getWindowProperty("Zoom", cv.WND_PROP_VISIBLE) > 0:
                            cv.destroyWindow("Zoom")
                    except:
                        pass
            # bias trim hotkeys
            if k == ord("h"):  # nudge left
                BIAS_TRIM[0] -= 5
                print("Bias:", BIAS_TRIM)
            if k == ord("l"):  # nudge right
                BIAS_TRIM[0] += 5
                print("Bias:", BIAS_TRIM)
            if k == ord("k"):  # nudge up
                BIAS_TRIM[1] -= 5
                print("Bias:", BIAS_TRIM)
            if k == ord("j"):  # nudge down
                BIAS_TRIM[1] += 5
                print("Bias:", BIAS_TRIM)
            if k == ord("b"):  # reset bias
                BIAS_TRIM[0] = 0.0
                BIAS_TRIM[1] = 0.0
                print("Bias reset.")

    finally:
        cam.release()
        face_mesh.close()
        try:
            cv.destroyAllWindows()
        except:
            pass


if __name__ == "__main__":             # <-- fix
    main()