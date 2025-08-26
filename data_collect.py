# data_collect.py
# Collect TWO-EYE crops (50x200) named as "<x>.<y>.<idx>.jpg" for training.
import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import pyautogui
# --- Windows DPI awareness (ensures real pixel coords) ---
import sys


if sys.platform.startswith("win"):
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass



# ===== settings =====
CAPTURES_PER_POINT = 30          # fewer per point; collect more points instead
FOLDER = "eye_images"
CAM_INDEX = 1                    # try 0 if 1 doesn't show your camera
DOT_RADIUS = 14
MARGIN = 40
GRID_N = 7                       # 7x7 grid gives better coverage

# screen
SCREEN_W, SCREEN_H = pyautogui.size()

def clamp(v, lo, hi): return max(lo, min(hi, v))

def draw_target(x, y, msg="Look at the red dot. SPACE=start, Q=skip, ESC=quit"):
    img = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    cv.circle(img, (int(x), int(y)), DOT_RADIUS, (0, 0, 255), -1)
    cv.putText(img, msg, (40, SCREEN_H-60), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv.LINE_AA)
    return img

def get_both_eyes(frame, face_mesh):
    """Return concatenated both-eye grayscale image of shape (50, 200), or None."""
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None

    lm = res.multi_face_landmarks[0].landmark
    h, w = frame.shape[:2]

    left_ids  = [33,133,159,145,158,144,153,154,155]
    right_ids = [362,263,386,374,385,373,380,381,382]

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

    lx1,ly1,lx2,ly2 = eye_bbox(left_ids)
    rx1,ry1,rx2,ry2 = eye_bbox(right_ids)
    if lx2<=lx1 or ly2<=ly1 or rx2<=rx1 or ry2<=ry1:
        return None

    L = cv.cvtColor(frame[ly1:ly2, lx1:lx2], cv.COLOR_BGR2GRAY)
    R = cv.cvtColor(frame[ry1:ry2, rx1:rx2], cv.COLOR_BGR2GRAY)
    L = cv.resize(L, (100,50)); R = cv.resize(R, (100,50))
    both = np.hstack([L, R])  # (50, 200)
    return both

def main():
    os.makedirs(FOLDER, exist_ok=True)

    xs = np.linspace(MARGIN, SCREEN_W-MARGIN, GRID_N, dtype=int).tolist()
    ys = np.linspace(MARGIN, SCREEN_H-MARGIN, GRID_N, dtype=int).tolist()

    win = "Calibration"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.setWindowProperty(win, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    cam = cv.VideoCapture(CAM_INDEX)
    if not cam.isOpened():
        cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("Camera failed to open. Try CAM_INDEX=0/1.")
        return

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
    )

    try:
        for x in xs:
            for y in ys:
                x = clamp(x, MARGIN, SCREEN_W-MARGIN)
                y = clamp(y, MARGIN, SCREEN_H-MARGIN)

                # wait-for-start (SPACE) or skip (Q)
                while True:
                    cv.imshow(win, draw_target(x, y))
                    k = cv.waitKey(10) & 0xFF
                    if k == 27:  # ESC
                        return
                    if k == ord('q'):
                        break
                    if k == 32:  # SPACE -> start
                        # warm-up a few frames
                        for _ in range(5): cam.read()
                        saved = 0
                        while saved < CAPTURES_PER_POINT:
                            ok, frame = cam.read()
                            if not ok: continue
                            eye = get_both_eyes(frame, face_mesh)
                            cv.imshow(win, draw_target(x, y, f"Capturing {saved+1}/{CAPTURES_PER_POINT} (ESC quit)"))
                            if eye is not None:
                                cv.imshow("eye_box", eye)
                                cv.imwrite(os.path.join(FOLDER, f"{int(x)}.{int(y)}.{saved}.jpg"), eye)
                                saved += 1
                            k2 = cv.waitKey(1) & 0xFF
                            if k2 == 27: return
                            if k2 == ord('q'): break
                        # small confirmation flash
                        cv.imshow(win, draw_target(x, y, "DONE!"))
                        cv.waitKey(300)
                        break  # next point
    finally:
        cam.release()
        face_mesh.close()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
