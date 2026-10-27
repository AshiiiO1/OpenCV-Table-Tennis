"""
2D Pong — MediaPipe hand tracking. AI opponent OR 2-player.

    pip install mediapipe opencv-python numpy
    wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
    python OpenCV_tabletennis.py

Controls:
    A       toggle AI opponent  <->  2-player   (default: AI on)
    1-5     AI level: Easy / Medium / Hard / Expert / Insane
    T       cycle match length: no limit / 1 / 2 / 3 / 5 min (restarts match)
    P       pause / resume
    D       toggle debug view (camera feed + hand skeletons + stats)
    W / S   keyboard fallback for left  paddle
    I / K   keyboard fallback for right paddle
    R       reset score
    Q       quit

Goal pause: 1.0 s freeze + message after every point.
Ready pause: 0.4 s "READY!" on restart - paddles still movable.
"""

import math
import random
import threading
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

# ----------------------------- config ------------------------------------- #
CAM_INDEX = 0
CAM_W, CAM_H = 320, 240
MODEL_PATH = "hand_landmarker.task"

GW, GH = 1280, 720
TARGET_FPS = 60
DEBUG_DEFAULT = False

Y_LO, Y_HI = 0.15, 0.85
SMOOTH_K = 14.0

PADDLE_W, PADDLE_H = 20, 140
P1_X = 40
P2_X = GW - 40 - PADDLE_W
BALL_R = 12
BALL_SPEED, BALL_SPEED_MAX = 700.0, 1800.0
SPEED_UP, MIN_VX_FRAC = 1.05, 0.45
SPIN = 4.0
KB_STEP = 28

GOAL_FREEZE_DURATION = 1.0
READY_DELAY          = 0.4     # brief pause on restart; paddles still movable

AI_LEVELS = (
    ("Easy",   320.0, 120.0, 0.50),
    ("Medium", 480.0,  70.0, 0.30),
    ("Hard",   650.0,  35.0, 0.10),
    ("Expert", 850.0,  12.0, 0.00),
    ("Insane", 1200.0,  0.0, 0.00),
)
AI_DEFAULT_LEVEL = 1

MATCH_OPTIONS = (0, 60, 120, 180, 300)
MATCH_DEFAULT_IDX = 3
PLAYING, PAUSED, OVER = 0, 1, 2

WIN = "Pong"
PALM_IDS = (0, 5, 9, 13, 17)
HALF_PH = PADDLE_H * 0.5
FRAME_DT = 1.0 / TARGET_FPS


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def map_y(ny):
    t = clamp((ny - Y_LO) / (Y_HI - Y_LO), 0.0, 1.0)
    return HALF_PH + t * (GH - PADDLE_H)


# ----------------------------- camera + landmarker ------------------------ #
cv2.setUseOptimized(True)
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise SystemExit("Camera %d not available" % CAM_INDEX)


def make_landmarker(num_hands):
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


# ----------------------------- tracker thread ----------------------------- #
stop_ev = threading.Event()
debug_ev = threading.Event()
if DEBUG_DEFAULT:
    debug_ev.set()

shared = {"data": ([], None, 0.0)}
want_hands = {"n": 1}


def tracker_loop(landmarker, cur_hands):
    start_t = time.perf_counter()
    last_ts = -1
    fails = 0
    n, t0, tfps = 0, start_t, 0.0
    try:
        while not stop_ev.is_set():
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails > 100:
                    break
                time.sleep(0.01)
                continue
            fails = 0

            wanted = want_hands["n"]
            if wanted != cur_hands:
                new_lm = make_landmarker(wanted)
                landmarker.close()
                landmarker, cur_hands = new_lm, wanted

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int((time.perf_counter() - start_t) * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts
            result = landmarker.detect_for_video(mp_image, ts)

            want_dbg = debug_ev.is_set()
            hands = []
            for lm in result.hand_landmarks:
                nx = 1.0 - sum(lm[i].x for i in PALM_IDS) / len(PALM_IDS)
                ny = sum(lm[i].y for i in PALM_IDS) / len(PALM_IDS)
                pts = [(1.0 - p.x, p.y) for p in lm] if want_dbg else None
                hands.append((nx, ny, pts))
            hands.sort(key=lambda h: h[0])

            n += 1
            now = time.perf_counter()
            if now - t0 >= 0.5:
                tfps = n / (now - t0)
                n, t0 = 0, now

            shared["data"] = (hands, frame if want_dbg else None, tfps)
    finally:
        stop_ev.set()
        landmarker.close()


tracker = threading.Thread(target=tracker_loop, args=(make_landmarker(1), 1),
                           daemon=True)
tracker.start()

# ----------------------------- game state --------------------------------- #
ball_x, ball_y = GW * 0.5, GH * 0.5
ball_vx, ball_vy = 0.0, 0.0
tgt1 = tgt2 = sm1 = sm2 = GH * 0.5
p2_ai_y = GH * 0.5
score1 = score2 = 0
ai_mode = True
ai_level = AI_DEFAULT_LEVEL
ai_err = 0.0
match_idx = MATCH_DEFAULT_IDX
match_len = MATCH_OPTIONS[match_idx]
elapsed = 0.0
state = PLAYING

goal_freeze = 0.0
pending_serve_dir = 1.0
goal_msg = ""
goal_col = (255, 255, 255)

ready_timer = READY_DELAY


def new_ai_error():
    global ai_err
    e = AI_LEVELS[ai_level][2]
    ai_err = random.uniform(-e, e)


def serve(direction):
    global ball_x, ball_y, ball_vx, ball_vy
    ball_x, ball_y = GW * 0.5, GH * 0.5
    ball_vx = BALL_SPEED * direction
    ball_vy = BALL_SPEED * 0.5 * random.choice((-1.0, 1.0))
    new_ai_error()


def reset_match():
    global score1, score2, elapsed, state, match_len
    global goal_freeze, goal_msg, ball_x, ball_y, ball_vx, ball_vy
    global ready_timer
    score1 = score2 = 0
    elapsed = 0.0
    state = PLAYING
    match_len = MATCH_OPTIONS[match_idx]
    goal_freeze = 0.0
    goal_msg = ""
    ready_timer = READY_DELAY
    ball_x, ball_y = GW * 0.5, GH * 0.5
    ball_vx = ball_vy = 0.0


def bounce_off_paddle(paddle_y):
    global ball_vx, ball_vy
    ball_vx = -ball_vx * SPEED_UP
    ball_vy += (ball_y - paddle_y) * SPIN
    sp = math.hypot(ball_vx, ball_vy)
    if sp > BALL_SPEED_MAX:
        k = BALL_SPEED_MAX / sp
        ball_vx *= k
        ball_vy *= k
    if abs(ball_vx) < BALL_SPEED * MIN_VX_FRAC:
        ball_vx = math.copysign(BALL_SPEED * MIN_VX_FRAC, ball_vx)


def draw_hand(canvas, pts, color):
    px = [(int(x * GW), int(y * GH)) for x, y in pts]
    for a, b in HAND_CONNECTIONS:
        cv2.line(canvas, px[a], px[b], color, 2)
    for p in px:
        cv2.circle(canvas, p, 3, color, -1)


def draw_centered(img, text, cy, scale, color, thick):
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thick)
    cv2.putText(img, text, ((GW - w) // 2, cy + h // 2), FONT, scale,
                color, thick, cv2.LINE_AA)


# ----------------------------- window ------------------------------------- #
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, GW, GH)
canvas = np.zeros((GH, GW, 3), np.uint8)

FONT = cv2.FONT_HERSHEY_SIMPLEX
SCORE_SCALE, SCORE_THICK = 2.6, 5
HINT = ("A: mode  1-5: AI level  T: time limit  P: pause  D: debug  "
        "W/S: P1  I/K: P2  R: restart  Q: quit")


def draw_static(img):
    for yy in range(0, GH, 48):
        cv2.rectangle(img, (GW // 2 - 2, yy), (GW // 2 + 2, yy + 24),
                      (70, 70, 70), -1)
    cv2.putText(img, HINT, (16, GH - 16), FONT, 0.6, (130, 130, 130), 1,
                cv2.LINE_AA)


bg = np.zeros((GH, GW, 3), np.uint8)
draw_static(bg)

t_prev = time.perf_counter()
fps_t, fps_n, fps = t_prev, 0, 0.0

try:
    while not stop_ev.is_set():
        now = time.perf_counter()
        dt = now - t_prev
        t_prev = now
        if dt > 0.05:
            dt = 0.05
        elif dt <= 0.0:
            dt = FRAME_DT

        hands, cam, tfps = shared["data"]
        debug = debug_ev.is_set()

        # ---- ready pause / clock ----
        if state == PLAYING:
            if ready_timer > 0:
                ready_timer -= dt
                if ready_timer <= 0:
                    ready_timer = 0
                    serve(1.0)
            else:
                elapsed += dt
                if match_len and elapsed >= match_len:
                    elapsed = float(match_len)
                    if goal_freeze <= 0.0:
                        state = OVER
        gdt = dt if (state == PLAYING and ready_timer <= 0) else 0.0

        # ---- hand assignment ----
        if ai_mode:
            if hands:
                tgt1 = map_y(hands[0][1])
        else:
            if len(hands) >= 2:
                tgt1 = map_y(hands[0][1])
                tgt2 = map_y(hands[-1][1])
            elif len(hands) == 1:
                if hands[0][0] < 0.5:
                    tgt1 = map_y(hands[0][1])
                else:
                    tgt2 = map_y(hands[0][1])

        alpha = 1.0 - math.exp(-SMOOTH_K * dt)
        sm1 += alpha * (tgt1 - sm1)
        sm2 += alpha * (tgt2 - sm2)
        sm1 = clamp(sm1, HALF_PH, GH - HALF_PH)
        sm2 = clamp(sm2, HALF_PH, GH - HALF_PH)
        p1_y = sm1

        # ---- P2 controller ----
        if ai_mode:
            _, ai_speed, _, react = AI_LEVELS[ai_level]
            if ball_vx > 0 and ball_x > GW * react:
                target = ball_y + ai_err
            else:
                target = GH * 0.5
            step = ai_speed * gdt
            p2_ai_y += clamp(target - p2_ai_y, -step, step)
            p2_ai_y = clamp(p2_ai_y, HALF_PH, GH - HALF_PH)
            p2_y = p2_ai_y
        else:
            p2_y = sm2

        # ---- goal freeze ----
        freeze_active = goal_freeze > 0.0
        if freeze_active and gdt > 0.0:
            goal_freeze -= gdt
            if goal_freeze <= 0.0:
                goal_freeze = 0.0
                serve(pending_serve_dir)
                goal_msg = ""
                freeze_active = False

        # ---- physics (skipped during ready / goal freeze) ----
        if gdt > 0.0 and not freeze_active:
            speed = math.hypot(ball_vx, ball_vy)
            n = int(speed * gdt / (BALL_R * 1.2)) + 1
            sdt = gdt / n
            for _ in range(n):
                ball_x += ball_vx * sdt
                ball_y += ball_vy * sdt
                if ball_y < BALL_R:
                    ball_y, ball_vy = BALL_R, abs(ball_vy)
                elif ball_y > GH - BALL_R:
                    ball_y, ball_vy = GH - BALL_R, -abs(ball_vy)

                if (ball_vx < 0.0
                        and ball_x - BALL_R <= P1_X + PADDLE_W
                        and ball_x >= P1_X - BALL_R
                        and abs(ball_y - p1_y) <= HALF_PH + BALL_R):
                    ball_x = P1_X + PADDLE_W + BALL_R
                    bounce_off_paddle(p1_y)
                    new_ai_error()
                elif (ball_vx > 0.0
                        and ball_x + BALL_R >= P2_X
                        and ball_x <= P2_X + PADDLE_W + BALL_R
                        and abs(ball_y - p2_y) <= HALF_PH + BALL_R):
                    ball_x = P2_X - BALL_R
                    bounce_off_paddle(p2_y)

            if ball_x < -BALL_R * 3:
                score2 += 1
                goal_freeze = GOAL_FREEZE_DURATION
                pending_serve_dir = 1.0
                goal_msg = "AI SCORED!" if ai_mode else "P2 SCORED!"
                goal_col = (180, 120, 80) if ai_mode else (0, 200, 255)
                ball_x, ball_y = GW * 0.5, GH * 0.5
                ball_vx = ball_vy = 0.0
            elif ball_x > GW + BALL_R * 3:
                score1 += 1
                goal_freeze = GOAL_FREEZE_DURATION
                pending_serve_dir = -1.0
                goal_msg = "GOAL!" if ai_mode else "P1 SCORED!"
                goal_col = (0, 255, 0)
                ball_x, ball_y = GW * 0.5, GH * 0.5
                ball_vx = ball_vy = 0.0

        # ---- render ----
        if debug and cam is not None:
            cv2.resize(cv2.flip(cam, 1), (GW, GH), dst=canvas)
            np.right_shift(canvas, 1, out=canvas)
            draw_static(canvas)
        else:
            np.copyto(canvas, bg)

        cv2.rectangle(canvas, (P1_X, int(p1_y - HALF_PH)),
                      (P1_X + PADDLE_W, int(p1_y + HALF_PH)), (0, 255, 0), -1)
        cv2.rectangle(canvas, (P2_X, int(p2_y - HALF_PH)),
                      (P2_X + PADDLE_W, int(p2_y + HALF_PH)),
                      (180, 120, 80) if ai_mode else (0, 200, 255), -1)
        cv2.circle(canvas, (int(ball_x), int(ball_y)), BALL_R,
                   (0, 0, 255), -1, cv2.LINE_AA)

        s1, s2 = str(score1), str(score2)
        w1 = cv2.getTextSize(s1, FONT, SCORE_SCALE, SCORE_THICK)[0][0]
        cv2.putText(canvas, s1, (GW // 2 - 60 - w1, 110), FONT, SCORE_SCALE,
                    (255, 255, 255), SCORE_THICK, cv2.LINE_AA)
        cv2.putText(canvas, s2, (GW // 2 + 60, 110), FONT, SCORE_SCALE,
                    (255, 255, 255), SCORE_THICK, cv2.LINE_AA)
        if match_len:
            remaining = max(0.0, match_len - elapsed)
            secs = int(math.ceil(remaining))
            clock_col = (0, 0, 255) if remaining <= 10.0 else (255, 255, 255)
        else:
            secs = int(elapsed)
            clock_col = (255, 255, 255)
        draw_centered(canvas, "%d:%02d" % (secs // 60, secs % 60), 44, 1.3,
                      clock_col, 3)

        info = []
        if ai_mode:
            info.append("AI: %s (%d)" % (AI_LEVELS[ai_level][0], ai_level + 1))
        info.append("Match: %s" % ("no limit" if not match_len
                                   else "%d min" % (match_len // 60)))
        for i, line in enumerate(info):
            w = cv2.getTextSize(line, FONT, 0.7, 2)[0][0]
            cv2.putText(canvas, line, (GW - 16 - w, 32 + 30 * i), FONT, 0.7,
                        (200, 200, 200), 2, cv2.LINE_AA)

        # brief ready overlay
        if ready_timer > 0:
            draw_centered(canvas, "READY!", GH // 2 - 60, 2.0,
                          (0, 200, 255), 4)
            draw_centered(canvas, "Position your paddle", GH // 2 + 20,
                          0.7, (150, 150, 150), 2)

        # goal-freeze overlay
        if goal_freeze > 0.0 and goal_msg:
            draw_centered(canvas, goal_msg, GH // 2 - 60, 2.0, goal_col, 4)
            draw_centered(canvas, "%d  -  %d" % (score1, score2),
                          GH // 2 + 10, 1.2, (220, 220, 220), 3)

        if state == PAUSED:
            draw_centered(canvas, "PAUSED", GH // 2, 2.2, (255, 255, 255), 4)
        elif state == OVER:
            np.right_shift(canvas, 1, out=canvas)
            if score1 == score2:
                verdict = "DRAW"
            elif score1 > score2:
                verdict = "YOU WIN!" if ai_mode else "PLAYER 1 WINS!"
            else:
                verdict = "AI WINS" if ai_mode else "PLAYER 2 WINS!"
            draw_centered(canvas, "TIME UP", GH // 2 - 70, 2.4, (255, 255, 255), 5)
            draw_centered(canvas, verdict, GH // 2 + 10, 1.6, (0, 255, 255), 4)
            draw_centered(canvas, "%d - %d" % (score1, score2), GH // 2 + 80,
                          1.2, (200, 200, 200), 3)
            draw_centered(canvas, "R: new match", GH // 2 + 140, 0.9,
                          (180, 180, 180), 2)

        if ai_mode:
            ok_hands = len(hands) >= 1
            scol = (0, 255, 0) if ok_hands else (0, 0, 255)
            status = "AI  |  " + ("HAND" if ok_hands else "NO HAND")
        elif len(hands) >= 2:
            scol, status = (0, 255, 0), "2P  |  BOTH HANDS"
        elif len(hands) == 1:
            scol = (0, 255, 255)
            status = "2P  |  ONE HAND -> " + ("P1" if hands[0][0] < 0.5 else "P2")
        else:
            scol, status = (0, 0, 255), "2P  |  NO HANDS"
        cv2.circle(canvas, (24, 24), 8, scol, -1, cv2.LINE_AA)

        if debug:
            for i, (nx, ny, pts) in enumerate(hands):
                color = (0, 255, 0) if i == 0 else (0, 200, 255)
                if pts:
                    draw_hand(canvas, pts, color)
                cv2.circle(canvas, (int(nx * GW), int(ny * GH)), 10,
                           (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, status, (44, 32), FONT, 0.7, scol, 2, cv2.LINE_AA)
            cv2.putText(canvas, "game %.0f fps | tracker %.0f fps" % (fps, tfps),
                        (16, 66), FONT, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        fps_n += 1
        if now - fps_t >= 0.5:
            fps = fps_n / (now - fps_t)
            fps_n, fps_t = 0, now

        cv2.imshow(WIN, canvas)

        wait = max(1, int((FRAME_DT - (time.perf_counter() - now)) * 1000))
        key = cv2.waitKey(wait) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key == ord('r'):
            reset_match()
        elif key == ord('p'):
            if state == PLAYING:
                state = PAUSED
            elif state == PAUSED:
                state = PLAYING
        elif key == ord('t'):
            match_idx = (match_idx + 1) % len(MATCH_OPTIONS)
            reset_match()
        elif ord('1') <= key <= ord('5'):
            ai_level = key - ord('1')
            new_ai_error()
        elif key == ord('a'):
            ai_mode = not ai_mode
            want_hands["n"] = 1 if ai_mode else 2
        elif key == ord('d'):
            if debug_ev.is_set():
                debug_ev.clear()
            else:
                debug_ev.set()
        elif key == ord('w'):
            tgt1 = clamp(tgt1 - KB_STEP, HALF_PH, GH - HALF_PH)
        elif key == ord('s'):
            tgt1 = clamp(tgt1 + KB_STEP, HALF_PH, GH - HALF_PH)
        elif key == ord('i'):
            tgt2 = clamp(tgt2 - KB_STEP, HALF_PH, GH - HALF_PH)
        elif key == ord('k'):
            tgt2 = clamp(tgt2 + KB_STEP, HALF_PH, GH - HALF_PH)

finally:
    stop_ev.set()
    tracker.join(timeout=2.0)
    cap.release()
    cv2.destroyAllWindows()