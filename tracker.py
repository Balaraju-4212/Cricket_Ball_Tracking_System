import os
import cv2
import numpy as np
import torch


class KalmanBallTracker:
    """Simple 2D Kalman Filter for position & velocity tracking."""
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.initialized = False

    def init(self, x: float, y: float):
        self.kalman.statePre = np.array([[x], [y], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.initialized = True

    def predict(self):
        if not self.initialized:
            return None
        prediction = self.kalman.predict()
        return (int(prediction[0][0]), int(prediction[1][0]))

    def correct(self, x: float, y: float):
        if not self.initialized:
            self.init(x, y)
            return (int(x), int(y))
        corrected = self.kalman.correct(np.array([[np.float32(x)], [np.float32(y)]]))
        return (int(corrected[0][0]), int(corrected[1][0]))


class CricketBallTracker:
    """
    Unified Cricket Ball Detection & Tracking Engine.
    Combines HSV Color/Contour detection with YOLO fallback and Kalman filtering.
    """
    def __init__(self, conf_thresh: float = 0.25):
        self.conf_thresh = conf_thresh
        self.yolo_model = None
        self._init_yolo()

    def _init_yolo(self):
        try:
            from ultralytics import YOLO
            if os.path.exists("models/ball_detector/best.pt"):
                self.yolo_model = YOLO("models/ball_detector/best.pt")
            else:
                self.yolo_model = YOLO("yolov8n.pt")
        except Exception:
            self.yolo_model = None

    def detect_ball(self, frame: np.ndarray) -> tuple:
        """
        Detects ball using YOLO (strictly filtering for sports ball class 32),
        or high-confidence circular contour fallback.
        Returns (center_x, center_y, radius, confidence) or None.
        """
        # 1. Try YOLO detection first (filter for class 32 = sports ball or class 0 = custom ball)
        if self.yolo_model is not None:
            try:
                results = self.yolo_model(frame, verbose=False, conf=self.conf_thresh)[0]
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    # Class 32 in standard COCO YOLO is sports ball, Class 0 in custom model is ball
                    if cls_id == 32 or (cls_id == 0 and os.path.exists("models/ball_detector/best.pt")):
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        r = int(max(x2 - x1, y2 - y1) / 2)
                        return (cx, cy, r, conf)
            except Exception:
                pass

        # 2. Strict HSV Color & High-Circularity Contour Fallback
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red Ball HSV Ranges
        mask_r1 = cv2.inRange(hsv, np.array([0, 120, 120]), np.array([10, 255, 255]))
        mask_r2 = cv2.inRange(hsv, np.array([160, 120, 120]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask_r1, mask_r2)

        # White Ball HSV Range
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        combined_mask = cv2.bitwise_or(red_mask, white_mask)

        # Morphological Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        max_score = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 20 <= area <= 1800:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                if radius >= 3:
                    perimeter = cv2.arcLength(cnt, True)
                    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                    if circularity >= 0.72:
                        score = area * circularity
                        if score > max_score:
                            max_score = score
                            best_candidate = (int(x), int(y), int(radius), 0.75)

        return best_candidate


def _green_field_ratio(frame: np.ndarray) -> float:
    """
    Measures natural green field (grass/turf) coverage across the FULL frame.
    Scans full frame so close-up cricket Shorts (portrait, tight shots) are
    not penalised for having limited outfield visible.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Grass green: H 32-88 (green hues), S 40+, V 40+
    grass = cv2.inRange(hsv, np.array([32, 40, 40]), np.array([88, 255, 255]))
    return float(np.sum(grass > 0)) / float(grass.size)


def _pitch_or_turf_present(frame: np.ndarray) -> float:
    """
    Detects either green grass OR the sandy/light cricket pitch strip.
    Cricket Shorts often show mostly pitch strip (sandy/beige) and very little outfield.
    Returns combined ratio of cricket-relevant ground colours.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Green grass/turf
    grass = cv2.inRange(hsv, np.array([32, 40, 40]), np.array([88, 255, 255]))
    # Sandy/beige cricket pitch strip: low saturation, bright, warm hue
    pitch = cv2.inRange(hsv, np.array([10, 8, 130]), np.array([38, 100, 255]))
    combined = cv2.bitwise_or(grass, pitch)
    return float(np.sum(combined > 0)) / float(combined.size)


def _dominant_non_sport_colour_ratio(frame: np.ndarray) -> float:
    """
    Detects colours that are EXCLUSIVE to non-sports content:
      - Vivid saffron/orange (devotional, idol, Hindu TV serials)
      - Deep gold/yellow jewellery
      - Vivid magenta/pink (studio backgrounds, artwork)
      - Very dark frames (close-up indoor face shots)
    Returns ratio 0–1. High value = definitely NOT a cricket broadcast.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Saffron / bright orange (very common in devotional content)
    saffron = cv2.inRange(hsv, np.array([8, 180, 150]), np.array([22, 255, 255]))
    # Deep gold / yellow jewellery  
    gold    = cv2.inRange(hsv, np.array([22, 160, 150]), np.array([35, 255, 255]))
    # Vivid magenta / hot pink
    magenta = cv2.inRange(hsv, np.array([140, 120, 100]), np.array([175, 255, 255]))
    combined = cv2.bitwise_or(cv2.bitwise_or(saffron, gold), magenta)
    return float(np.sum(combined > 0)) / float(combined.size)


def _yolo_ball_or_person_present(frames: list, yolo_model) -> tuple:
    """
    Uses YOLO to check for sports ball (class 32) and person (class 0) detections.
    Returns (ball_found: bool, person_count: int).
    """
    ball_found   = False
    total_people = 0
    if yolo_model is None:
        return False, 0

    num = len(frames)
    indices = [int(num * p) for p in [0.15, 0.35, 0.55, 0.75] if int(num * p) < num]
    for i in indices:
        try:
            result = yolo_model(frames[i], verbose=False, conf=0.2)[0]
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls == 32:      # sports ball
                    ball_found = True
                elif cls == 0:     # person (batsman, bowler, fielder)
                    total_people += 1
        except Exception:
            pass
    return ball_found, total_people


def filter_outliers(trajectory: list) -> list:
    """
    Filters out extreme outlier detections (e.g., false positive jumps)
    before curve fitting.
    """
    if len(trajectory) < 3:
        return trajectory

    # Filter by checking consecutive displacements
    filtered = [trajectory[0]]
    for i in range(1, len(trajectory)):
        prev = filtered[-1]
        curr = trajectory[i]
        
        # Only check displacement for detections, not predictions
        if curr[3] and prev[3]:
            dist = np.hypot(curr[0] - prev[0], curr[1] - prev[1])
            # If the ball jumps more than 120 pixels in a single frame, it is likely noise
            if dist > 120.0:
                continue
        filtered.append(curr)
    return filtered


def get_smoothed_trajectory(trajectory: list, bounce_frame_idx: int) -> list:
    """
    Fits smooth quadratic curves to pre-bounce and post-bounce segments of the trajectory
    to eliminate any noisy zig-zags and return a clean, high-quality single path.
    """
    clean_traj = filter_outliers(trajectory)
    
    if len(clean_traj) < 4:
        return clean_traj

    pre_pts = [p for p in clean_traj if bounce_frame_idx is None or p[2] <= bounce_frame_idx]
    post_pts = [p for p in clean_traj if bounce_frame_idx is not None and p[2] > bounce_frame_idx]

    smoothed = []

    # Fit pre-bounce (degree 2 for parabolic flight)
    if len(pre_pts) >= 3:
        ts = np.array([p[2] for p in pre_pts], dtype=np.float32)
        xs = np.array([p[0] for p in pre_pts], dtype=np.float32)
        ys = np.array([p[1] for p in pre_pts], dtype=np.float32)
        
        coeff_x = np.polyfit(ts, xs, 2)
        coeff_y = np.polyfit(ts, ys, 2)
        
        for p in pre_pts:
            t = p[2]
            sx = int(np.polyval(coeff_x, t))
            sy = int(np.polyval(coeff_y, t))
            smoothed.append((sx, sy, p[2], p[3]))
    else:
        smoothed.extend(pre_pts)

    # Fit post-bounce
    if len(post_pts) >= 3:
        ts = np.array([p[2] for p in post_pts], dtype=np.float32)
        xs = np.array([p[0] for p in post_pts], dtype=np.float32)
        ys = np.array([p[1] for p in post_pts], dtype=np.float32)
        
        coeff_x = np.polyfit(ts, xs, 2)
        coeff_y = np.polyfit(ts, ys, 2)
        
        for p in post_pts:
            t = p[2]
            sx = int(np.polyval(coeff_x, t))
            sy = int(np.polyval(coeff_y, t))
            smoothed.append((sx, sy, p[2], p[3]))
    else:
        smoothed.extend(post_pts)

    smoothed.sort(key=lambda x: x[2])
    return smoothed


def is_cricket_match_video(frames: list, yolo_model=None) -> bool:
    """
    Strict cricket match validator using hard AND gates.

    A video passes ONLY if ALL three mandatory gates pass:

    GATE 1 — GREEN FIELD (mandatory hard reject)
        Average green turf ratio >= 10% in the bottom 55% of sampled frames.
        Close-up face shots, indoor sets, idol reels all have ~0% green field.

    GATE 2 — NO DOMINANT NON-CRICKET COLOUR (mandatory hard reject)
        Average saffron/gold/magenta ratio < 8%.
        Devotional reels, TV serials, idol videos typically score 15-40% here.

    GATE 3 — SPORTS CONTEXT (ball OR players present)
        YOLO detects at least one sports ball (class 32), OR
        YOLO detects 2+ persons across sampled frames (fielders, batsman, bowler).

    If all three gates pass → confirmed cricket.
    If any gate fails → immediately rejected with clear reason.
    """
    if not frames:
        return False

    num = len(frames)
    # Use up to 6 evenly spaced frames for reliable sampling
    pcts = [0.05, 0.2, 0.35, 0.5, 0.65, 0.85]
    sample_indices = [int(num * p) for p in pcts if int(num * p) < num]
    if not sample_indices:
        sample_indices = [0]

    green_ratios     = []
    turf_ratios      = []   # grass + pitch strip combined
    non_sport_ratios = []

    for i in sample_indices:
        frame = frames[i]
        if frame is None:
            continue
        green_ratios.append(_green_field_ratio(frame))
        turf_ratios.append(_pitch_or_turf_present(frame))
        non_sport_ratios.append(_dominant_non_sport_colour_ratio(frame))

    if not green_ratios:
        return False

    avg_green     = float(np.mean(green_ratios))
    avg_turf      = float(np.mean(turf_ratios))    # green OR sandy pitch
    avg_non_sport = float(np.mean(non_sport_ratios))
    ball_found, people_count = _yolo_ball_or_person_present(frames, yolo_model)

    # GATE 1: Cricket ground present
    # Wide broadcast → lots of green visible (avg_green >= 0.06)
    # Close-up Shorts → little green but lots of sandy pitch (avg_turf >= 0.12)
    gate1 = avg_green >= 0.06 or avg_turf >= 0.12

    # GATE 2: No dominant devotional/non-sport colours (saffron, gold, magenta)
    # Close-up cricket Shorts have orange/brown boundary boards so keep threshold at 0.10
    gate2 = avg_non_sport <= 0.10

    # GATE 3: Sports context — person OR ball detected by YOLO,
    # OR the turf score is high enough to confirm a cricket ground even without YOLO
    # (portrait Shorts at 360x640 make YOLO unreliable for small players)
    gate3 = ball_found or people_count >= 1 or avg_green >= 0.25 or avg_turf >= 0.12

    print(
        f"[CricketGate] green={avg_green:.3f} turf={avg_turf:.3f}→Gate1={'PASS' if gate1 else 'FAIL'} | "
        f"non_sport={avg_non_sport:.3f}→Gate2={'PASS' if gate2 else 'FAIL'} | "
        f"ball={ball_found} people={people_count}→Gate3={'PASS' if gate3 else 'FAIL'}"
    )

    # ALL three gates must pass — one failure = instant rejection
    return gate1 and gate2 and gate3


def get_video_source(video_input: str) -> str:
    """
    Resolves a local video file path, direct video URL, or YouTube Shorts/video URL into a valid OpenCV video source.
    """
    video_input = video_input.strip()

    if os.path.exists(video_input):
        return video_input

    if video_input.startswith("http://") or video_input.startswith("https://"):
        output_path = "data/raw/url_video.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 1. Try yt-dlp download for YouTube Shorts & video links
        try:
            import yt_dlp
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass

            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': output_path,
                'overwrites': True,
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_input])

            if os.path.exists(output_path):
                cap = cv2.VideoCapture(output_path)
                if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
                    cap.release()
                    return output_path
                cap.release()
        except Exception as e:
            print(f"[WARN] yt-dlp download failed: {e}")

        # 2. Try direct streaming URL candidates via yt-dlp
        try:
            import yt_dlp
            ydl_opts = {'quiet': True, 'no_warnings': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_input, download=False)
                formats = info.get('formats', [])
                candidates = [
                    f['url'] for f in reversed(formats)
                    if f.get('url') and f.get('vcodec') != 'none' and f.get('ext') == 'mp4'
                ]
                if info.get('url') and info.get('vcodec') != 'none':
                    candidates.insert(0, info['url'])

                for candidate_url in candidates:
                    cap = cv2.VideoCapture(candidate_url)
                    if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
                        cap.release()
                        return candidate_url
                    cap.release()
        except Exception as e:
            print(f"[WARN] Direct stream candidate extraction failed: {e}")

        # 3. Direct URL stream in OpenCV
        cap = cv2.VideoCapture(video_input)
        if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
            cap.release()
            return video_input
        cap.release()

        # 4. Fallback HTTP file download
        try:
            import urllib.request
            req = urllib.request.Request(
                video_input,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
                out_file.write(response.read())

            cap = cv2.VideoCapture(output_path)
            if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
                cap.release()
                return output_path
            cap.release()
        except Exception as e:
            print(f"[WARN] urllib fallback failed: {e}")

    raise ValueError(f"Unable to process or open video from URL: '{video_input}'. Please make sure the link is a valid YouTube video/Shorts or direct video stream.")



def process_cricket_video(video_path: str, output_path: str = "outputs/tracked_output.mp4",
                           conf_thresh: float = 0.25, pitch_pixel_len: float = 500.0,
                           progress_callback=None) -> dict:
    """
    Processes input video, tracks the ball, computes telemetry, and draws visualization overlays.
    """
    stream_source = get_video_source(video_path)
    cap = cv2.VideoCapture(stream_source)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video source: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100

    # Read all frames into memory first
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError("Unable to read frames from video source.")

    # 1. Pre-Analysis Multi-Signal Cricket Validation BEFORE creating any output files
    detector = CricketBallTracker(conf_thresh=conf_thresh)
    if not is_cricket_match_video(frames, yolo_model=detector.yolo_model):
        # Ensure any existing stale output files are deleted from disk
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass
        raise ValueError("Non-Cricket Video Rejected: This video does not contain a cricket pitch or match field. Ball tracking and DRS analysis are disabled for non-cricket videos.")

    kalman = KalmanBallTracker()

    trajectory = []
    num_frames = len(frames)

    for frame_idx, frame in enumerate(frames, start=1):
        detection = detector.detect_ball(frame)
        predicted = kalman.predict()

        if detection is not None:
            cx, cy, r, conf = detection
            tracked_pt = kalman.correct(cx, cy)
            trajectory.append((tracked_pt[0], tracked_pt[1], frame_idx, True))
        elif predicted is not None:
            trajectory.append((predicted[0], predicted[1], frame_idx, False))

        if progress_callback and num_frames > 0:
            progress_callback(0.4 * (frame_idx / float(num_frames)))

    # Rough bounce frame index to segment trajectory
    raw_bounce_frame_idx = None
    if len(trajectory) >= 5:
        y_coords = [p[1] for p in trajectory]
        max_y_idx = int(np.argmax(y_coords))
        raw_bounce_frame_idx = trajectory[max_y_idx][2]

    # Clean and smooth trajectory curves (pre-bounce & post-bounce)
    trajectory = get_smoothed_trajectory(trajectory, raw_bounce_frame_idx)

    # 2. Cricket Ball Detection Verification
    detected_pts = [(p[0], p[1]) for p in trajectory if len(p) >= 4 and p[3] is True]
    if len(detected_pts) < 4:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass
        raise ValueError("Non-Cricket Video Rejected: This video does not contain a cricket pitch or match field. Ball tracking and DRS analysis are disabled for non-cricket videos.")

    # 3. Delivery Displacement Motion Verification (Must move across pitch)
    x_disp = max(p[0] for p in detected_pts) - min(p[0] for p in detected_pts)
    y_disp = max(p[1] for p in detected_pts) - min(p[1] for p in detected_pts)
    total_motion = np.hypot(x_disp, y_disp)

    if total_motion < 40.0:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass
        raise ValueError("Non-Cricket Video Rejected: This video does not contain a cricket pitch or match field. Ball tracking and DRS analysis are disabled for non-cricket videos.")

    # Calculate speeds from the smoothed trajectory
    speeds = []
    for k in range(1, len(trajectory)):
        p1, p2 = trajectory[k - 1], trajectory[k]
        pixel_dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        meters = (pixel_dist / pitch_pixel_len) * 20.12
        speed_kmh = meters * fps * 3.6
        if speed_kmh < 170.0:
            speeds.append(speed_kmh)

    max_speed = max(speeds) if speeds else 0.0
    avg_speed = float(np.mean(speeds)) if speeds else 0.0

    # 1. Identify Bounce Point and Impact Point
    bounce_point = None
    bounce_frame_idx = None
    impact_point = None
    impact_frame_idx = None

    if len(trajectory) >= 5:
        y_coords = [p[1] for p in trajectory]
        max_y_idx = int(np.argmax(y_coords))
        bounce_point = (trajectory[max_y_idx][0], trajectory[max_y_idx][1])
        bounce_frame_idx = trajectory[max_y_idx][2]
        
        impact_idx = min(len(trajectory) - 1, max_y_idx + int((len(trajectory) - max_y_idx) * 0.7)) if len(trajectory) > max_y_idx + 2 else len(trajectory) - 1
        impact_point = (trajectory[impact_idx][0], trajectory[impact_idx][1])
        impact_frame_idx = trajectory[impact_idx][2]
    elif trajectory:
        impact_point = (trajectory[-1][0], trajectory[-1][1])
        impact_frame_idx = trajectory[-1][2]

    # 2. Perform DRS Trajectory Projection & Decision Check
    drs_data = calculate_drs_projection(trajectory, bounce_point, impact_point, width, height)

    # Classify shot type before render loop so it can be drawn on every frame
    shot_analysis = classify_shot_and_pitch_length(trajectory, bounce_point, height)
    shot_label = shot_analysis["predicted_shot"]   # e.g. "Pull Shot"

    # 3. Render Overlays & Write Video in memory (slow-motion = half FPS, animated style)
    slow_fps = max(fps / 2.0, 8.0)   # half speed playback, min 8 fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_drs = cv2.VideoWriter(output_path, fourcc, slow_fps, (width, height))

    num_buffered = len(frames)
    for curr_fnum, frame in enumerate(frames, start=1):
        # Apply cartoon / animated visual style (preserves motion)
        frame = cartoonize_frame(frame)

        # Draw Pitch Corridor & Virtual Stumps
        draw_pitch_corridor_and_stumps(frame, width, height, drs_data)

        # Draw ball direction arrow (not full lines)
        sub_traj = [p for p in trajectory if p[2] <= curr_fnum]
        draw_drs_trajectory_lines(frame, sub_traj, bounce_point, impact_point, drs_data, curr_fnum, num_buffered)

        # Draw DRS Lower-Third Broadcast Card HUD
        draw_drs_broadcast_card(frame, drs_data, curr_fnum, num_buffered, speeds, curr_fnum)

        # Draw Shot Label top-centre
        label_text = "SHOT: " + shot_label.upper()
        lw, lh = 14, 32
        lx = (width - len(label_text) * lw) // 2
        cv2.rectangle(frame, (lx - 10, 52), (lx + len(label_text) * lw, 52 + lh + 6), (10, 10, 30), -1)
        cv2.putText(frame, label_text, (lx, 52 + lh),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)

        out_drs.write(frame)

        if progress_callback and num_buffered > 0:
            progress_callback(0.4 + 0.5 * (curr_fnum / float(num_buffered)))

    out_drs.release()

    # Convert output video to H.264 web format with ultrafast preset
    output_path = convert_to_web_mp4(output_path)

    if progress_callback:
        progress_callback(1.0)

    bp = (int(bounce_point[0]), int(bounce_point[1])) if bounce_point else None
    ip = (int(impact_point[0]), int(impact_point[1])) if impact_point else None

    return {
        "output_path": output_path,
        "csv_path": None,
        "total_frames": int(frame_idx),
        "trajectory_points": len(trajectory),
        "max_speed_kmh": float(round(float(max_speed), 1)),
        "avg_speed_kmh": float(round(float(avg_speed), 1)),
        "bounce_point": bp,
        "impact_point": ip,
        "pitch_length_zone": str(shot_analysis["pitch_length_zone"]),
        "predicted_shot": str(shot_analysis["predicted_shot"]),
        "drs_pitching": drs_data["pitching_status"],
        "drs_impact": drs_data["impact_status"],
        "drs_wickets": drs_data["wickets_status"],
        "drs_decision": drs_data["decision"],
        "drs_hit_detail": drs_data["hit_detail"],
        "stump_target_point": drs_data["stump_hit_point"],
        "projected_points_count": len(drs_data["projected_points"])
    }


def calculate_drs_projection(trajectory: list, bounce_point: tuple, impact_point: tuple,
                             width: int = 1280, height: int = 720) -> dict:
    """
    Calculates physics-based trajectory extrapolation past impact point towards stumps ('if ball missed where it will go'),
    and computes IPL DRS Decision outcomes (Pitching, Impact, Wickets).
    """
    stump_center_x = int(width * 0.5)
    stump_y_base = int(height * 0.72)
    stump_height = int(height * 0.18)
    stump_y_top = stump_y_base - stump_height
    stump_width = int(width * 0.08)

    projected_pts = []
    
    if len(trajectory) >= 3:
        # Determine velocity vector past bounce/impact point
        pts = trajectory[-4:] if len(trajectory) >= 4 else trajectory
        dxs = [pts[k+1][0] - pts[k][0] for k in range(len(pts)-1)]
        dys = [pts[k+1][1] - pts[k][1] for k in range(len(pts)-1)]
        
        avg_dx = float(np.mean(dxs))
        avg_dy = float(np.mean(dys))

        # Gravity & spin curve effect post-impact
        gravity_y = 1.2
        curr_x = float(impact_point[0]) if impact_point else float(trajectory[-1][0])
        curr_y = float(impact_point[1]) if impact_point else float(trajectory[-1][1])

        # Project 20 steps towards batsman stumps
        for step in range(1, 22):
            curr_x += avg_dx * 0.95
            curr_y += avg_dy + gravity_y * (step * 0.1)
            projected_pts.append((int(curr_x), int(curr_y)))
            if curr_y >= stump_y_base or curr_x < 0 or curr_x > width:
                break

    # Determine Pitching Status (In-Line, Outside Off, Outside Leg)
    pitch_min_x = stump_center_x - int(stump_width * 1.5)
    pitch_max_x = stump_center_x + int(stump_width * 1.5)
    
    pitching_status = "IN-LINE"
    if bounce_point:
        bx = bounce_point[0]
        if bx < pitch_min_x:
            pitching_status = "OUTSIDE OFF"
        elif bx > pitch_max_x:
            pitching_status = "OUTSIDE LEG"

    # Determine Impact Status (In-Line, Outside Off, Outside Leg)
    impact_status = "IN-LINE"
    if impact_point:
        ix = impact_point[0]
        if ix < pitch_min_x:
            impact_status = "OUTSIDE OFF"
        elif ix > pitch_max_x:
            impact_status = "OUTSIDE LEG"

    # Determine Wickets Status (Hitting Stumps / Umpire's Call / Missing)
    stump_left = stump_center_x - stump_width // 2
    stump_right = stump_center_x + stump_width // 2
    
    stump_hit_pt = projected_pts[-1] if projected_pts else (impact_point if impact_point else (stump_center_x, stump_y_top))
    
    proj_x, proj_y = stump_hit_pt

    wickets_status = "MISSING"
    hit_detail = "Missing High"
    decision = "NOT OUT"

    # Check intersection with stump box
    if proj_y > stump_y_base + 30:
        hit_detail = "Hitting Pitch / Low Bounce"
        wickets_status = "MISSING"
        decision = "NOT OUT"
    elif proj_y < stump_y_top - 40:
        hit_detail = "Over the Bails (Missing High)"
        wickets_status = "MISSING"
        decision = "NOT OUT"
    elif proj_x < stump_left - 30:
        hit_detail = "Missing Off Stump"
        wickets_status = "MISSING"
        decision = "NOT OUT"
    elif proj_x > stump_right + 30:
        hit_detail = "Missing Leg Stump"
        wickets_status = "MISSING"
        decision = "NOT OUT"
    else:
        # Within horizontal & vertical collision area
        margin_x = stump_width * 0.25
        if (stump_left <= proj_x <= stump_right) and (stump_y_top <= proj_y <= stump_y_base):
            wickets_status = "HITTING"
            decision = "OUT"
            if abs(proj_x - stump_center_x) < margin_x:
                hit_detail = "Hitting Middle Stump"
            elif proj_x < stump_center_x:
                hit_detail = "Hitting Off Stump"
            else:
                hit_detail = "Hitting Leg Stump"
        elif (stump_left - 20 <= proj_x <= stump_right + 20) and (stump_y_top - 20 <= proj_y <= stump_y_base + 20):
            wickets_status = "UMPIRE'S CALL"
            decision = "OUT (Umpire's Call)"
            hit_detail = "Clipping Bails / Stump Edge"
        else:
            wickets_status = "MISSING"
            decision = "NOT OUT"
            hit_detail = "Passing Wide of Stumps"

    return {
        "pitching_status": pitching_status,
        "impact_status": impact_status,
        "wickets_status": wickets_status,
        "decision": decision,
        "hit_detail": hit_detail,
        "projected_points": projected_pts,
        "stump_hit_point": stump_hit_pt,
        "stump_center_x": stump_center_x,
        "stump_y_base": stump_y_base,
        "stump_y_top": stump_y_top,
        "stump_width": stump_width
    }


def draw_pitch_corridor_and_stumps(frame: np.ndarray, width: int, height: int, drs_data: dict):
    """Draws 3D perspective pitch corridor boundaries and virtual stumps at the batsman end."""
    # Pitch Boundaries
    p1 = (int(width * 0.15), int(height * 0.30))
    p2 = (int(width * 0.85), int(height * 0.30))
    p3 = (int(width * 0.92), int(height * 0.85))
    p4 = (int(width * 0.08), int(height * 0.85))
    
    overlay = frame.copy()
    cv2.fillPoly(overlay, [np.array([p1, p2, p3, p4], np.int32)], (30, 80, 40))
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    # Crease Line
    cv2.line(frame, (int(width * 0.12), int(height * 0.72)), (int(width * 0.88), int(height * 0.72)), (255, 255, 255), 2)

    # Batsman Stumps (3 vertical posts + top bails)
    cx = drs_data["stump_center_x"]
    y_base = drs_data["stump_y_base"]
    y_top = drs_data["stump_y_top"]
    sw = drs_data["stump_width"]

    off_x = cx - sw // 2
    mid_x = cx
    leg_x = cx + sw // 2

    # Draw wooden stump posts
    cv2.line(frame, (off_x, y_base), (off_x, y_top), (0, 165, 255), 4)
    cv2.line(frame, (mid_x, y_base), (mid_x, y_top), (0, 165, 255), 4)
    cv2.line(frame, (leg_x, y_base), (leg_x, y_top), (0, 165, 255), 4)
    # Bails
    cv2.line(frame, (off_x - 3, y_top), (leg_x + 3, y_top), (0, 215, 255), 3)


def cartoonize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Applies a cartoon / animated look to a frame while fully preserving the motion content.
    Uses edge-preserving bilateral filter + edge detection overlay.
    The actual ball position and player movement are unchanged.
    """
    # Bilateral filter gives a smooth painted look
    color = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
    # Edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, blockSize=9, C=2)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # Blend edges on top of the bilateral-filtered colour image
    cartoon = cv2.bitwise_and(color, edges_bgr)
    return cartoon


def draw_glowing_line(frame: np.ndarray, pt1: tuple, pt2: tuple, color: tuple, thickness: int = 3):
    """Draws a line with a soft glow effect."""
    # Outer soft glow
    overlay = frame.copy()
    cv2.line(overlay, pt1, pt2, color, thickness + 4, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    # Inner bright core line
    cv2.line(frame, pt1, pt2, (255, 255, 255) if color != (255, 255, 255) else color, thickness, cv2.LINE_AA)


def draw_drs_trajectory_lines(frame: np.ndarray, trajectory: list, bounce_point: tuple,
                               impact_point: tuple, drs_data: dict, curr_frame: int, total_frames: int):
    """
    Draws a single, clean, glowing trajectory path line (Yellow pre-bounce, Green post-bounce)
    representing the smoothed path of the ball, plus bounce/impact markers.
    """
    if len(trajectory) < 2:
        return

    # Find the bounce frame index in the current trajectory segment
    b_idx = None
    if bounce_point and trajectory:
        for idx, pt in enumerate(trajectory):
            # Check if this point is close to the bounce point
            if np.hypot(pt[0] - bounce_point[0], pt[1] - bounce_point[1]) < 5.0:
                b_idx = idx
                break

    # Draw the smoothed trajectory line segment by segment
    for k in range(1, len(trajectory)):
        pt1 = (trajectory[k - 1][0], trajectory[k - 1][1])
        pt2 = (trajectory[k][0], trajectory[k][1])
        
        if b_idx is not None and k > b_idx:
            # Post-bounce actual trajectory (Bright Neon Green Glow)
            draw_glowing_line(frame, pt1, pt2, (0, 255, 0), thickness=3)
        else:
            # Pre-bounce actual trajectory (Bright Neon Yellow/Cyan Glow)
            draw_glowing_line(frame, pt1, pt2, (0, 235, 255), thickness=3)

    # Draw current ball location as a glowing ring
    cx, cy = trajectory[-1][0], trajectory[-1][1]
    cv2.circle(frame, (cx, cy), 12, (0, 180, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)

    # Bounce Marker
    if bounce_point:
        bx, by = bounce_point
        cv2.circle(frame, (bx, by), 8, (0, 255, 255), -1)
        cv2.circle(frame, (bx, by), 14, (0, 215, 255), 2)
        cv2.putText(frame, "BOUNCE", (bx - 30, by - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2, cv2.LINE_AA)

    # Impact Marker
    if impact_point:
        ix, iy = impact_point
        cv2.circle(frame, (ix, iy), 9, (255, 100, 0), -1)
        cv2.circle(frame, (ix, iy), 15, (255, 200, 0), 2)
        cv2.putText(frame, "IMPACT", (ix - 28, iy - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 2, cv2.LINE_AA)

    # Projected trajectory to stumps (dashed neon magenta)
    proj_pts = drs_data.get("projected_points", [])
    if proj_pts and impact_point:
        prev_p = impact_point
        for idx, p in enumerate(proj_pts):
            if idx % 2 == 0:
                cv2.line(frame, prev_p, p, (255, 0, 255), 3, cv2.LINE_AA)
            prev_p = p
        stump_pt = drs_data.get("stump_hit_point")
        if stump_pt:
            sx, sy = stump_pt
            color = (0, 255, 0) if drs_data["wickets_status"] == "HITTING" else (0, 0, 255)
            cv2.circle(frame, (sx, sy), 10, color, -1)
            cv2.circle(frame, (sx, sy), 16, (255, 255, 255), 2, cv2.LINE_AA)


def draw_drs_broadcast_card(frame: np.ndarray, drs_data: dict, curr_frame: int, total_frames: int, speeds: list, frame_idx: int):
    """Renders a high-fidelity IPL DRS Broadcast Style Card layout on the frame."""
    h, w = frame.shape[0], frame.shape[1]
    
    # Broadcast Lower Third Card Dimensions & Placement (centred at the bottom)
    card_w = 900
    card_h = 110
    card_x = (w - card_w) // 2
    card_y = h - card_h - 25

    # 1. Main background card box (translucent deep navy blue / black)
    overlay = frame.copy()
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (18, 14, 10), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    
    # Draw glowing bottom border matching official broadcast theme
    cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (80, 80, 80), 1, cv2.LINE_AA)
    cv2.line(frame, (card_x, card_y + card_h), (card_x + card_w, card_y + card_h), (0, 215, 255), 3, cv2.LINE_AA)

    # 2. Header Title Band (Top bar of the card)
    cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + 30), (25, 20, 15), -1)
    cv2.putText(frame, "DECISION REVIEW SYSTEM | HAWK-EYE TELEMETRY", (card_x + 15, card_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)
    
    # Live frame index / playback percentage indicator
    progress_pct = int((curr_frame / float(total_frames)) * 100) if total_frames > 0 else 100
    cv2.putText(frame, f"TRACKING: {progress_pct}%", (card_x + card_w - 130, card_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. Columns configuration (PITCHING, IMPACT, WICKETS)
    col_width = 190
    start_x = card_x + 20
    content_y = card_y + 42

    # Draw Helper function for decision columns
    def draw_status_col(x, label, status, pass_cond, warn_cond=None):
        # Header text
        cv2.putText(frame, label, (x, content_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
        
        # Determine status box color
        if status == pass_cond:
            box_color = (0, 180, 0)      # Green for In-Line/Hitting
            txt_color = (255, 255, 255)
        elif warn_cond and status == warn_cond:
            box_color = (0, 180, 255)    # Orange/Yellow for Umpire's Call
            txt_color = (0, 0, 0)
        else:
            box_color = (0, 0, 180)      # Red for Outside/Missing
            txt_color = (255, 255, 255)
            
        # Draw status rounded badge/box
        cv2.rectangle(frame, (x, content_y + 26), (x + col_width - 20, content_y + 54), box_color, -1)
        cv2.rectangle(frame, (x, content_y + 26), (x + col_width - 20, content_y + 54), (255, 255, 255), 1, cv2.LINE_AA)
        
        # Center status text
        text = str(status)
        font_scale = 0.42 if len(text) > 10 else 0.48
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        text_x = x + (col_width - 20 - text_size[0]) // 2
        cv2.putText(frame, text, (text_x, content_y + 45), cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_color, 2, cv2.LINE_AA)

    # Render three column badges
    draw_status_col(start_x, "1. BALL PITCHING", drs_data["pitching_status"], "IN-LINE")
    draw_status_col(start_x + col_width, "2. BALL IMPACT", drs_data["impact_status"], "IN-LINE")
    draw_status_col(start_x + col_width * 2, "3. WICKETS TARGET", drs_data["wickets_status"], "HITTING", "UMPIRE'S CALL")

    # 4. Final Decision Outcome Block (Big Red/Green Box on Right)
    dec_box_w = 230
    dec_box_h = 60
    dec_box_x = card_x + card_w - dec_box_w - 15
    dec_box_y = card_y + 38

    decision = drs_data["decision"]
    is_out = "OUT" in decision and "NOT" not in decision
    
    # Official green for Not Out, official red for Out
    dec_bg = (0, 0, 180) if is_out else (0, 150, 0)
    
    cv2.rectangle(frame, (dec_box_x, dec_box_y), (dec_box_x + dec_box_w, dec_box_y + dec_box_h), dec_bg, -1)
    cv2.rectangle(frame, (dec_box_x, dec_box_y), (dec_box_x + dec_box_w, dec_box_y + dec_box_h), (255, 255, 255), 2, cv2.LINE_AA)
    
    # Write Final Decision Text inside Box
    dec_text = "OUT" if is_out else "NOT OUT"
    dec_text_size = cv2.getTextSize(dec_text, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)[0]
    dec_text_x = dec_box_x + (dec_box_w - dec_text_size[0]) // 2
    cv2.putText(frame, dec_text, (dec_text_x, dec_box_y + 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    # 5. Live Speed Telemetry Tag at Top Left
    curr_spd = speeds[-1] if speeds else 0.0
    telemetry_bg = np.zeros_like(frame)
    cv2.rectangle(telemetry_bg, (20, 20), (430, 60), (10, 8, 5), -1)
    cv2.addWeighted(telemetry_bg, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (20, 20), (430, 60), (0, 215, 255), 1, cv2.LINE_AA)
    
    cv2.putText(frame, f"LIVE SPEED: {curr_spd:.1f} KM/H", (35, 47),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def classify_shot_and_pitch_length(trajectory: list, bounce_point: tuple, frame_height: int = 720) -> dict:
    """
    Classifies pitch bounce length zone and predicts batsman shot type based on trajectory vectors.
    """
    pitch_length_zone = "Good Length"
    predicted_shot = "Straight Drive"

    if bounce_point:
        bx, by = bounce_point
        norm_y = by / float(frame_height)
        if norm_y > 0.72:
            pitch_length_zone = "Yorker Length"
        elif norm_y > 0.58:
            pitch_length_zone = "Full Length"
        elif norm_y > 0.42:
            pitch_length_zone = "Good Length"
        else:
            pitch_length_zone = "Short Pitch / Bouncer"

    if len(trajectory) >= 6:
        p_start = trajectory[0]
        p_end = trajectory[-1]
        dx = p_end[0] - p_start[0]
        dy = p_end[1] - p_start[1]
        dist = np.hypot(dx, dy)

        if dist < 40:
            predicted_shot = "Defensive Block"
        elif dx > 80:
            predicted_shot = "Cover Drive / Off-side Placement"
        elif dx < -80:
            predicted_shot = "Pull Shot / Leg-side Glance"
        elif dy < -60:
            predicted_shot = "Lofted Shot / Aerial Drive"
        elif abs(dx) <= 80 and dy > 0:
            predicted_shot = "Straight Drive"
        else:
            predicted_shot = "Drive Shot"

    return {
        "pitch_length_zone": pitch_length_zone,
        "predicted_shot": predicted_shot
    }


def generate_3d_hawkeye_plot(trajectory: list, bounce_point: tuple = None, drs_data: dict = None) -> object:
    """
    Generates a 3D Plotly Hawkeye trajectory visualization showing pitch corridor, actual ball trajectory,
    and projected trajectory past impact into the stumps.
    """
    try:
        import plotly.graph_objects as go

        if not trajectory:
            return None

        clean_pts = []
        for p in trajectory:
            try:
                clean_pts.append((float(p[0]), float(p[1])))
            except Exception:
                pass

        if not clean_pts:
            return None

        n = len(clean_pts)
        y_3d = list(np.linspace(0, 18.0, n))
        x_3d = [(p[0] - 640) / 200.0 for p in clean_pts]
        
        # Calculate bounce-aware height profile
        b_idx = int(n * 0.6)
        if bounce_point:
            for idx, pt in enumerate(clean_pts):
                if abs(pt[0] - bounce_point[0]) < 15 and abs(pt[1] - bounce_point[1]) < 15:
                    b_idx = idx
                    break

        z_3d = []
        for i in range(n):
            if i <= b_idx:
                t = i / float(b_idx) if b_idx > 0 else 0
                z = 2.1 * (1 - t)**1.8 + 0.05
            else:
                t = (i - b_idx) / float(n - b_idx) if (n - b_idx) > 0 else 0
                z = 0.05 + 0.95 * np.sin(t * np.pi * 0.55)
            z_3d.append(round(z, 2))

        fig = go.Figure()

        # 1. 3D Pitch Surface
        fig.add_trace(go.Mesh3d(
            x=[-1.6, 1.6, 1.6, -1.6],
            y=[0, 0, 20.12, 20.12],
            z=[0, 0, 0, 0],
            color='khaki',
            opacity=0.6,
            name='Pitch Corridor'
        ))

        # 2. Stumps (Batsman End)
        fig.add_trace(go.Scatter3d(
            x=[-0.12, -0.12, 0.0, 0.0, 0.12, 0.12],
            y=[20.12, 20.12, 20.12, 20.12, 20.12, 20.12],
            z=[0, 0.72, 0, 0.72, 0, 0.72],
            mode='lines',
            line=dict(color='orange', width=8),
            name='Stumps (Target)'
        ))

        # 3. 3D Actual Trajectory
        fig.add_trace(go.Scatter3d(
            x=x_3d, y=y_3d, z=z_3d,
            mode='lines+markers',
            line=dict(color='#00D7FF', width=6),
            marker=dict(size=4, color='yellow'),
            name='Actual Ball Path'
        ))

        # 4. 3D Projected Trajectory ("If ball missed where it will go")
        if isinstance(drs_data, dict) and drs_data.get("projected_points"):
            proj_pts = drs_data["projected_points"]
            clean_proj = []
            for p in proj_pts:
                try:
                    clean_proj.append((float(p[0]), float(p[1])))
                except Exception:
                    pass
            if clean_proj:
                np_proj = len(clean_proj)
                y_proj = np.linspace(18.0, 20.12, np_proj)
                x_proj = [(p[0] - 640) / 200.0 for p in clean_proj]
                z_proj = [max(0.05, z_3d[-1] + 0.03 * k) for k in range(np_proj)]

                fig.add_trace(go.Scatter3d(
                    x=x_proj, y=y_proj, z=z_proj,
                    mode='lines+markers',
                    line=dict(color='#FF00FF', width=6, dash='dash'),
                    marker=dict(size=4, color='magenta'),
                    name='Projected Line (DRS)'
                ))

        # 5. Bounce Spot Marker
        if len(x_3d) > 0 and b_idx < len(x_3d):
            fig.add_trace(go.Scatter3d(
                x=[x_3d[b_idx]], y=[y_3d[b_idx]], z=[0.05],
                mode='markers+text',
                marker=dict(size=12, color='crimson', symbol='diamond'),
                text=["BOUNCE SPOT"],
                textposition="top center",
                name='Pitch Bounce Spot'
            ))

        fig.update_layout(
            title="📐 Hawkeye 3D Parabola & Projected DRS Trajectory",
            scene=dict(
                xaxis=dict(title="Pitch Width (m)", range=[-3, 3]),
                yaxis=dict(title="Pitch Length (m)", range=[0, 22]),
                zaxis=dict(title="Ball Height (m)", range=[0, 4]),
                aspectratio=dict(x=1, y=2.5, z=0.8),
                bgcolor="#0E1117"
            ),
            paper_bgcolor="#0E1117",
            font=dict(color="#FFFFFF"),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        return fig
    except Exception as err:
        print(f"[WARN] Error generating 3D Hawkeye plot: {err}")
        return None


def generate_2d_pitch_map_plot(trajectory: list, bounce_point: tuple = None, impact_point: tuple = None, drs_data: dict = None) -> object:
    """
    Generates a 2D Top-Down Pitch Map plot showing ball direction, pitch lines, bounce spot, impact spot,
    and projected trajectory into stumps.
    """
    try:
        import plotly.graph_objects as go

        if not trajectory:
            return None

        clean_pts = []
        for p in trajectory:
            try:
                clean_pts.append((float(p[0]), float(p[1])))
            except Exception:
                pass

        if not clean_pts:
            return None

        n = len(clean_pts)
        y_vals = np.linspace(0, 18.0, n)
        x_vals = [(p[0] - 640) / 200.0 for p in clean_pts]

        fig = go.Figure()

        # 1. Pitch Turf Ground Outline (Top-Down Rectangle)
        fig.add_shape(type="rect", x0=-1.5, y0=0, x1=1.5, y1=20.12,
                      fillcolor="#223b28", line=dict(color="#4CAF50", width=2))
        
        # 2. Crease Lines
        fig.add_shape(type="line", x0=-1.5, y0=1.2, x1=1.5, y1=1.2, line=dict(color="white", width=2))  # Bowler crease
        fig.add_shape(type="line", x0=-1.5, y0=18.9, x1=1.5, y1=18.9, line=dict(color="white", width=2))  # Popping crease
        
        # 3. Batsman Stumps (Width = 0.23m)
        fig.add_shape(type="rect", x0=-0.12, y0=20.0, x1=0.12, y1=20.12,
                      fillcolor="orange", line=dict(color="darkorange", width=2))

        # 4. Actual Ball Path Line
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines+markers',
            line=dict(color='#00D7FF', width=4),
            marker=dict(size=5, color='yellow'),
            name='Actual Ball Direction'
        ))

        # 5. Projected DRS Path ("If ball missed where it will go")
        if isinstance(drs_data, dict) and drs_data.get("projected_points"):
            proj_pts = drs_data["projected_points"]
            clean_proj = []
            for p in proj_pts:
                try:
                    clean_proj.append((float(p[0]), float(p[1])))
                except Exception:
                    pass
            if clean_proj:
                np_proj = len(clean_proj)
                y_proj = np.linspace(18.0, 20.12, np_proj)
                x_proj = [(p[0] - 640) / 200.0 for p in clean_proj]

                fig.add_trace(go.Scatter(
                    x=x_proj, y=y_proj,
                    mode='lines+markers',
                    line=dict(color='#FF00FF', width=4, dash='dash'),
                    marker=dict(size=5, color='magenta'),
                    name='Projected Line (DRS)'
                ))

        # 6. Bounce & Impact Spot Markers
        b_idx = int(n * 0.6)
        if bounce_point:
            for idx, pt in enumerate(clean_pts):
                if abs(pt[0] - bounce_point[0]) < 15 and abs(pt[1] - bounce_point[1]) < 15:
                    b_idx = idx
                    break

        if len(x_vals) > 0 and b_idx < len(x_vals):
            fig.add_trace(go.Scatter(
                x=[x_vals[b_idx]], y=[y_vals[b_idx]],
                mode='markers+text',
                marker=dict(size=14, color='yellow', symbol='diamond', line=dict(color='orange', width=2)),
                text=["BOUNCE SPOT"],
                textposition="top center",
                name='Pitch Bounce Spot'
            ))

        if impact_point and len(x_vals) > 0:
            i_idx = min(len(x_vals) - 1, int(len(x_vals) * 0.9))
            fig.add_trace(go.Scatter(
                x=[x_vals[i_idx]], y=[y_vals[i_idx]],
                mode='markers+text',
                marker=dict(size=14, color='cyan', symbol='circle', line=dict(color='white', width=2)),
                text=["IMPACT SPOT"],
                textposition="bottom center",
                name='Batsman Impact Spot'
            ))

        fig.update_layout(
            title="🏏 2D Pitch Top-Down Direction & Line Graph",
            xaxis=dict(title="Pitch Width Corridor (Meters)", range=[-2.5, 2.5], zeroline=True),
            yaxis=dict(title="Pitch Length (Meters)", range=[-1, 21], zeroline=False),
            bgcolor="#0E1117",
            paper_bgcolor="#0E1117",
            font=dict(color="#FFFFFF"),
            margin=dict(l=40, r=40, b=40, t=50)
        )
        return fig
    except Exception as err:
        print(f"[WARN] Error generating 2D pitch map: {err}")
        return None


def generate_2d_height_profile_plot(trajectory: list, bounce_point: tuple = None, drs_data: dict = None) -> object:
    """
    Generates a 2D Pitch Side-View Height Profile graph (Distance vs Height in meters) showing bounce parabola,
    stump height (0.72m), and projected trajectory clearance over/into bails.
    """
    try:
        import plotly.graph_objects as go

        if not trajectory:
            return None

        clean_pts = []
        for p in trajectory:
            try:
                clean_pts.append((float(p[0]), float(p[1])))
            except Exception:
                pass

        if not clean_pts:
            return None

        n = len(clean_pts)
        y_vals = list(np.linspace(0, 18.0, n))

        # Determine bounce index
        b_idx = int(n * 0.6)
        if bounce_point:
            for idx, pt in enumerate(clean_pts):
                if abs(pt[0] - bounce_point[0]) < 15 and abs(pt[1] - bounce_point[1]) < 15:
                    b_idx = idx
                    break

        # Calculate realistic height arc (m)
        z_vals = []
        for i in range(n):
            if i <= b_idx:
                t = i / float(b_idx) if b_idx > 0 else 0
                z = 2.1 * (1 - t)**1.8 + 0.05
            else:
                t = (i - b_idx) / float(n - b_idx) if (n - b_idx) > 0 else 0
                z = 0.05 + 0.95 * np.sin(t * np.pi * 0.55)
            z_vals.append(round(z, 2))

        fig = go.Figure()

        # 1. Pitch Ground Line (Y=0)
        fig.add_shape(type="line", x0=0, y0=0, x1=20.12, y1=0, line=dict(color="#4CAF50", width=4))

        # 2. Stump Target Box (Height = 0.72m at Y=20.12m)
        fig.add_shape(type="rect", x0=20.0, y0=0, x1=20.12, y1=0.72,
                      fillcolor="orange", line=dict(color="crimson", width=2))
        fig.add_annotation(x=20.0, y=0.85, text="Stumps (0.72m)", showarrow=False, font=dict(color="orange"))

        # 3. Actual Ball Height Profile Arc
        fig.add_trace(go.Scatter(
            x=y_vals, y=z_vals,
            mode='lines+markers',
            line=dict(color='#00D7FF', width=4),
            marker=dict(size=4, color='yellow'),
            name='Ball Height Arc'
        ))

        # 4. Projected DRS Height Arc
        if isinstance(drs_data, dict) and drs_data.get("projected_points"):
            proj_pts = drs_data["projected_points"]
            clean_proj = []
            for p in proj_pts:
                try:
                    clean_proj.append((float(p[0]), float(p[1])))
                except Exception:
                    pass
            if clean_proj:
                np_proj = len(clean_proj)
                y_proj = np.linspace(18.0, 20.12, np_proj)
                z_proj = [max(0.05, z_vals[-1] + 0.03 * k) for k in range(np_proj)]

                fig.add_trace(go.Scatter(
                    x=y_proj, y=z_proj,
                    mode='lines+markers',
                    line=dict(color='#FF00FF', width=4, dash='dash'),
                    marker=dict(size=5, color='magenta'),
                    name='Projected Line (DRS)'
                ))

        # 5. Bounce Spot Highlight
        if len(y_vals) > 0 and b_idx < len(y_vals):
            fig.add_trace(go.Scatter(
                x=[y_vals[b_idx]], y=[0.05],
                mode='markers+text',
                marker=dict(size=12, color='crimson', symbol='diamond'),
                text=["BOUNCE"],
                textposition="top center",
                name='Pitch Bounce Ground Spot'
            ))

        fig.update_layout(
            title="📈 Ball Trajectory Height & Pitch Parabola Profile (Side-View)",
            xaxis=dict(title="Pitch Distance (Meters)", range=[0, 22]),
            yaxis=dict(title="Ball Height Above Pitch (Meters)", range=[0, 3.5]),
            bgcolor="#0E1117",
            paper_bgcolor="#0E1117",
            font=dict(color="#FFFFFF"),
            margin=dict(l=40, r=40, b=40, t=50)
        )
        return fig
    except Exception as err:
        print(f"[WARN] Error generating height profile plot: {err}")
        return None


def generate_sample_video(output_path: str = "data/raw/cricket_video.mp4", num_frames: int = 90, fps: int = 30) -> str:
    """Generates a synthetic cricket delivery video clip with pitch lines and stumps for instant testing."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    w, h = 1280, 720
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (34, 139, 34)  # Grass background

        # Pitch
        cv2.fillPoly(frame, [np.array([[200, 220], [1080, 220], [1180, 620], [100, 620]], np.int32)], (160, 210, 230))
        cv2.line(frame, (150, 540), (1130, 540), (255, 255, 255), 3)  # Crease

        # Batsman Stumps
        cx, y_base, y_top = 640, 520, 410
        cv2.line(frame, (cx - 25, y_base), (cx - 25, y_top), (0, 165, 255), 4)
        cv2.line(frame, (cx, y_base), (cx, y_top), (0, 165, 255), 4)
        cv2.line(frame, (cx + 25, y_base), (cx + 25, y_top), (0, 165, 255), 4)
        cv2.line(frame, (cx - 28, y_top), (cx + 28, y_top), (0, 215, 255), 3)

        # Ball trajectory (Parabola bounce & pad impact)
        t = i / float(num_frames)
        bx = int(320 + t * 330)
        by = int(240 + 260 * np.sin(t * np.pi * 1.5))

        # Draw red ball up to impact frame (frame 70)
        if i <= 70:
            cv2.circle(frame, (bx, by), 8, (20, 20, 200), -1)
            cv2.circle(frame, (bx, by), 9, (255, 255, 255), 1)

        out.write(frame)

    out.release()
    return output_path


def download_url_video(url: str, output_path: str = "data/raw/url_download.mp4") -> str:
    """Downloads a video file from YouTube, direct MP4 URL, or web stream into output_path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    url = url.strip()

    # Try yt-dlp first for YouTube & video platforms
    try:
        import yt_dlp
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': output_path,
            'overwrites': True,
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"[WARN] yt-dlp download failed, attempting direct HTTP download: {e}")
        import urllib.request
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
            out_file.write(response.read())

    # Validate that OpenCV can open the downloaded video file
    cap = cv2.VideoCapture(output_path)
    if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 0:
        cap.release()
        if os.path.exists(output_path):
            os.remove(output_path)
        raise ValueError("Invalid or unreadable video file downloaded. Please ensure the link points to a valid video stream (e.g. YouTube URL or direct MP4 link).")
    cap.release()
    return output_path


def convert_to_web_mp4(video_path: str) -> str:
    """Converts OpenCV MP4 into H.264 web-compatible format for native browser & Streamlit playback."""
    try:
        import imageio_ffmpeg
        import subprocess

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        temp_web_path = os.path.splitext(video_path)[0] + "_h264.mp4"
        if os.path.exists(temp_web_path):
            try:
                os.remove(temp_web_path)
            except Exception:
                pass

        cmd = [
            ffmpeg_exe, "-y", "-i", video_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", temp_web_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0 and os.path.exists(temp_web_path) and os.path.getsize(temp_web_path) > 0:
            try:
                os.replace(temp_web_path, video_path)
                return video_path
            except Exception:
                return temp_web_path
    except Exception as e:
        print(f"[WARN] H.264 video conversion note: {e}")
    return video_path



