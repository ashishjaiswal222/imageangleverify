# Configuration constants for heuristic thresholds

# --- Image I/O ---
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MIN_RESOLUTION = (480, 480) # (width, height)
ALLOWED_FORMATS = {"image/jpeg", "image/png", "image/webp"}

# --- Face Detection ---
MIN_FACE_DETECTION_CONFIDENCE = 0.5

# --- Angle Detection (Yaw/Pitch/Roll) ---
# Yaw thresholds (degrees)
YAW_FRONT_MAX = 15.0
YAW_TURN_MIN = 25.0
YAW_TURN_MAX = 90.0

# Pitch threshold (degrees)
PITCH_MAX = 20.0

# --- Blur Detection ---
LAPLACIAN_VAR_THRESHOLD = 10.0 # Extremely permissive to allow smooth AI images to pass, while catching heavy camera blur

# --- Lighting Detection ---
MIN_BRIGHTNESS = 80.0
MAX_BRIGHTNESS = 230.0

# --- Eye/Occlusion Checks ---
EAR_THRESHOLD = 0.20 # Eye Aspect Ratio for blink/closed eye detection
EYEWEAR_CONFIDENCE_THRESHOLD = 0.3 # If iris landmarks are missing and face is present

# --- Heavy Editing (ELA) ---
ELA_JPEG_QUALITY = 90
ELA_VARIANCE_THRESHOLD = 250.0 # Needs calibration against real data

# --- Face Centering ---
FACE_CENTER_TOLERANCE_PCT = 0.25 # 25% from center

# --- Pose Landmarker (Full Body) ---
MIN_POSE_PRESENCE_SCORE = 0.5
MIN_FACE_DETECTION_SCORE = 0.5 # Reverted to 0.5 to allow real human side-profiles to pass
MAX_FACE_EMBEDDING_NORM = 26.0 # Drastically increased to 26.0 so we don't falsely reject highly stylized real humans
REQUIRED_POSE_LANDMARKS = [
    0,  # nose (head approx)
    11, 12, # shoulders
    23, 24, # hips
    25, 26, # knees
    27, 28  # ankles
]
