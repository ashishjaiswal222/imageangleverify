import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

MODELS_TO_DOWNLOAD = {
    "blaze_face_short_range.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "selfie_multiclass_256x256.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
}

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for filename, url in MODELS_TO_DOWNLOAD.items():
        filepath = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(filepath):
            logger.info(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                logger.info(f"Successfully downloaded {filename}")
            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")
        else:
            logger.info(f"{filename} already exists, skipping.")

if __name__ == "__main__":
    main()
