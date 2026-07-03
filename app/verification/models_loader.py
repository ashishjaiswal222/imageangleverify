import os
import mediapipe as mp
import insightface
from insightface.app import FaceAnalysis
from nudenet import NudeDetector

class MediaPipeModels:
    _instance = None

    def __init__(self):
        self.face_detector = None
        self.pose_landmarker = None
        self.face_landmarker = None
        self.image_segmenter = None
        self.face_analysis = None
        self.nude_detector = None
        self._load_models()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_models(self):
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        
        # 1. Face Detector
        fd_path = os.path.join(models_dir, "blaze_face_short_range.tflite")
        fd_options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=fd_path),
            min_detection_confidence=0.5
        )
        self.face_detector = mp.tasks.vision.FaceDetector.create_from_options(fd_options)

        # 2. Pose Landmarker
        pl_path = os.path.join(models_dir, "pose_landmarker_heavy.task")
        pl_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=pl_path),
            output_segmentation_masks=False
        )
        self.pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(pl_options)

        # 3. Face Landmarker (with blendshapes and iris)
        fl_path = os.path.join(models_dir, "face_landmarker.task")
        fl_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=fl_path),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(fl_options)

        # 4. Image Segmenter (Selfie Multiclass)
        seg_path = os.path.join(models_dir, "selfie_multiclass_256x256.tflite")
        seg_options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=seg_path),
            output_category_mask=True,
            output_confidence_masks=False
        )
        self.image_segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(seg_options)

        # 5. Identity Verification (InsightFace)
        # Note: This will download 'buffalo_l' models to ~/.insightface on first run if missing.
        self.face_analysis = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))

        # 6. Explicit Content (NudeNet)
        # Note: This downloads the default model (~80MB) to ~/.NudeNet on first run if missing.
        self.nude_detector = NudeDetector()

def get_models() -> MediaPipeModels:
    return MediaPipeModels.get_instance()
