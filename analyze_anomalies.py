import cv2
import glob
import sys
import mediapipe as mp
import numpy as np

# Add project path to sys.path so we can import modules
sys.path.append(r"c:\Users\Ashish jaiswal\OneDrive\Desktop\projects\photoanglecheck\photo-verify-api")
from app.verification.models_loader import get_models

def analyze_image(path):
    print(f"\n--- Analyzing: {path} ---")
    image_cv = cv2.imread(path)
    if image_cv is None:
        print("Failed to read image")
        return
        
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    models = get_models()
    
    # MediaPipe Face Detector
    fd_res = models.face_detector.detect(mp_image)
    print(f"MediaPipe Face Detector faces: {len(fd_res.detections) if fd_res.detections else 0}")
    
    # MediaPipe Pose
    pose_res = models.pose_landmarker.detect(mp_image)
    print(f"MediaPipe Poses: {len(pose_res.pose_landmarks) if pose_res.pose_landmarks else 0}")
    
    # MediaPipe Face Mesh
    fm_res = models.face_landmarker.detect(mp_image)
    print(f"MediaPipe Face Meshes: {len(fm_res.face_landmarks) if fm_res.face_landmarks else 0}")
    
    # InsightFace (buffalo_l)
    # InsightFace expects BGR image
    faces = models.face_analysis.get(image_cv)
    print(f"InsightFace Faces: {len(faces)}")
    for i, f in enumerate(faces):
        print(f"  Face {i}: det_score={f.det_score:.3f}")
        
    # Image Segmenter (to see if it detects face-skin = 3)
    seg_res = models.image_segmenter.segment(mp_image)
    if seg_res.category_mask:
        mask = seg_res.category_mask.numpy_view()
        face_skin_pixels = np.sum(mask == 3)
        total_pixels = mask.size
        print(f"Image Segmenter Face-Skin Ratio: {face_skin_pixels/total_pixels:.4f}")

if __name__ == "__main__":
    cartoon_files = glob.glob(r"C:\Users\Ashish jaiswal\.gemini\antigravity\brain\206a5104-c14a-46b3-9b88-293d0df72ae8\cartoon_face*.png")
    group_files = glob.glob(r"C:\Users\Ashish jaiswal\.gemini\antigravity\brain\206a5104-c14a-46b3-9b88-293d0df72ae8\group_photo*.png")
    
    if cartoon_files: analyze_image(cartoon_files[0])
    if group_files: analyze_image(group_files[0])
