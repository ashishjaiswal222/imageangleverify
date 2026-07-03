import requests
import os
import cv2
import numpy as np
import time

base_url = 'http://127.0.0.1:8000/api/v1/photo-verify/check-single'
img_dir = 'tests/sampleimage'
front_path = os.path.join(img_dir, 'front.png')

# Ensure front.png exists
if not os.path.exists(front_path):
    print("Base image front.png not found!")
    exit(1)

# Read base image
img = cv2.imread(front_path)

def test_heuristic(test_name, modified_img, position="front", expected_reason=None):
    print(f"\n--- Testing: {test_name} ---")
    tmp_path = 'temp_test.jpg'
    cv2.imwrite(tmp_path, modified_img)
    
    with open(tmp_path, 'rb') as f:
        res = requests.post(base_url, data={'position': position}, files={'file': ('temp.jpg', f, 'image/jpeg')})
    
    if res.status_code == 200:
        data = res.json()
        passed = data.get('passed')
        reason = data.get('primary_reason')
        print(f"Overall Passed: {passed}")
        if not passed and reason:
            code = reason.get('code')
            print(f"Failed as expected? {code == expected_reason} (Got: {code}, Expected: {expected_reason})")
            print(f"Message: {reason.get('message')}")
        elif passed and expected_reason:
            print(f"FAIL: Image was supposed to fail with {expected_reason} but it PASSED!")
            
        # Print the specific check that failed
        for check_name, check_res in data.get('checks', {}).items():
            if not check_res['passed']:
                print(f"  -> Check '{check_name}' failed: score={check_res.get('score')}")
    else:
        print(f"Error: {res.status_code} - {res.text}")

# 1. Blur Test
blurred = cv2.GaussianBlur(img, (35, 35), 0)
test_heuristic("BLUR DETECTOR", blurred, expected_reason="IMAGE_BLURRY")

# 2. Too Dark
dark = cv2.convertScaleAbs(img, alpha=0.2, beta=0)
test_heuristic("POOR LIGHTING (DARK)", dark, expected_reason="POOR_LIGHTING")

# 3. Too Bright
bright = cv2.convertScaleAbs(img, alpha=2.5, beta=50)
test_heuristic("POOR LIGHTING (BRIGHT)", bright, expected_reason="POOR_LIGHTING")

# 4. Group Photo (2 Faces)
# Concatenate the image horizontally with itself
group = np.concatenate((img, img), axis=1)
test_heuristic("GROUP PHOTO (2 Faces)", group, expected_reason="GROUP_PHOTO_DETECTED")

# 5. Not Full Body
# Submitting the 'front' face photo to the 'full_body' position
test_heuristic("NOT FULL BODY (Face submitted to full_body box)", img, position="full_body", expected_reason="NOT_FULL_BODY")

# 6. Face Not Centered
# Create a large black canvas and put the face in the top left corner
h, w, c = img.shape
canvas = np.zeros((h*2, w*2, c), dtype=np.uint8)
# Resize face to be smaller
small_face = cv2.resize(img, (w//2, h//2))
canvas[0:h//2, 0:w//2] = small_face
test_heuristic("FACE NOT CENTERED", canvas, expected_reason="FACE_NOT_CENTERED")

# 7. Eyewear / Occlusion
# Draw a solid black box over the middle of the face (where eyes usually are) to simulate heavy sunglasses
eyewear_img = img.copy()
# Assuming face is somewhat centered, draw a black rectangle
cv2.rectangle(eyewear_img, (int(w*0.2), int(h*0.3)), (int(w*0.8), int(h*0.5)), (0,0,0), -1)
test_heuristic("EYEWEAR / OCCLUSION", eyewear_img, expected_reason="EYEWEAR_DETECTED")

# Clean up
if os.path.exists('temp_test.jpg'):
    os.remove('temp_test.jpg')
