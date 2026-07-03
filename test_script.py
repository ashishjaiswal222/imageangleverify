import requests
import os
import json

base_url = 'http://127.0.0.1:8000/api/v1/photo-verify/check-single'
img_dir = 'tests/sampleimage'
files = ['front.png', 'left.png', 'right.png', 'back.png', 'fullbody.png']
positions = ['front', 'left', 'right', 'back', 'full_body']

for fname, pos in zip(files, positions):
    path = os.path.join(img_dir, fname)
    if not os.path.exists(path):
        print(f"File {path} not found.")
        continue
    
    with open(path, 'rb') as f:
        res = requests.post(base_url, data={'position': pos}, files={'file': (fname, f, 'image/png')})
    
    print(f"\n--- Testing {fname} as {pos} ---")
    print(f"Status: {res.status_code}")
    if res.status_code == 200:
        data = res.json()
        print(f"Passed: {data.get('passed')}")
        print(f"Primary Reason: {data.get('primary_reason')}")
        print("Checks Summary:")
        for check_name, check_res in data.get('checks', {}).items():
            print(f"  {check_name}: passed={check_res['passed']}, score={check_res.get('score')}")
            if not check_res['passed']:
                print(f"    -> msg: {check_res.get('message')}")
                if check_res.get('details'):
                    print(f"    -> details: {check_res.get('details')}")
    else:
        print(res.text)
