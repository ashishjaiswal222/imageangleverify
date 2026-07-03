import requests
import json
import os
import urllib.request

base_url = 'http://127.0.0.1:8000/api/v1/photo-verify/check-batch'
img_dir = 'tests/sampleimage'

# Files
front = os.path.join(img_dir, 'front.png')
left = os.path.join(img_dir, 'left.png')
right = os.path.join(img_dir, 'right.png')
back = os.path.join(img_dir, 'back.png')
fullbody = os.path.join(img_dir, 'fullbody.png')

# Download a "different person" photo for identity mismatch testing
diff_person = 'diff_person.jpg'
if not os.path.exists(diff_person):
    res = requests.get("https://upload.wikimedia.org/wikipedia/commons/8/85/Elon_Musk_Royal_Society_%28crop1%29.jpg", headers={"User-Agent": "Mozilla/5.0"})
    with open(diff_person, 'wb') as f:
        f.write(res.content)

def run_test(name, files_dict):
    print(f"\n======================================")
    print(f"TEST: {name}")
    print(f"======================================")
    
    # Prepare files payload
    payload = {}
    opened_files = []
    
    for key, path in files_dict.items():
        f = open(path, 'rb')
        opened_files.append(f)
        
        # detect extension
        ext = path.split('.')[-1]
        mime = f"image/{'jpeg' if ext == 'jpg' else ext}"
        
        payload[key] = (os.path.basename(path), f, mime)
        
    try:
        res = requests.post(base_url, files=payload)
        data = res.json()
        print(f"Overall Passed: {data.get('overall_passed')}")
        
        # Identity
        id_cons = data.get('identity_consistency')
        if id_cons:
            print(f"Identity Passed: {id_cons.get('passed')}")
            if not id_cons.get('passed'):
                print(f"  -> Identity Reason: {id_cons.get('reason_code')} - {id_cons.get('message')}")
        
        # Individual results
        print("File Results:")
        for pos, result in data.get('results', {}).items():
            if result.get("passed"):
                print(f"  [{pos}] PASSED")
            else:
                reason = result.get('primary_reason', {})
                if not reason:
                    reason = result # For missing/error
                print(f"  [{pos}] FAILED -> {reason.get('code', reason.get('status'))}: {reason.get('message')}")
                
    finally:
        for f in opened_files:
            f.close()


# --- TEST 1: The Duplication Test ---
# Uploading the exact same 'front.png' for every single angle
run_test("DUPLICATION TEST (Same 'front' photo for all 5 slots)", {
    'front': front,
    'left': front,
    'right': front,
    'back': front,
    'full_body': front
})

# --- TEST 2: The Identity Mismatch Test ---
# Uploading correct angles, but one of the photos is a completely different person!
run_test("IDENTITY MISMATCH TEST (Different person in the 'front' photo)", {
    'front': diff_person, # Different person!
    'left': left,
    'right': right,
    'back': back,
    'full_body': fullbody
})

# --- TEST 3: The Missing File Test ---
# Submitting a batch without a 'back' photo
run_test("MISSING FILE TEST (No 'back' photo provided)", {
    'front': front,
    'left': left,
    'right': right,
    'full_body': fullbody
})

