import httpx
import asyncio

async def test_front_fallback():
    url = "http://127.0.0.1:8000/api/v1/photo-verify/check-single"
    
    # Test 1: Uploading a full body image for the 'front' position
    # The face detector should find a face, but face mesh will likely fail (too small).
    # It should fallback to Pose Landmarker and pass the 'front' check.
    
    try:
        with open("tests/sampleimage/fullbody.png", "rb") as f:
            files = {"file": ("fullbody.png", f, "image/png")}
            data = {"position": "front"}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, data=data, files=files, timeout=30.0)
                
                print("=== TEST 1: Full Body uploaded as Front ===")
                print(f"Status Code: {response.status_code}")
                if response.status_code == 200:
                    json_data = response.json()
                    print(f"Passed: {json_data.get('passed')}")
                    print(f"Angle Match Check: {json_data.get('checks', {}).get('angle_match', {})}")
                    if json_data.get('passed') is False:
                        print(f"Primary Reason: {json_data.get('primary_reason')}")
                else:
                    print(response.text)
                    
    except Exception as e:
        print(f"Error reading or sending file: {e}")

if __name__ == "__main__":
    asyncio.run(test_front_fallback())
