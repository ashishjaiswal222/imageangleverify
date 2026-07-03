import asyncio
from app.verification.pipeline import verify_image
from app.verification.models_loader import get_models

async def main():
    print("Loading models...")
    get_models() # Warm up
    print("Models loaded. Running verification...")
    
    try:
        with open("tests/sampleimage/fullbody.png", "rb") as f:
            file_bytes = f.read()
            
        # Test fullbody image as 'front' position
        result = verify_image(file_bytes, "image/png", "front")
        print("\n=== TEST RESULT: Full Body uploaded as 'front' ===")
        print(f"Passed: {result.passed}")
        
        angle_check = result.checks.get("angle_match")
        print(f"Angle Match Check: {angle_check}")
        
        person_check = result.checks.get("person_and_face")
        print(f"Person Check: {person_check}")
        
        if not result.passed:
            print(f"Primary Reason: {result.primary_reason}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
