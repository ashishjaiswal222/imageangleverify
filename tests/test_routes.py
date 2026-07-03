import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
import io
from PIL import Image

@pytest.fixture
def test_image():
    # Create a simple valid mock image (blue square)
    img = Image.new('RGB', (500, 500), color = 'blue')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/v1/photo-verify/health")
    # By default in tests IS_READY is false because we don't await warmup
    assert response.status_code == 503

@pytest.mark.asyncio
async def test_check_single_invalid_position(test_image):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {'file': ('test.jpg', test_image, 'image/jpeg')}
        data = {'position': 'invalid_pos'}
        response = await ac.post("/api/v1/photo-verify/check-single", files=files, data=data)
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_check_single_malformed_upload():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Pass a text file disguised as an image
        bad_file = b"This is not an image."
        files = {'file': ('bad.jpg', bad_file, 'image/jpeg')}
        data = {'position': 'front'}
        response = await ac.post("/api/v1/photo-verify/check-single", files=files, data=data)
    
    # Should return 422 Unprocessable Entity for corrupt image
    assert response.status_code == 422
    json_resp = response.json()
    assert "detail" in json_resp
    assert "msg" in json_resp["detail"]
    assert "corrupted" in json_resp["detail"]["msg"]



@pytest.mark.asyncio
async def test_check_batch_missing():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Send empty batch
        response = await ac.post("/api/v1/photo-verify/check-batch")
    
    assert response.status_code == 200
    data = response.json()
    assert data["overall_passed"] is False
    assert data["results"]["front"]["status"] == "MISSING"
