import pytest
import numpy as np
from app.verification.identity_checks import verify_identity_consistency

def test_verify_identity_consistency_match(mocker):
    # Mock embeddings to be identical
    mocker.patch("app.verification.identity_checks.get_face_embedding", return_value=np.array([0.5]*128))
    # Mock identical histograms
    mocker.patch("app.verification.identity_checks.get_clothing_histogram", return_value=np.ones((16,16), dtype=np.float32))
    
    # We don't need real images because we mocked the extraction functions
    images = {
        "front": np.zeros((10,10,3), dtype=np.uint8),
        "left": np.zeros((10,10,3), dtype=np.uint8),
        "back": np.zeros((10,10,3), dtype=np.uint8)
    }
    
    res = verify_identity_consistency(images)
    assert res.passed is True
    # The cosine distance between identical vectors is 0
    assert res.face_similarity_pairs["front_left"] < 0.01 
    # The bhattacharyya distance between identical histograms is 0, score is 1.0
    assert res.clothing_consistency_score == 1.0

def test_verify_identity_consistency_mismatch(mocker):
    # Return opposite embeddings to ensure mismatch
    def mock_emb(img):
        if img.shape[0] == 10: 
            return np.array([0.5]*128)
        else: 
            return np.array([-0.5]*128)
            
    mocker.patch("app.verification.identity_checks.get_face_embedding", side_effect=mock_emb)
    mocker.patch("app.verification.identity_checks.get_clothing_histogram", return_value=None)
    
    images = {
        "front": np.zeros((10,10,3), dtype=np.uint8), # returns [0.5]
        "left": np.zeros((11,11,3), dtype=np.uint8)  # returns [-0.5] -> completely opposite
    }
    
    res = verify_identity_consistency(images)
    assert res.passed is False
    assert res.reason_code == "IDENTITY_MISMATCH_ACROSS_PHOTOS"
    assert "doesn't match" in res.message
