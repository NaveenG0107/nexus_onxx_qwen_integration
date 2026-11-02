from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health_check")
    assert response.status_code == 200
    assert response.json() == {"message": "light-weight-llm service is running"}


@patch("src.generation.api.get_model_manager")
def test_generate_endpoint_qwen(mock_get_manager):
    mock_manager = MagicMock()
    mock_manager.generate.return_value = ("Qwen response text.", "qwen2.5-0.5b-onnx")
    mock_get_manager.return_value = mock_manager

    payload = {"prompt": "Explain AI", "max_tokens": 50, "use_custom_onnx": False}
    response = client.post("/api/v1/chat", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "response": "Qwen response text.",
        "model_used": "qwen2.5-0.5b-onnx",
    }
    mock_manager.generate.assert_called_once_with(
        prompt="Explain AI", max_tokens=50, use_custom_onnx=False
    )


@patch("src.generation.api.get_model_manager")
def test_generate_endpoint_custom(mock_get_manager):
    mock_manager = MagicMock()
    mock_manager.generate.return_value = ("Mitrahsoft second branch is located in madurai.", "custom-slm-onnx")
    mock_get_manager.return_value = mock_manager

    payload = {"prompt": "Where is Mitrahsoft located?", "max_tokens": 50, "use_custom_onnx": True}
    response = client.post("/api/v1/chat", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "response": "Mitrahsoft second branch is located in madurai.",
        "model_used": "custom-slm-onnx",
    }
    mock_manager.generate.assert_called_once_with(
        prompt="Where is Mitrahsoft located?", max_tokens=50, use_custom_onnx=True
    )
