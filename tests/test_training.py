from __future__ import annotations

from src.train import train_and_save


def test_training_creates_required_artifacts(tmp_path):
    result = train_and_save(random_state=42, artifact_dir=tmp_path)
    assert (tmp_path / "pipeline.joblib").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "model_card.json").exists()
    assert (tmp_path / "feature_importance.json").exists()
    assert result["selected_model"]
    assert "f1" in result["metrics"]
