from pathlib import Path
import os
import pytest


def test_uploaded_yolo_model_loads_and_has_waste_label():
    # Optional/manual because loading torch/ultralytics can be slow on small runtimes.
    if os.environ.get("RUN_YOLO_LOAD_TEST") != "1":
        pytest.skip("Set RUN_YOLO_LOAD_TEST=1 to validate the YOLO .pt load locally.")
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO
    model_path = Path("models/best.pt")
    assert model_path.exists(), "models/best.pt should exist for trained-model builds"
    model = YOLO(str(model_path))
    names = {int(k): v for k, v in model.names.items()}
    assert names.get(0) == "waste"
