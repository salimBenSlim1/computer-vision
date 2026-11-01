import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import app


class FakeModel:
    def __init__(self):
        self.names = {0: "person"}

    def train(self, **kwargs):
        raise ValueError("training failed")

    def __call__(self, image, **kwargs):
        return [SimpleNamespace(boxes=[])]


class AppGuardrailTests(unittest.TestCase):
    def test_missing_custom_weights_do_not_replace_pretrained_class_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_yaml = root / "data.yaml"
            data_yaml.write_text("names:\n  0: cup\n  1: espresso\n", encoding="utf-8")
            pretrained_model = FakeModel()

            with self.assertLogs(app.logger, level="WARNING"):
                with patch.object(app, "YOLO", return_value=pretrained_model):
                    agent = app.CoffeeDetectionAgent(
                        model_path=str(root / "missing.pt"),
                        data_yaml=str(data_yaml),
                        config=app.DetectionConfig(reports_dir=str(root / "reports")),
                    )

            self.assertEqual(pretrained_model.names, {0: "person"})
            self.assertFalse(agent._custom_model_loaded)

    def test_custom_weights_use_the_configured_zero_based_class_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_yaml = root / "data.yaml"
            data_yaml.write_text("names:\n  0: cup\n  1: espresso\n", encoding="utf-8")
            model_path = root / "custom.pt"
            model_path.touch()
            custom_model = FakeModel()
            custom_model.names = {0: "old_a", 1: "old_b"}

            with self.assertLogs(app.logger, level="INFO"):
                with patch.object(app, "YOLO", return_value=custom_model):
                    agent = app.CoffeeDetectionAgent(
                        model_path=str(model_path),
                        data_yaml=str(data_yaml),
                        config=app.DetectionConfig(reports_dir=str(root / "reports")),
                    )

            self.assertEqual(custom_model.names, {0: "cup", 1: "espresso"})

    def test_custom_weight_class_count_must_match_data_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_yaml = root / "data.yaml"
            data_yaml.write_text("names:\n  0: cup\n  1: espresso\n", encoding="utf-8")
            model_path = root / "custom.pt"
            model_path.touch()

            with patch.object(app, "YOLO", return_value=FakeModel()):
                with self.assertRaisesRegex(ValueError, "class count"):
                    app.CoffeeDetectionAgent(
                        model_path=str(model_path),
                        data_yaml=str(data_yaml),
                        config=app.DetectionConfig(reports_dir=str(root / "reports")),
                    )

    def test_sparse_dataset_class_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_yaml = root / "data.yaml"
            data_yaml.write_text("names:\n  0: cup\n  2: espresso\n", encoding="utf-8")

            with patch.object(app, "YOLO", return_value=FakeModel()):
                with self.assertRaisesRegex(ValueError, "contiguous"):
                    app.CoffeeDetectionAgent(
                        model_path=str(root / "missing.pt"),
                        data_yaml=str(data_yaml),
                        config=app.DetectionConfig(reports_dir=str(root / "reports")),
                    )

    def test_image_inference_rejects_pretrained_fallback(self):
        agent = app.CoffeeDetectionAgent.__new__(app.CoffeeDetectionAgent)
        agent._model = FakeModel()
        agent._custom_model_loaded = False
        agent.config = app.DetectionConfig()

        with patch.object(app.cv2, "imread", return_value=object()):
            with self.assertRaisesRegex(RuntimeError, "custom.*model"):
                agent.test_on_image("sample.jpg")

    def test_live_inference_rejects_pretrained_fallback(self):
        agent = app.CoffeeDetectionAgent.__new__(app.CoffeeDetectionAgent)
        agent._model = FakeModel()
        agent._custom_model_loaded = False

        with patch.object(app.cv2, "VideoCapture", return_value=SimpleNamespace(isOpened=lambda: False)):
            with self.assertRaisesRegex(RuntimeError, "custom.*model"):
                agent.start_detection(source=0)

    def test_training_failure_reaches_the_caller(self):
        agent = app.CoffeeDetectionAgent.__new__(app.CoffeeDetectionAgent)
        agent._model = FakeModel()

        with self.assertLogs(app.logger, level="ERROR"):
            with self.assertRaisesRegex(ValueError, "training failed"):
                agent.train_model("data.yaml")


if __name__ == "__main__":
    unittest.main()
