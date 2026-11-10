"""Regression checks using generated fixtures; no private dataset or model access."""

import copy
import ctypes
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

from src.config import load_config, resolve_split_path
from src.data.loader import list_csv_files
from src.data.replay import dataset_manifest, rebuild_training_dataframe
from src.features.feature_list import FEATURE_LIST
from src.features.preprocess import fit_preprocessor, transform
from src.main_export_rf_c import generate_header, generate_preprocessor, generate_source
from src.model.evaluate import evaluate_predictions
from src.uart.evaluate import uart_infer_split
from src.uart.protocol import encode_row, parse_response

ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_response_formats_and_confidence_scaling(self):
        self.assertEqual(parse_response("7,2,850000"), (7, 2, 0.85))
        self.assertEqual(parse_response("2,0.85"), (None, 2, 0.85))
        self.assertEqual(
            parse_response('{"row_id":7,"pred_id":2,"confidence":0.85}'), (7, 2, 0.85)
        )
        for value in ("", "1,nan", "1,-0.2", "1,2000000", "bad"):
            with self.assertRaises(ValueError):
                parse_response(value)

    def test_wire_encoding_preserves_float32(self):
        values = np.asarray([0.123456789, 1e-12, 123456789], dtype=np.float32)
        decoded = np.asarray(
            encode_row(4, values).decode().strip().split(",")[1:], dtype=np.float32
        )
        np.testing.assert_array_equal(decoded, values)
        with self.assertRaises(ValueError):
            encode_row(0, [float("inf")])

    def test_uart_skips_keep_exact_probability_row_indices(self):
        class FakeSerial:
            def __init__(self):
                self.responses = iter([b"0,0,1\n", b"", b"2,1,900000\n", b"99,0,1\n"])

            def reset_input_buffer(self):
                pass

            def reset_output_buffer(self):
                pass

            def write(self, data):
                pass

            def readline(self):
                return next(self.responses)

        truth, predictions, accepted = uart_infer_split(
            FakeSerial(),
            np.zeros((4, 15)),
            np.array(["A", "A", "B", "B"]),
            ["A", "B"],
            "test",
            4,
            0,
            0,
        )
        self.assertEqual(accepted, [0, 2])
        self.assertEqual(truth.tolist(), ["A", "B"])
        self.assertEqual(predictions.tolist(), ["A", "B"])


class DataTests(unittest.TestCase):
    def test_csv_discovery_does_not_duplicate_case_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fixture.csv").write_text("x\n1\n")
            (root / "upper.CSV").write_text("x\n2\n")
            self.assertEqual(len(list_csv_files(directory)), 2)

    def test_replay_uses_run_snapshot_and_rejects_changed_data(self):
        cfg, _ = load_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data/raw"
            raw.mkdir(parents=True)
            frame = pd.DataFrame(
                [{**{f: 1.0 for f in FEATURE_LIST}, "Label": "BENIGN"}] * 10
            )
            frame.to_csv(raw / "fixture.csv", index=False)
            meta = root / "artifacts/metadata/test"
            meta.mkdir(parents=True)
            snapshot = copy.deepcopy(cfg)
            snapshot["data"]["sample_frac"] = 0.5
            (meta / "config.json").write_text(json.dumps(snapshot))
            (meta / "splits.json").write_text('{"test":[0]}')
            (meta / "dataset_manifest.json").write_text(
                json.dumps(dataset_manifest(raw))
            )
            self.assertEqual(len(rebuild_training_dataframe(cfg, root, "test")), 5)
            self.assertEqual(
                resolve_split_path(cfg, root, "test"), meta / "splits.json"
            )
            with (raw / "fixture.csv").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                rebuild_training_dataframe(cfg, root, "test")


class PublicationTests(unittest.TestCase):
    def test_source_data_module_is_allowed_but_local_inputs_are_not(self):
        from tools.check_publication import local_only

        self.assertFalse(local_only("src/data/loader.py"))
        for path in [
            "data/raw/input.csv",
            "data/processed/splits.json",
            "notebooks/train.ipynb",
            "artifacts/models/run/model.joblib",
            "capture.pcap",
            "Wireshark",
        ]:
            self.assertTrue(local_only(path), path)


class FlowTests(unittest.TestCase):
    def test_shared_flow_table_tracks_both_directions_and_expiry(self):
        from scapy.all import IP, TCP, Raw

        from src.flows import FlowTable

        table = FlowTable(timeout_seconds=2.0, max_duration_seconds=10.0)
        forward = (
            IP(src="192.0.2.1", dst="192.0.2.2")
            / TCP(sport=1000, dport=80, flags="S")
            / Raw(b"a")
        )
        forward.time = 1.0
        reverse = (
            IP(src="192.0.2.2", dst="192.0.2.1")
            / TCP(sport=80, dport=1000, flags="A")
            / Raw(b"b")
        )
        reverse.time = 1.5
        table.update_with_packet(forward)
        flow, expired = table.update_with_packet(reverse)
        self.assertFalse(expired)
        self.assertEqual(flow.total_packets(), 2)
        features = flow.to_feature_dict()
        self.assertEqual(features["Tot Fwd Pkts"], 1)
        self.assertEqual(features["Tot Bwd Pkts"], 1)
        self.assertEqual(features["Down/Up Ratio"], 1)
        self.assertEqual(features["Flow Duration"], 500000)
        self.assertEqual(len(table.pop_expired(4.0)), 1)
        self.assertFalse(table.table)


class DashboardTests(unittest.TestCase):
    def test_event_ingestion_filters_and_stats(self):
        from src.dashboard.app import create_app
        from src.dashboard.storage import EventStore

        with patch("src.dashboard.app.store", EventStore()):
            client = create_app().test_client()
            self.assertEqual(client.get("/").status_code, 200)
            response = client.post(
                "/api/events",
                json=[
                    {"predicted_class": "Normal", "confidence": 0.8},
                    {"predicted_class": "DoS", "confidence": 0.9},
                ],
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(client.get("/api/stats").json["total"], 2)
            events = client.get("/api/events?class=DoS&min_confidence=0.85").json[
                "events"
            ]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["predicted_class"], "DoS")


class ReportTests(unittest.TestCase):
    def test_device_report_identifies_pc_probability_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            classes = ["A", "B", "C"]
            evaluate_predictions(
                np.array(["A", "B", "C"]),
                np.array(["A", "B", "A"]),
                np.eye(3),
                classes,
                directory,
                prediction_source="stm32_uart",
                probability_source="pc_joblib_reference",
                extra_metadata={"accepted_rows": [0, 2, 5]},
            )
            report = json.loads((Path(directory) / "test_report.json").read_text())
            self.assertEqual(report["prediction_source"], "stm32_uart")
            self.assertEqual(report["probability_source"], "pc_joblib_reference")
            self.assertEqual(report["evaluation"]["accepted_rows"], [0, 2, 5])


@unittest.skipUnless(shutil.which("gcc"), "GCC required for host C export verification")
class CExportTests(unittest.TestCase):
    def test_c_predictions_probabilities_and_threshold_boundaries(self):
        rng = np.random.default_rng(42)
        X = rng.random((150, 15), dtype=np.float32)
        y = np.array(["A", "B", "C"])[np.arange(150) % 3]
        model = RandomForestClassifier(
            n_estimators=5, max_depth=4, random_state=42
        ).fit(X, y)
        probes = [X]
        for tree in model.estimators_:
            for node in range(tree.tree_.node_count):
                feature = tree.tree_.feature[node]
                if feature >= 0:
                    center = np.float32(tree.tree_.threshold[node])
                    rows = np.full((3, 15), 0.5, dtype=np.float32)
                    rows[:, feature] = [
                        np.nextafter(center, np.float32(-np.inf)),
                        center,
                        np.nextafter(center, np.float32(np.inf)),
                    ]
                    probes.append(rows)
        probes = np.concatenate(probes)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rf_model.h").write_text(
                generate_header("rf_model", list(model.classes_), FEATURE_LIST, 5)
            )
            (root / "rf_model.c").write_text(
                generate_source(model, "rf_model", list(model.classes_))
            )
            frame = pd.DataFrame(X, columns=FEATURE_LIST)
            preprocessor = fit_preprocessor(frame, FEATURE_LIST, 0.0)
            (root / "rf_preprocessor.h").write_text(generate_preprocessor(preprocessor))
            (root / "preprocess.c").write_text(
                '#include "rf_preprocessor.h"\nvoid preprocess(const float *i, float *o) {rf_preprocess(i,o);}\n'
            )
            library = root / ("model.dll" if sys.platform == "win32" else "model.so")
            subprocess.run(
                [
                    shutil.which("gcc"),
                    "-shared",
                    "-O2",
                    "-std=c99",
                    "-fPIC",
                    str(root / "rf_model.c"),
                    str(root / "preprocess.c"),
                    "-lm",
                    "-o",
                    str(library),
                ],
                check=True,
                capture_output=True,
            )
            (root / "main.h").write_text(
                """#include <stdint.h>
+typedef struct { int placeholder; } UART_HandleTypeDef;
+#define HAL_OK 0
+#define HAL_MAX_DELAY 0xffffffffU
+int HAL_UART_Receive(UART_HandleTypeDef *, uint8_t *, uint16_t, uint32_t);
+int HAL_UART_Transmit(UART_HandleTypeDef *, uint8_t *, uint16_t, uint32_t);
+""".replace("\n+", "\n")
            )
            # The integration header expects five classes; use a separate five-class fixture header.
            adapter_dir = root / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "rf_model.h").write_text(
                generate_header("rf_model", ["A", "B", "C", "D", "E"], FEATURE_LIST, 5)
            )
            for device_mode in (0, 1):
                subprocess.run(
                    [
                        shutil.which("gcc"),
                        "-std=c99",
                        "-Wall",
                        "-Werror",
                        "-c",
                        str(ROOT / "firmware/app_nids.c"),
                        "-I",
                        str(adapter_dir),
                        "-I",
                        str(root),
                        "-I",
                        str(ROOT / "firmware"),
                        f"-DNIDS_DEVICE_PREPROCESS={device_mode}",
                        "-o",
                        str(root / f"adapter-{device_mode}.o"),
                    ],
                    check=True,
                    capture_output=True,
                )
            lib = ctypes.CDLL(str(library))
            pointer = ctypes.POINTER(ctypes.c_float)
            lib.rf_model_predict.argtypes = [pointer]
            lib.rf_model_predict.restype = ctypes.c_int32
            lib.rf_model_predict_proba.argtypes = [pointer, pointer]
            lib.preprocess.argtypes = [pointer, pointer]
            c_probs, c_predictions, c_scaled = [], [], []
            for row in probes:
                output = np.zeros(3, dtype=np.float32)
                lib.rf_model_predict_proba(
                    row.ctypes.data_as(pointer), output.ctypes.data_as(pointer)
                )
                c_probs.append(output.copy())
                c_predictions.append(lib.rf_model_predict(row.ctypes.data_as(pointer)))
            for row in X:
                output = np.zeros(15, dtype=np.float32)
                lib.preprocess(
                    row.ctypes.data_as(pointer), output.ctypes.data_as(pointer)
                )
                c_scaled.append(output)
            np.testing.assert_allclose(c_probs, model.predict_proba(probes), atol=2e-7)
            np.testing.assert_array_equal(
                np.asarray(model.classes_)[c_predictions], model.predict(probes)
            )
            np.testing.assert_allclose(
                c_scaled, transform(frame, preprocessor), atol=2e-7
            )
            if sys.platform == "win32":
                import _ctypes

                _ctypes.FreeLibrary(lib._handle)


class TrainingSmokeTests(unittest.TestCase):
    def test_small_training_export_and_test_split(self):
        cfg, _ = load_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg["data"]["raw_dir"] = str(root / "raw")
            cfg["data"]["processed_dir"] = str(root / "processed")
            cfg["train"].update(
                rf_n_estimators=3, rf_max_depth=3, rf_min_samples_leaf=1, rf_n_jobs=1
            )
            for key in cfg["artifacts"]:
                cfg["artifacts"][key] = str(root / key)
            (root / "raw").mkdir()
            rows = []
            for class_id, label in enumerate(
                ["BENIGN", "DoS", "FTP-BruteForce", "Bot", "SQL Injection"]
            ):
                for i in range(20):
                    rows.append(
                        {
                            **{f: float(class_id * 50 + i + 1) for f in FEATURE_LIST},
                            "Label": label,
                        }
                    )
            pd.DataFrame(rows).to_csv(root / "raw/fixture.csv", index=False)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(cfg))
            commands = [
                ["src.main_train", "--run_name", "smoke"],
                [
                    "src.main_export_rf_c",
                    "--run_name",
                    "smoke",
                    "--output_dir",
                    str(root / "export"),
                ],
                [
                    "src.main_export_test_split",
                    "--run_name",
                    "smoke",
                    "--preprocessed",
                    "--cubemx",
                ],
            ]
            for module, *args in commands:
                result = subprocess.run(
                    [sys.executable, "-m", module, "--config", str(config_path), *args],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((root / "metadata_dir/smoke/splits.json").is_file())
            self.assertTrue((root / "export/rf_preprocessor.h").is_file())
            self.assertTrue(
                (root / "processed/test_split_smoke_cubemx_preprocessed.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
