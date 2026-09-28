"""Synthetic API/isolation tests; these data are not finite-element evidence."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from lining_port.digital import ObservationNormalizer, _make_torch_model, load_observations, run_digital_comparison


def fixture():
    rng = np.random.default_rng(19)
    n = 12
    return {"z": rng.normal(size=(n, 400, 4)), "q": np.tile([-1., 1.], 200),
            "time": 2.005 + np.arange(400) * .005, "mask": np.ones((n, 4), dtype=int),
            "y": np.tile([0, 1], 6), "split": np.repeat(["train", "validation", "test"], 4),
            "scenario_family_id": np.repeat(np.arange(6), 2).astype(str),
            "layout_id": np.repeat(["layout_0", "layout_1", "layout_2"], 4),
            "sample_id": np.asarray([f"sample_{i:03d}" for i in range(n)])}


class DigitalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "observations.npz"

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, values):
        np.savez(self.path, **values)
        return load_observations(self.path)

    def test_hidden_field_poisoning_has_no_effect(self):
        values = fixture()
        clean = self.save(values)
        normalizer = ObservationNormalizer.fit(clean)
        clean_input, _ = normalizer.transform(clean)
        # Loading this object array would fail with allow_pickle=False. It is
        # deliberately never deserialized by the signal whitelist.
        values.update(private_fe=np.array([{"class": 999}], dtype=object),
                      private_network=np.full((12, 400, 64), np.nan), node_eps=np.full((12, 64), 1e99))
        poisoned = self.save(values)
        np.testing.assert_array_equal(clean_input, normalizer.transform(poisoned)[0])
        self.assertFalse(hasattr(poisoned, "private_fe"))

    def test_normalizer_ignores_validation_and_test(self):
        values = fixture()
        original = ObservationNormalizer.fit(self.save(values))
        values["z"][4:] += 1e12
        changed = ObservationNormalizer.fit(self.save(values))
        np.testing.assert_array_equal(original.mean, changed.mean)
        np.testing.assert_array_equal(original.scale, changed.scale)
        np.testing.assert_array_equal(original.fit_ids, values["sample_id"][:4])

    def test_family_leakage_is_rejected(self):
        values = fixture()
        values["scenario_family_id"][8] = values["scenario_family_id"][0]
        with self.assertRaisesRegex(ValueError, "Scenario-family leakage"):
            self.save(values)

    def test_missing_channels_are_masked_and_all_missing_abstains(self):
        values = fixture()
        values["mask"][8] = 0
        values["z"][8] = np.nan
        data = self.save(values)
        inputs, dt = ObservationNormalizer.fit(data).transform(data)
        self.assertFalse(data.observable[8])
        self.assertTrue(np.isfinite(inputs).all())
        np.testing.assert_array_equal(inputs[8, :, :4], 0.)
        np.testing.assert_allclose(dt, .005)

    def test_ncps_api_timespans_and_seed_reproducibility(self):
        import torch

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            inputs = torch.zeros((2, 7, 10))
            elapsed = torch.full((2, 7), .005)
            for name in ("gru", "ltc", "cfc"):
                first = _make_torch_model(name, 123)
                second = _make_torch_model(name, 123)
                a, b = first(inputs, elapsed), second(inputs, elapsed)
                self.assertEqual(a.shape, (2,))
                self.assertTrue(torch.isfinite(a).all())
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                if name != "gru":
                    # Author's full sequence API works for a single row,
                    # where squeeze yields a scalar. Verify the minibatch
                    # adapter against that independent public entry point.
                    rows = []
                    for row in range(2):
                        output, _ = first.recurrent(inputs[row:row + 1], timespans=elapsed[row:row + 1])
                        features = torch.cat((output.mean(1), output[:, -1]), dim=1)
                        rows.append(first.head(features).squeeze(-1))
                    torch.testing.assert_close(a, torch.cat(rows), rtol=1e-6, atol=1e-7)
                a.sum().backward()
                self.assertTrue(all(p.grad is not None for p in first.parameters() if p.requires_grad))
        finally:
            torch.set_num_threads(old_threads)

    def test_one_epoch_training_reproduces_all_five_scores(self):
        self.save(fixture())
        first_path = Path(self.temporary.name) / "first"
        second_path = Path(self.temporary.name) / "second"
        first = run_digital_comparison(self.path, first_path, seed=73, epochs=1)
        second = run_digital_comparison(self.path, second_path, seed=73, epochs=1)
        self.assertEqual(first["status"], "completed")
        self.assertEqual(second["status"], "completed")
        for model in ("ridge", "esn", "gru", "ltc", "cfc"):
            with np.load(first_path / model / "predictions.npz") as a, np.load(second_path / model / "predictions.npz") as b:
                np.testing.assert_array_equal(a["scores"], b["scores"])
                np.testing.assert_array_equal(a["prediction"], b["prediction"])


if __name__ == "__main__":
    unittest.main()
