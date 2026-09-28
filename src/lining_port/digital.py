"""Small, external-observation-only digital model comparison.

This is an API/learning smoke test, not the specification's full M7 budget.
No function in this module accepts FE results, node positions, or physical states.
"""

from __future__ import annotations

import copy
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import threading
import time as wall_time
import traceback
from dataclasses import dataclass

import numpy as np
import psutil
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


SIGNAL_KEYS = ("z", "q", "time", "mask")
METADATA_KEYS = ("y", "split", "scenario_family_id", "layout_id", "sample_id")
MODEL_NAMES = ("ridge", "esn", "gru", "ltc", "cfc")
HIDDEN_SIZE = 8
BATCH_SIZE = 32
PATIENCE = 4
ODE_UNFOLDS = 6


@dataclass(frozen=True)
class ObservableDataset:
    z: np.ndarray
    q: np.ndarray
    time: np.ndarray
    mask: np.ndarray
    y: np.ndarray
    split: np.ndarray
    scenario_family_id: np.ndarray
    layout_id: np.ndarray
    sample_id: np.ndarray

    @property
    def observable(self):
        return self.mask.astype(bool).any(axis=1)


def load_observations(path: str | Path) -> ObservableDataset:
    """Read a strict field whitelist; never deserialize private/unknown arrays."""
    with np.load(path, allow_pickle=False) as archive:
        fields = {key: np.asarray(archive[key]) for key in SIGNAL_KEYS + METADATA_KEYS}
    data = ObservableDataset(**fields)
    n = len(data.y)
    if data.z.shape != (n, 400, 4):
        raise ValueError("Expected genuine sampled z with shape [n,400,4].")
    if data.q.shape != (400,) or data.time.shape != (400,) or data.mask.shape != (n, 4):
        raise ValueError("Expected q/time [400] and mask [n,4].")
    if any(getattr(data, key).shape != (n,) for key in METADATA_KEYS):
        raise ValueError("Each metadata field must have shape [n].")
    if not np.isfinite(data.time).all() or not (np.diff(data.time) > 0).all():
        raise ValueError("Observation times must be finite and strictly increasing.")
    if not np.isfinite(data.q).all() or not np.isin(data.mask, [0, 1]).all():
        raise ValueError("Probe must be finite; mask must be binary.")
    observed = np.broadcast_to(data.mask[:, None, :].astype(bool), data.z.shape)
    if not np.isfinite(data.z[observed]).all():
        raise ValueError("Non-finite observations on a measured channel.")
    if not np.isin(data.y, [0, 1]).all():
        raise ValueError("Only binary support-state labels are accepted.")
    if len(set(data.sample_id.tolist())) != n:
        raise ValueError("sample_id must be unique.")
    expected = {"train", "validation", "test"}
    if set(data.split.tolist()) != expected:
        raise ValueError("Exactly train, validation and test splits are required.")
    families = {s: set(data.scenario_family_id[data.split == s].tolist()) for s in expected}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if families[a] & families[b]:
            raise ValueError(f"Scenario-family leakage between {a} and {b}.")
    # Repeated layouts can be intentional T1 data. Record overlaps; do not call
    # this a new-layout experiment unless the corresponding overlap is empty.
    for split in ("train", "validation"):
        selected = (data.split == split) & data.observable
        if set(data.y[selected].tolist()) != {0, 1}:
            raise ValueError(f"Both classes need observable {split} samples.")
    return data


@dataclass(frozen=True)
class ObservationNormalizer:
    mean: np.ndarray
    scale: np.ndarray
    fit_ids: np.ndarray

    @classmethod
    def fit(cls, data: ObservableDataset):
        train = (data.split == "train") & data.observable
        mean, scale = np.zeros(4), np.ones(4)
        for channel in range(4):
            selected = train & data.mask[:, channel].astype(bool)
            if selected.any():
                values = data.z[selected, :, channel]
                mean[channel] = values.mean()
                standard_deviation = values.std()
                scale[channel] = standard_deviation if standard_deviation > 1e-12 else 1.0
        return cls(mean, scale, data.sample_id[train].copy())

    def transform(self, data: ObservableDataset):
        """All models see [z1..z4, q, dt_seconds, mask1..mask4]."""
        n, steps, _ = data.z.shape
        z = np.where(data.mask[:, None, :].astype(bool), (data.z - self.mean) / self.scale, 0.0)
        dt = np.diff(data.time, prepend=data.time[0] - (data.time[1] - data.time[0]))
        q = np.broadcast_to(data.q[None, :, None], (n, steps, 1))
        elapsed = np.broadcast_to(dt[None, :, None], (n, steps, 1))
        mask = np.broadcast_to(data.mask[:, None, :], (n, steps, 4))
        sequence = np.concatenate((z, q, elapsed, mask), axis=-1).astype(np.float32)
        return sequence, dt.astype(np.float32)


class PeakRSS:
    """Sample current process RSS at 10 ms, including the phase's boundaries."""
    def __enter__(self):
        self.process = psutil.Process(os.getpid())
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop.wait(0.01):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __exit__(self, *_):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


def _json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _metrics(y, scores, layouts, threshold):
    valid = np.isfinite(scores)
    result = {"n_total": int(len(y)), "n_observable": int(valid.sum()),
              "coverage": float(valid.mean()) if len(y) else 0.0,
              "balanced_accuracy": None, "auroc": None, "fpr": None, "tpr": None,
              "layout_mean_balanced_accuracy": None, "by_layout": {}}
    if not valid.any():
        return result
    truth, predicted = y[valid], scores[valid] >= threshold
    if set(truth.tolist()) == {0, 1}:
        result["balanced_accuracy"] = float(balanced_accuracy_score(truth, predicted))
        result["auroc"] = float(roc_auc_score(truth, scores[valid]))
        result["fpr"] = float(predicted[truth == 0].mean())
        result["tpr"] = float(predicted[truth == 1].mean())
    for layout in np.unique(layouts):
        rows = (layouts == layout) & valid
        record = {"n_total": int((layouts == layout).sum()), "n_observable": int(rows.sum()),
                  "balanced_accuracy": None, "auroc": None}
        if set(y[rows].tolist()) == {0, 1}:
            record["balanced_accuracy"] = float(balanced_accuracy_score(y[rows], scores[rows] >= threshold))
            record["auroc"] = float(roc_auc_score(y[rows], scores[rows]))
        result["by_layout"][str(layout)] = record
    layout_ba = [v["balanced_accuracy"] for v in result["by_layout"].values() if v["balanced_accuracy"] is not None]
    if layout_ba:
        result["layout_mean_balanced_accuracy"] = float(np.mean(layout_ba))
    return result


def _threshold(y, scores, layouts):
    """Choose only on validation; deterministic ties favor score threshold 0.5."""
    unique = np.unique(scores)
    candidates = np.r_[np.nextafter(unique[0], -np.inf), (unique[:-1] + unique[1:]) / 2,
                       np.nextafter(unique[-1], np.inf), 0.5]
    quality = []
    for candidate in candidates:
        values = []
        for layout in np.unique(layouts):
            rows = layouts == layout
            if set(y[rows].tolist()) == {0, 1}:
                values.append(balanced_accuracy_score(y[rows], scores[rows] >= candidate))
        value = float(np.mean(values)) if values else float(balanced_accuracy_score(y, scores >= candidate))
        quality.append(value)
    index = max(range(len(candidates)), key=lambda i: (quality[i], -abs(candidates[i] - 0.5), -i))
    return float(candidates[index])


def _esn_features(sequence, weights):
    state = np.zeros((len(sequence), HIDDEN_SIZE), dtype=np.float64)
    summed = np.zeros_like(state)
    for t in range(sequence.shape[1]):
        state = 0.7 * state + 0.3 * np.tanh(sequence[:, t] @ weights["input"] + state @ weights["recurrent"] + weights["bias"])
        summed += state
    return np.concatenate((summed / sequence.shape[1], state), axis=1)


def _make_torch_model(name, seed):
    import torch
    from ncps.torch import CfC, LTC

    torch.manual_seed(seed)

    class SequenceClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if name == "gru":
                self.recurrent = torch.nn.GRU(10, HIDDEN_SIZE, batch_first=True)
            elif name == "ltc":
                self.recurrent = LTC(10, HIDDEN_SIZE, batch_first=True, return_sequences=True,
                                     mixed_memory=False, ode_unfolds=ODE_UNFOLDS)
            elif name == "cfc":
                self.recurrent = CfC(10, HIDDEN_SIZE, batch_first=True, return_sequences=True,
                                     mixed_memory=False, mode="default", backbone_units=8,
                                     backbone_layers=1, backbone_dropout=0.0)
            else:
                raise ValueError(name)
            self.head = torch.nn.Linear(2 * HIDDEN_SIZE, 1)
            self.forward_cell_steps = 0

        def forward(self, sequence, elapsed):
            if name == "gru":
                output, _ = self.recurrent(sequence)
            else:
                # ncps 1.0.1's sequence wrappers squeeze [B,1] elapsed
                # intervals to [B], which does not broadcast against [B,H].
                # Call the author's unchanged cells with [B,1] seconds.
                state = sequence.new_zeros((len(sequence), HIDDEN_SIZE))
                outputs = []
                for t in range(sequence.shape[1]):
                    output_t, state = self.recurrent.rnn_cell(sequence[:, t], state, elapsed[:, t:t + 1])
                    if name == "cfc":
                        output_t = self.recurrent.fc(output_t)
                    outputs.append(output_t)
                output = torch.stack(outputs, dim=1)
            self.forward_cell_steps += len(sequence) * sequence.shape[1]
            features = torch.cat((output.mean(dim=1), output[:, -1]), dim=1)
            return self.head(features).squeeze(-1)

    return SequenceClassifier()


def _torch_scores(model, sequence, dt):
    import torch

    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequence), BATCH_SIZE):
            batch = torch.from_numpy(sequence[start:start + BATCH_SIZE])
            elapsed = torch.from_numpy(np.broadcast_to(dt, (len(batch), len(dt))).copy())
            values.append(torch.sigmoid(model(batch, elapsed)).numpy())
    return np.concatenate(values) if values else np.empty(0)


def _fit_torch(model, sequence, dt, y, train, validation, epochs, seed):
    import torch

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    order_rng = np.random.default_rng(seed)
    best_loss, best_state, best_epoch, stale = np.inf, None, -1, 0
    history = []
    for epoch in range(epochs):
        model.train()
        train_loss, count = 0.0, 0
        for selected in np.array_split(order_rng.permutation(train), max(1, int(np.ceil(len(train) / BATCH_SIZE)))):
            batch = torch.from_numpy(sequence[selected])
            elapsed = torch.from_numpy(np.broadcast_to(dt, (len(batch), len(dt))).copy())
            truth = torch.from_numpy(y[selected].astype(np.float32))
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch, elapsed), truth)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.detach()) * len(batch)
            count += len(batch)
        scores = np.clip(_torch_scores(model, sequence[validation], dt), 1e-7, 1 - 1e-7)
        labels = y[validation]
        validation_loss = float(-np.mean(labels * np.log(scores) + (1 - labels) * np.log1p(-scores)))
        if not np.isfinite(validation_loss):
            raise FloatingPointError("Non-finite validation loss.")
        history.append({"epoch": epoch + 1, "train_loss": train_loss / count, "validation_loss": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_state, best_epoch, stale = validation_loss, copy.deepcopy(model.state_dict()), epoch + 1, 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return history, best_epoch


def run_digital_comparison(data_path, output_dir, seed=20260926, epochs=12):
    """Fit five frozen one-configuration baselines to the same external signals.

    Call only after structural/network checks and the physical baseline pass.
    Existing nonempty output directories are rejected to preserve earlier runs.
    """
    import torch
    from ncps.torch import CfC, LTC

    if not 1 <= epochs <= 12:
        raise ValueError("MVP smoke budget is 1..12 epochs; no implicit scan.")
    target = Path(output_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {target}")
    data = load_observations(data_path)
    target.mkdir(parents=True, exist_ok=True)
    normalizer = ObservationNormalizer.fit(data)
    sequence, dt = normalizer.transform(data)
    train = np.flatnonzero((data.split == "train") & data.observable)
    validation = np.flatnonzero((data.split == "validation") & data.observable)
    available = np.flatnonzero(data.observable)
    np.savez(target / "normalizer.npz", mean=normalizer.mean, scale=normalizer.scale, fit_ids=normalizer.fit_ids)
    layouts = {s: set(data.layout_id[data.split == s].tolist()) for s in ("train", "validation", "test")}
    _json(target / "access_audit.json", {
        "feature_field_whitelist": list(SIGNAL_KEYS), "feature_order": ["z1", "z2", "z3", "z4", "q", "dt_seconds", "mask1", "mask2", "mask3", "mask4"],
        "train_ids": data.sample_id[train].tolist(), "normalizer_fit_ids": normalizer.fit_ids.tolist(),
        "early_stopping_ids": data.sample_id[validation].tolist(), "threshold_fit_ids": data.sample_id[validation].tolist(),
        "test_ids": data.sample_id[data.split == "test"].tolist(), "scenario_family_overlap": [],
        "layout_overlap_train_validation": sorted(layouts["train"] & layouts["validation"]),
        "layout_overlap_train_test": sorted(layouts["train"] & layouts["test"]),
        "unobservable_ids": data.sample_id[~data.observable].tolist(),
        "policy": "All-zero masks abstain; excluded from fitting and ranked metrics, retained in coverage."})
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "cpu": platform.processor(), "device": "cpu", "torch_threads": 1,
                   "versions": {p: importlib.metadata.version(p) for p in ("numpy", "torch", "ncps", "scikit-learn", "psutil")},
                   "api": {"LTC": str(inspect.signature(LTC)), "CfC": str(inspect.signature(CfC)),
                           "LTC_source": inspect.getfile(LTC), "CfC_source": inspect.getfile(CfC),
                           "time_adapter": "Author rnn_cell called with [batch,1] elapsed seconds; avoids installed wrapper squeeze broadcasting defect."}}
    _json(target / "environment.json", environment)
    records, errors = [], []
    try:
        for name in MODEL_NAMES:
            directory = target / name
            directory.mkdir()
            configuration = {"model": name, "seed": seed, "configurations": 1,
                             "scope": "MVP API/learning smoke; not full M7 or equal-cost study", "ridge_alpha": 1.0,
                             "hidden_size": 0 if name == "ridge" else HIDDEN_SIZE, "maximum_epochs": epochs,
                             "patience": PATIENCE, "batch_size": BATCH_SIZE, "learning_rate": 1e-3,
                             "mixed_memory": False, "ltc_ode_unfolds": ODE_UNFOLDS if name == "ltc" else 0,
                             "cfc_mode": "default" if name == "cfc" else None,
                             "cfc_backbone_units": 8 if name == "cfc" else None,
                             "cfc_backbone_layers": 1 if name == "cfc" else None,
                             "cfc_backbone_dropout": 0.0 if name == "cfc" else None,
                             "esn": {"leak": 0.3, "input_scale": 0.2, "spectral_radius": 0.9, "density": 1.0} if name == "esn" else None}
            _json(directory / "config.json", configuration)
            try:
                started = wall_time.perf_counter()
                with PeakRSS() as memory:
                    if name in ("ridge", "esn"):
                        if name == "ridge":
                            features = sequence.reshape(len(sequence), -1).astype(np.float64)
                            fixed_params, state_dim, weights = 0, 0, {}
                        else:
                            rng = np.random.default_rng(seed)
                            recurrent = rng.normal(size=(HIDDEN_SIZE, HIDDEN_SIZE))
                            recurrent *= 0.9 / np.max(np.abs(np.linalg.eigvals(recurrent)))
                            weights = {"input": rng.uniform(-0.2, 0.2, (10, HIDDEN_SIZE)), "recurrent": recurrent,
                                       "bias": rng.uniform(-0.2, 0.2, HIDDEN_SIZE)}
                            features = None
                            fixed_params, state_dim = sum(v.size for v in weights.values()), HIDDEN_SIZE
                            np.savez(directory / "reservoir.npz", **weights)
                        # ESN transforms each partition only when needed. Ridge
                        # flattening is a fixed indexing operation, not fitting.
                        train_features = features[train] if features is not None else _esn_features(sequence[train], weights)
                        val_features = features[validation] if features is not None else _esn_features(sequence[validation], weights)
                        estimator = Ridge(alpha=1.0, solver="svd")
                        estimator.fit(train_features, data.y[train])
                        val_scores = estimator.predict(val_features)
                        np.savez(directory / "readout.npz", coef=estimator.coef_, intercept=estimator.intercept_)
                        trainable_params = estimator.coef_.size + np.asarray(estimator.intercept_).size
                        history, best_epoch, nfev = [], 0, 0
                        buffers = 0
                    else:
                        estimator = _make_torch_model(name, seed)
                        history, best_epoch = _fit_torch(estimator, sequence, dt, data.y, train, validation, epochs, seed)
                        val_scores = _torch_scores(estimator, sequence[validation], dt)
                        torch.save(estimator.state_dict(), directory / "state_dict.pt")
                        trainable_params = sum(p.numel() for p in estimator.parameters() if p.requires_grad)
                        fixed_params = sum(p.numel() for p in estimator.parameters() if not p.requires_grad)
                        buffers = sum(p.numel() for p in estimator.buffers())
                        state_dim = HIDDEN_SIZE
                        nfev = estimator.forward_cell_steps * ODE_UNFOLDS if name == "ltc" else 0
                    threshold = _threshold(data.y[validation], val_scores, data.layout_id[validation])
                    _json(directory / "training_history.json", history)
                    _json(directory / "frozen_predictor.json", {"threshold": threshold, "best_epoch": best_epoch,
                          "trainable_parameters": int(trainable_params), "fixed_parameters": int(fixed_params),
                          "buffer_elements": int(buffers), "state_dimension": state_dim,
                          "saved_at_stage": "frozen_before_test_evaluation"})
                fit_time = wall_time.perf_counter() - started
                fit_peak = memory.peak
                # Every fitting/selection object and threshold is saved before
                # the first test prediction or metric computation.
                started = wall_time.perf_counter()
                with PeakRSS() as inference_memory:
                    scores = np.full(len(data.y), np.nan)
                    if name in ("ridge", "esn"):
                        inference_features = features[available] if features is not None else _esn_features(sequence[available], weights)
                        scores[available] = estimator.predict(inference_features)
                        inference_nfev = 0
                    else:
                        before = estimator.forward_cell_steps
                        scores[available] = _torch_scores(estimator, sequence[available], dt)
                        inference_nfev = (estimator.forward_cell_steps - before) * ODE_UNFOLDS if name == "ltc" else 0
                inference_time = wall_time.perf_counter() - started
                if not np.isfinite(scores[available]).all():
                    raise FloatingPointError("Non-finite classification scores.")
                prediction = np.where(np.isfinite(scores), (scores >= threshold).astype(int), -1)
                np.savez(directory / "predictions.npz", sample_id=data.sample_id, scenario_family_id=data.scenario_family_id,
                         layout_id=data.layout_id, split=data.split, y=data.y, scores=scores,
                         prediction=prediction, observable=data.observable)
                metrics = {split: _metrics(data.y[data.split == split], scores[data.split == split], data.layout_id[data.split == split], threshold)
                           for split in ("train", "validation", "test")}
                record = {"model": name, "status": "completed", "seed": seed,
                          "trainable_parameters": int(trainable_params), "fixed_parameters": int(fixed_params),
                          "buffer_elements": int(buffers), "state_dimension": state_dim, "epochs_run": len(history),
                          "best_epoch": best_epoch, "threshold": threshold,
                          "fit_seconds": fit_time, "inference_seconds_all_partitions": inference_time,
                          "fit_peak_process_rss_bytes": fit_peak, "inference_peak_process_rss_bytes": inference_memory.peak,
                          "rss_definition": "Process RSS sampled every 10 ms; includes imported runtimes and retained process allocations.",
                          "fit_nfev": nfev, "inference_nfev": inference_nfev,
                          "nfev_definition": "LTC: sample-steps times 6 fused implicit ODE unfolds in forward calls, including validation; backward not counted. Other models: 0 (no numerical ODE solver); CfC is a learned closed-form update, not a physical ODE solver.",
                          "metrics": metrics}
                _json(directory / "metrics.json", record)
                records.append(record)
            except Exception as error:
                detail = traceback.format_exc()
                (directory / "failure.log").write_text(detail, encoding="utf-8")
                record = {"model": name, "status": "failed", "error": str(error)}
                records.append(record)
                errors.append(record)
            with (target / "runs.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(records[-1], ensure_ascii=False, allow_nan=False) + "\n")
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic)
    result = {"status": "completed" if not errors else "completed_with_failures", "seed": seed,
              "models": records, "failures": errors, "interpretation": "External digital algorithm smoke comparison only; no physical device implementation or performance conclusion."}
    _json(target / "comparison.json", result)
    return result
