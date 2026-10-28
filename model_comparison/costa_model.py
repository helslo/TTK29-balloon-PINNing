"""CoSTA-style BOLD residual correction on top of the Balloon model."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class CoSTAConfig:
    """Training settings for the learned physical-model correction."""

    epochs: int = 1000
    learning_rate: float = 1e-3
    hidden_width: int = 32
    hidden_layers: int = 2
    weight_decay: float = 1e-4
    correction_penalty: float = 1e-2
    smoothness_penalty: float = 1e-2
    max_correction_scale: float = 3.0
    patience: int = 100
    seed: int = 17


class CoSTAResidual(nn.Module):
    """Bounded BOLD correction conditioned on the physical trajectory."""

    def __init__(
        self,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        residual_scale: float,
        time_origin: float,
        duration: float,
        config: CoSTAConfig,
    ) -> None:
        super().__init__()
        self.register_buffer("feature_mean", torch.tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("feature_scale", torch.tensor(feature_scale, dtype=torch.float32))
        self.residual_scale = residual_scale
        self.time_origin = time_origin
        self.duration = duration
        self.max_correction_scale = config.max_correction_scale

        layers = [nn.Linear(len(feature_mean), config.hidden_width), nn.Tanh()]
        for _ in range(config.hidden_layers - 1):
            layers.extend([nn.Linear(config.hidden_width, config.hidden_width), nn.Tanh()])
        output_layer = nn.Linear(config.hidden_width, 1)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_scale
        elapsed = ((features[:, 0:1] - self.time_origin) / self.duration).clamp(0.0, 1.0)
        return (
            elapsed
            * self.residual_scale
            * self.max_correction_scale
            * torch.tanh(self.network(normalized))
        )


def _features(
    times: np.ndarray,
    input_values: np.ndarray,
    states: np.ndarray,
    physical_bold: np.ndarray,
) -> np.ndarray:
    return np.column_stack((times, input_values, states.T, physical_bold)).astype(np.float32)


def train_costa(
    times: np.ndarray,
    input_values: np.ndarray,
    observed_bold: np.ndarray,
    physical_states: np.ndarray,
    physical_bold: np.ndarray,
    config: CoSTAConfig | None = None,
) -> Tuple[CoSTAResidual, Dict[str, list]]:
    """Learn a regularized residual correction from one observed BOLD trace."""
    if config is None:
        config = CoSTAConfig()
    if config.epochs <= 0 or config.patience <= 0:
        raise ValueError("epochs and patience must be positive")
    if config.hidden_width <= 0 or config.hidden_layers <= 0:
        raise ValueError("hidden_width and hidden_layers must be positive")
    if config.max_correction_scale <= 0:
        raise ValueError("max_correction_scale must be positive")

    times = np.asarray(times, dtype=np.float32)
    input_values = np.asarray(input_values, dtype=np.float32)
    observed_bold = np.asarray(observed_bold, dtype=np.float32)
    physical_states = np.asarray(physical_states, dtype=np.float32)
    physical_bold = np.asarray(physical_bold, dtype=np.float32)
    sample_count = len(times)
    if times.ndim != 1 or sample_count < 10 or np.any(np.diff(times) <= 0):
        raise ValueError("times must contain at least ten strictly increasing samples")
    if input_values.shape != (sample_count,) or observed_bold.shape != (sample_count,):
        raise ValueError("input_values and observed_bold must match times")
    if physical_states.shape != (4, sample_count) or physical_bold.shape != (sample_count,):
        raise ValueError("physical_states and physical_bold must match the Balloon output shapes")
    if not all(
        np.all(np.isfinite(values))
        for values in (times, input_values, observed_bold, physical_states, physical_bold)
    ):
        raise ValueError("training arrays must contain only finite values")

    duration = float(times[-1] - times[0])
    if duration <= 0:
        raise ValueError("times must span a positive duration")
    features = _features(times, input_values, physical_states, physical_bold)
    residual = observed_bold - physical_bold
    validation_mask = np.arange(sample_count) % 5 == 0
    validation_mask[0] = False
    if np.count_nonzero(validation_mask) < 2:
        raise ValueError("at least two validation samples are required")
    training_mask = ~validation_mask
    feature_mean = features[training_mask].mean(axis=0)
    feature_scale = features[training_mask].std(axis=0)
    feature_scale[feature_scale < 1e-6] = 1.0
    residual_scale = max(float(np.std(residual[training_mask])), 1e-6)

    torch.manual_seed(config.seed)
    feature_tensor = torch.tensor(features)
    residual_tensor = torch.tensor(residual[:, None])
    training_tensor = torch.tensor(training_mask)
    validation_tensor = torch.tensor(validation_mask)
    model = CoSTAResidual(
        feature_mean, feature_scale, residual_scale, float(times[0]), duration, config
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=max(10, config.patience // 4), min_lr=1e-5
    )
    history: Dict[str, list] = {"train": [], "validation": [], "correction_rms": []}
    best_validation = float("inf")
    best_state = deepcopy(model.state_dict())
    stale_epochs = 0

    for _ in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        correction = model(feature_tensor)
        normalized_error = (correction - residual_tensor) / residual_scale
        correction_size = correction / residual_scale
        second_difference = correction[2:] - 2.0 * correction[1:-1] + correction[:-2]
        loss = (
            normalized_error[training_tensor].square().mean()
            + config.correction_penalty * correction_size.square().mean()
            + config.smoothness_penalty * (second_difference / residual_scale).square().mean()
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_correction = model(feature_tensor)
            validation_loss = (
                (validation_correction[validation_tensor] - residual_tensor[validation_tensor])
                .div(residual_scale)
                .square()
                .mean()
            )
            validation_value = float(validation_loss)
            history["train"].append(float(loss.detach()))
            history["validation"].append(validation_value)
            history["correction_rms"].append(
                float(torch.sqrt(torch.mean(validation_correction.square())))
            )
        scheduler.step(validation_value)
        if validation_value < best_validation - 1e-8:
            best_validation = validation_value
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    model.load_state_dict(best_state)
    return model, history


def predict_costa(
    model: CoSTAResidual,
    times: np.ndarray,
    input_values: np.ndarray,
    physical_states: np.ndarray,
    physical_bold: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the unchanged physical states and corrected BOLD prediction."""
    features = _features(
        np.asarray(times),
        np.asarray(input_values),
        np.asarray(physical_states),
        np.asarray(physical_bold),
    )
    with torch.no_grad():
        correction = model(torch.tensor(features)).squeeze(1).numpy()
    return np.asarray(physical_states), np.asarray(physical_bold) + correction