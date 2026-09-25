"""Physics-informed neural network for the Balloon-Windkessel model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import torch
from torch import nn

from balloon_model import BalloonParams


@dataclass
class PINNConfig:
    """Training settings for one PINN fit."""

    epochs: int = 1000
    learning_rate: float = 1e-3
    physics_weight: float = 1.0
    data_weight: float = 10.0
    hidden_width: int = 32
    hidden_layers: int = 3
    seed: int = 7


class BalloonPINN(nn.Module):
    """Neural state trajectory constrained by the Balloon ODE."""

    def __init__(self, params: BalloonParams, hidden_width: int = 32, hidden_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(1, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_width, hidden_width), nn.Tanh()])
        layers.append(nn.Linear(hidden_width, 4))
        self.network = nn.Sequential(*layers)
        self.params = params

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Return states ``[s, f, v, q]`` with the physical initial state enforced."""
        raw = self.network(time)
        elapsed = time[:, 0:1]
        return torch.cat(
            [
                elapsed * raw[:, 0:1],
                1.0 + 0.95 * torch.tanh(elapsed * raw[:, 1:2]),
                1.0 + 0.95 * torch.tanh(elapsed * raw[:, 2:3]),
                1.0 + 0.95 * torch.tanh(elapsed * raw[:, 3:4]),
            ],
            dim=1,
        )

    def residuals(self, time: torch.Tensor, input_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ODE residuals and the predicted BOLD signal."""
        states = self(time)
        derivatives = torch.cat(
            [torch.autograd.grad(states[:, index].sum(), time, create_graph=True)[0][:, 0:1]
             for index in range(4)],
            dim=1,
        )
        s, flow, volume, deoxy = states.unbind(dim=1)
        params = self.params
        extraction = 1.0 - (1.0 - params.E0) ** (1.0 / flow)
        flow_term = volume ** (1.0 / params.alpha)
        expected = torch.stack(
            [
                params.eps * input_values - params.kappa * s - params.gamma * (flow - 1.0),
                s,
                (flow - flow_term) / params.tau,
                (flow * extraction / params.E0 - (flow_term / volume) * deoxy) / params.tau,
            ],
            dim=1,
        )
        residual = derivatives - expected
        k1 = 4.3 * params.nu0 * params.E0 * params.TE
        k2 = params.epsilon_r * params.r0 * params.E0 * params.TE
        k3 = 1.0 - params.epsilon_r
        bold = params.V0 * (k1 * (1.0 - deoxy) + k2 * (1.0 - deoxy / volume) + k3 * (1.0 - volume))
        return residual, bold


def train_pinn(
    times: np.ndarray,
    input_values: np.ndarray,
    observed_bold: np.ndarray,
    params: BalloonParams | None = None,
    config: PINNConfig | None = None,
) -> Tuple[BalloonPINN, Dict[str, list]]:
    """Fit a PINN to one time series and return the model plus loss history."""
    if params is None:
        params = BalloonParams()
    if config is None:
        config = PINNConfig()
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")

    times = np.asarray(times, dtype=np.float32)
    input_values = np.asarray(input_values, dtype=np.float32)
    observed_bold = np.asarray(observed_bold, dtype=np.float32)
    if not (times.ndim == input_values.ndim == observed_bold.ndim == 1):
        raise ValueError("times, input_values, and observed_bold must be one-dimensional")
    if not (len(times) == len(input_values) == len(observed_bold)):
        raise ValueError("times, input_values, and observed_bold must have equal length")
    if np.any(np.diff(times) <= 0) or times[0] < 0:
        raise ValueError("times must be strictly increasing and start at or after zero")

    torch.manual_seed(config.seed)
    time_tensor = torch.tensor((times - times[0])[:, None], requires_grad=True)
    input_tensor = torch.tensor(input_values)
    target_tensor = torch.tensor(observed_bold[:, None])
    model = BalloonPINN(params, config.hidden_width, config.hidden_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: Dict[str, list] = {"total": [], "data": [], "physics": []}

    for _ in range(config.epochs):
        optimizer.zero_grad()
        residual, predicted_bold = model.residuals(time_tensor, input_tensor)
        data_loss = torch.mean((predicted_bold[:, None] - target_tensor) ** 2)
        physics_loss = torch.mean(residual ** 2)
        total_loss = config.data_weight * data_loss + config.physics_weight * physics_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        history["total"].append(float(total_loss.detach()))
        history["data"].append(float(data_loss.detach()))
        history["physics"].append(float(physics_loss.detach()))
    return model, history


def predict_pinn(model: BalloonPINN, times: np.ndarray, input_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate a trained PINN and return states and BOLD values."""
    normalized_times = np.asarray(times, dtype=np.float32)
    time_tensor = torch.tensor((normalized_times - normalized_times[0])[:, None], requires_grad=True)
    with torch.no_grad():
        states = model(time_tensor).numpy()
    params = model.params
    bold = params.V0 * (
        4.3 * params.nu0 * params.E0 * params.TE * (1.0 - states[:, 3])
        + params.epsilon_r * params.r0 * params.E0 * params.TE * (1.0 - states[:, 3] / states[:, 2])
        + (1.0 - params.epsilon_r) * (1.0 - states[:, 2])
    )
    return states, bold


if __name__ == "__main__":
    from compare_models import main

    main(["--mode", "pinn"])