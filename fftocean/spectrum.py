import torch
import torch.nn as nn
from torch import Tensor

from math import pi, sqrt


class PhillipsSpectrum(nn.Module):
    def __init__(
        self,
        n: int,
        ocean_size: int,
        amplitude: float,
        wind_direction: Tensor,
        wind_speed: float,
        swell: int = 4,
        G: float = 9.81,
    ) -> None:
        super().__init__()

        self.eval()

        self.n = n
        self.ocean_size = ocean_size
        self.amplitude = amplitude
        self.wind_speed = wind_speed
        self.swell = swell
        self.G = G

        self.register_buffer(
            "wind_direction", wind_direction / torch.linalg.norm(wind_direction)
        )
        self.register_buffer("noise_map", torch.randn(2, n, n, 2))

        self.register_buffer("zero_real", torch.zeros(n, n))

        self._init_waves()
        self._generate_h0k_h0minusk()

    def _init_waves(self) -> None:
        _k = 2 * pi * (torch.arange(self.n) - (self.n // 2)) / self.ocean_size
        self.register_buffer(
            "k", torch.stack(torch.meshgrid(_k, _k, indexing="xy"), dim=-1)
        )
        self.register_buffer("magnitude_map", torch.linalg.norm(self.k, dim=-1))
        self.magnitude_map[self.magnitude_map < 0.00001] = 0.00001

        self.register_buffer("magnitude_sq", self.magnitude_map**2)
        self.register_buffer("magnitude_sqsq", self.magnitude_sq**2)
        self.register_buffer("magnitude_sqrt", torch.sqrt(self.magnitude_map * self.G))

        self.register_buffer("k_normalized", self.k / self.magnitude_map[..., None])

        dx = torch.stack((self.zero_real, -self.k[..., 0] / self.magnitude_map), dim=-1)
        dz = torch.stack((self.zero_real, -self.k[..., 1] / self.magnitude_map), dim=-1)

        self.register_buffer("dx", torch.view_as_complex(dx))
        self.register_buffer("dz", torch.view_as_complex(dz))

    def _generate_h0k_h0minusk(self) -> None:
        L = self.wind_speed**2 / self.G

        ph_term = torch.sqrt(self.amplitude / self.magnitude_sqsq) * torch.exp(
            (-1 / (self.magnitude_sq * L**2))
            + (-self.magnitude_sq * (self.ocean_size / 2000) ** 2) / 1.414213
        )

        _h0k = torch.clip(
            ph_term * ((self.k_normalized @ self.wind_direction).abs() ** self.swell),
            0,
            100000,
        )
        _h0minusk = torch.clip(
            ph_term * ((-self.k_normalized @ self.wind_direction).abs() ** self.swell),
            0,
            100000,
        )

        h0k = torch.view_as_complex(_h0k[..., None] * self.noise_map[0])
        h0minusk_conj = torch.view_as_complex(
            _h0minusk[..., None] * self.noise_map[1]
        ).conj()

        self.register_buffer("h0k", h0k)
        self.register_buffer("h0minusk_conj", h0minusk_conj)

    def _get_components(self, t: float) -> Tensor:
        wt = t * self.magnitude_sqrt

        cos_w_t = torch.cos(wt)
        sin_w_t = torch.sin(wt)

        exp_iwt = torch.view_as_complex(torch.stack((cos_w_t, sin_w_t), dim=-1))
        exp_iwt_inv = torch.view_as_complex(torch.stack((cos_w_t, -sin_w_t), dim=-1))

        hkt_dz = (self.h0k * exp_iwt) + (self.h0minusk_conj * exp_iwt_inv)
        hkt_dx = self.dx * hkt_dz
        hkt_dy = self.dz * hkt_dz

        components = torch.stack((hkt_dx, hkt_dy, hkt_dz))

        return components * (self.n / self.ocean_size) ** 2

    def forward(self, t: float) -> Tensor:
        return self._get_components(t)
