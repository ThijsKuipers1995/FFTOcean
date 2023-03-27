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

        self.register_buffer("wind_direction", wind_direction)
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

        self.register_buffer("magnitude_sqrt", torch.sqrt(self.magnitude_map * self.G))

        dx = torch.stack((self.zero_real, -self.k[..., 0] / self.magnitude_map), dim=-1)
        dy = torch.stack((self.zero_real, -self.k[..., 1] / self.magnitude_map), dim=-1)

        self.register_buffer("dx", torch.view_as_complex(dx))
        self.register_buffer("dy", torch.view_as_complex(dy))

    def _generate_h0k_h0minusk(self) -> None:
        L = self.wind_speed**2 / self.G

        k_n = self.k / self.magnitude_map[..., None]
        w_n = self.wind_direction / torch.linalg.norm(self.wind_direction)
        mag_sq = self.magnitude_map**2

        Ph_term = (
            self.amplitude
            * torch.exp(-1 / (mag_sq * L * L))
            / (mag_sq * mag_sq)
            * torch.exp(mag_sq * (self.ocean_size / 2000) ** 2)
            / sqrt(2)
        )

        _h0k = torch.clip(
            Ph_term * ((k_n @ w_n) ** self.swell),
            -4000,
            4000,
        )
        _h0minusk = torch.clip(
            Ph_term * ((-k_n @ w_n) ** self.swell),
            -4000,
            4000,
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

        hkt_dy = (self.h0k * exp_iwt) + (self.h0minusk_conj * exp_iwt_inv)
        hkt_dx = self.dx * hkt_dy
        hkt_dz = self.dy * hkt_dy

        components = torch.stack((hkt_dx, hkt_dy, hkt_dz))

        return components * (self.n / self.ocean_size) ** 2

    def forward(self, t: float) -> Tensor:
        return self._get_components(t)
