import torch
from torch import Tensor
import torch.nn as nn
from math import pi, sin, cos

import fftocean.spectrum as spectrum


class OceanCascade(nn.Module):

    _component_idx = {"x": 0, "y": 1, "z": 2}

    def __init__(
        self,
        resolution: int,
        size: float,
        amplitude: float,
        wind_speed: float,
        swell: float,
        wind_angle: float,
        angle_in: str = "deg",
        components: str = "xyz",
    ):
        super().__init__()

        self.eval()

        if angle_in.lower() in ["d", "deg", "degree"]:
            wind_angle *= pi / 180

        wind_direction = torch.Tensor([cos(wind_angle), sin(wind_angle)])

        self.spectrum = spectrum.PhillipsSpectrum(
            resolution, size, amplitude, wind_direction, wind_speed, swell
        )

        components_to_process = set()
        for component in components.lower():
            if component not in "xyz":
                raise ValueError(f"Component must be in 'xyz', found '{component}'.")
            components_to_process.add(self._component_idx[component])

        self.components = list(components_to_process)
        self.components.sort()

        x_range = torch.arange(resolution)
        x = torch.stack(torch.meshgrid(x_range, x_range, indexing="xy"))

        # IFFT index permutations
        self.register_buffer("perms", torch.where(torch.sum(x, dim=0) % 2 == 1, 1, -1))

    def _inversion_pass(self, components):
        return self.perms * components

    def forward(self, t: float) -> Tensor:
        hkt_components = self.spectrum(t)[self.components]
        components = torch.fft.ifft2(hkt_components).real
        return self._inversion_pass(components)
