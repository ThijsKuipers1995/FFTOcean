from fftocean.ocean_cascade import OceanCascade
from fftocean.render import render3D

import torch
from torch import Tensor

from time import perf_counter


class Ocean:
    def __init__(
        self,
        num_cascades: int = 2,
        resolution: int = 256,
        cascade_size: int | tuple[int, ...] = (1000, 300),
        cascade_strength: float | tuple[float, ...] = (1.2, 1.5),
        wind_speed: float | tuple[float, ...] = 4.0,
        wind_angle: float | tuple[float, ...] = 45.0,
        swell: float | tuple[float, ...] = 3.0,
        choppiness: float | tuple[float, ...] = 3.0,
        simulation_speed: float = 1.0,
        cascade_time_multiplier: float | tuple[float, ...] = 1.0,
        angle_unit: str = "deg",
    ) -> None:
        """
        Simulates an ocean surface through multiple ocean cascades.

        If providing tuple arguments, number of elements in the tuple must
        at least match the number of cascades.

        Arguments:
            - num_cascades: Number of cascades that will be simulated.
            - resolution: resolution of the simulation.
            - cascade_size: Size of simulated cascade.
            - cascade_strength: Determines how much the cascade contributes to
                                the ocean surface.
            - wind_speed: Speed of simulated wind.
            - wind_angle: Angle on unit circle that wind is coming from.
            - swell: Determines how much waves align perpendicular to
                                the wind direction.
            - choppiness: Determines wave choppiness.
            - simulation_speed: Speed of simulation.
            - cascade_time_multiplier: Alters speed of simulation per cascade,
                                       a lower multiplier simulates a slower, i.e.,
                                       larger body of water.
            - angle_unit: Unit in which angle is provided. Defaults to 'deg' for
                          degrees. For radians, use 'rad'.
        """

        self.simulation_speed = simulation_speed
        self.cascade_strength = cascade_strength
        self.cascade_time_multiplier = cascade_time_multiplier

        self.cascades = [
            OceanCascade(
                resolution,
                cascade_size[i]
                if isinstance(cascade_size, (str, list, tuple))
                else cascade_size,
                choppiness[i]
                if isinstance(choppiness, (str, list, tuple))
                else choppiness,
                wind_speed[i]
                if isinstance(wind_speed, (str, list, tuple))
                else wind_speed,
                swell[i] if isinstance(swell, (str, list, tuple)) else swell,
                wind_angle[i]
                if isinstance(wind_angle, (str, list, tuple))
                else wind_angle,
                angle_in=angle_unit,
            )
            for i in range(num_cascades)
        ]

        # initialize basegrid
        x = torch.linspace(-10, 10, resolution)
        self.grid = torch.stack(
            (
                *torch.meshgrid((x, x), indexing="xy"),
                torch.zeros(resolution, resolution),
            )
        )

        # buffers for x, y, and z vertex coordinates
        self.surface_buffer = torch.empty(3, resolution, resolution)

    def _reset_vertex_buffers(self):
        self.surface_buffer[...] = 0

    def calculate_ocean_surface(self, time: float) -> Tensor:
        """
        Calculates ocean surface at time step 'time'.

        Surface is saved in self.surface_buffer. Note that
        the function does not return a copy of the buffer.
        """
        self._reset_vertex_buffers()

        self.surface_buffer += self.grid

        for i, cascade in enumerate(self.cascades):
            cascade_strength = (
                self.cascade_strength[i]
                if isinstance(self.cascade_strength, (str, list, tuple))
                else self.cascade_strength
            )
            cascade_time_multiplier = (
                self.cascade_time_multiplier[i]
                if isinstance(self.cascade_time_multiplier, (str, list, tuple))
                else self.cascade_time_multiplier
            )

            self.surface_buffer += (
                cascade(cascade_time_multiplier * time) * cascade_strength
            )

        return self.surface_buffer

    def surfaceGenerator(self):
        """
        Generator that yields surface at each consecutive
        time step. Simulation time is not tied to framerate.
        """
        time, delta_time = 0, 0

        while True:
            current_time = perf_counter()
            time += self.simulation_speed * delta_time

            self.calculate_ocean_surface(time)

            yield self.surface_buffer

            delta_time = perf_counter() - current_time

    def run(self, render_resolution: int = 50):
        """
        Renders the ocean surface simulation.

        Arguments:
            - render_resolution: Resolution at whih the surface mesh will be
              rendered. NOTE: Significantly impacts performance!
        """
        render3D(self.surfaceGenerator(), render_resolution)
