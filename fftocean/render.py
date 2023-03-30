from typing import Iterable, Any
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.colors as colors
from time import perf_counter

import numpy as np


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def render(frame_generator: Iterable, normalize: bool = False, interval=30) -> Any:
    """
    Renders frames from provided generator at 30 fps.

    Arguments:
        - generator: generates frames to be rendered.

    Returns:
        Animation.
    """
    # plt.axis("off")

    fig = plt.figure()
    frame, _ = next(frame_generator)

    fps = 0
    fig.suptitle(f"FFT Ocean (fps: {fps})")

    if normalize:
        frame -= np.min(frame)
        frame /= np.max(frame)

    cmap = truncate_colormap(plt.get_cmap("ocean"), 0.2)

    im = plt.imshow(frame, cmap=cmap, vmin=-1, vmax=1)

    # animator simply renders next frame
    def animator(_):
        frame, delta_time = next(frame_generator)

        if normalize:
            frame -= np.min(frame)
            frame /= np.max(frame)
        im.set_array(frame)
        fps = 1 / delta_time
        fig.suptitle(f"FFT Ocean (fps: {fps:.2f}, frametime: {delta_time*1000:.2f}ms)")
        return (im,)

    anim = animation.FuncAnimation(fig, animator, interval=interval, blit=True)
    plt.show()

    return anim


def render3D(
    renderer: Iterable, render_resolution: int = 50, interval: int = 16
) -> Any:
    """
    Renders a 3D surface.

    Arguments:
        - renderer: Generator that returns 3D surface vertice coordinates
                    as a tuple (x, y, z).
        - render_resolution: Resolution of rendered surface (render_resolution**2 vertices).
        - interval: minimum render frame time. Default 16, i.e., 60 fps.
    """

    delta_time = 1

    def animator(_, plot):
        nonlocal delta_time
        prev_time = perf_counter()

        surface = next(renderer)

        fps = 1 / delta_time
        fig.suptitle(f"FFT Ocean (fps: {fps:.2f}, frametime: {delta_time*1000:.2f}ms)")
        fig.tight_layout(pad=0)

        plot[0].remove()
        plot[0] = ax.plot_surface(
            *surface,
            cmap=cmap,
            vmin=-0.15,
            vmax=0.15,
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            rcount=render_resolution,
            ccount=render_resolution,
            shade=False,
        )
        delta_time = perf_counter() - prev_time

    fig = plt.figure()

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(projection="3d")
    surface = next(renderer)

    fps = 0

    fig.suptitle(f"FFT Ocean (fps: {fps:.2f}, frametime: {0*1000:.2f}ms)")
    cmap = truncate_colormap(plt.get_cmap("ocean"), 0.2)

    plot = [
        ax.plot_surface(
            *surface,
            cmap=cmap,
            vmin=-0.1,
            vmax=0.1,
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            rcount=render_resolution,
            ccount=render_resolution,
        )
    ]
    ax.set_zlim(-2, 2)
    anim = animation.FuncAnimation(
        fig, animator, fargs=(plot,), interval=interval, cache_frame_data=False
    )

    ax.axis("off")

    plt.show()

    return anim
