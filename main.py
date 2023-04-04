from fftocean.ocean import Ocean


def main():
    num_cascades = 2
    resolution = 64
    cascade_sizes = (1000, 27)
    cascade_time_multipliers = (1, 0.7)
    cascade_strengths = (1, 0.6)
    wind_angles = (45, 45)
    swells = (4, 2)

    wind_speed = (21, 7)

    ocean = Ocean(
        resolution=resolution,
        wind_angle=wind_angles,
        swell=swells,
        num_cascades=num_cascades,
        cascade_size=cascade_sizes,
        cascade_strength=cascade_strengths,
        wind_speed=wind_speed,
        cascade_time_multiplier=cascade_time_multipliers,
    )

    ocean.run(render_resolution=64)


if __name__ == "__main__":
    main()
