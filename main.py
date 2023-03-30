from fftocean.ocean import Ocean


def main():
    num_cascades = 3
    resolution = 64
    cascade_sizes = (1000, 200, 50)
    cascade_time_multipliers = (1.2, 0.5, 0.5)
    cascade_strengths = (2, 1.5, 0.4)
    wind_angles = (36, 40, 51)
    swells = (2, 3, 4)

    wind_speed = 8

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
