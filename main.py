from fftocean.ocean import Ocean


def main():
    num_cascades = 3
    cascade_sizes = (1000, 200, 50)
    cascade_time_multipliers = (1.3, 0.8, 0.4)
    cascade_strengths = (1.5, 1.1, 0.7)

    wind_speed = 4.5

    ocean = Ocean(
        num_cascades=num_cascades,
        cascade_size=cascade_sizes,
        cascade_strength=cascade_strengths,
        wind_speed=wind_speed,
        cascade_time_multiplier=cascade_time_multipliers,
    )

    ocean.run(render_resolution=32)


if __name__ == "__main__":
    main()
