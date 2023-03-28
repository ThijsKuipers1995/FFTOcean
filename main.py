from fftocean.ocean import Ocean


def main():
    num_cascades = 3
    cascade_sizes = (1000, 200, 50)
    cascade_time_multipliers = (1.2, 0.7, 0.4)
    cascade_strengths = (3, 2, 0.7)

    wind_speed = 13

    ocean = Ocean(
        wind_angle=45,
        swell=5,
        num_cascades=num_cascades,
        cascade_size=cascade_sizes,
        cascade_strength=cascade_strengths,
        wind_speed=wind_speed,
        cascade_time_multiplier=cascade_time_multipliers,
    )

    ocean.run(render_resolution=40)


if __name__ == "__main__":
    main()
