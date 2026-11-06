from library_robot.sim.mobile_model import generate_mobile_models


def main():
    paths = generate_mobile_models()
    print("\nGenerated mobile Panda MJCF models:")
    for base_type, path in paths.items():
        print(f"  {base_type}: {path}")
    print("\nSelected base_type in configs/robot.yaml controls load_mobile_robot_model().")
    print("Review all config values marked 'needs tuning' before interpreting physical performance.")


if __name__ == "__main__":
    main()
