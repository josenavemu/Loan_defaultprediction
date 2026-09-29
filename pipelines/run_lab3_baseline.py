import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


def run_command(script_name):
    script_path = os.path.join(
        PROJECT_ROOT,
        "src",
        script_name
    )

    print("\n" + "=" * 60)
    print(f"Running: {script_name}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        print(f"\nERROR: {script_name} failed.")
        sys.exit(result.returncode)


def main():
    print("=" * 60)
    print("LAB 3 - BASELINE ML PIPELINE")
    print("=" * 60)

    # Step 1: Preprocessing
    run_command("preprocess.py")

    # Step 2: Model training
    run_command("train.py")

    # Step 3: Model evaluation
    run_command("evaluate.py")

    print("\n" + "=" * 60)
    print("LAB 3 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated folders/files:")
    print("  data/processed/")
    print("  models/")
    print("  outputs/")


if __name__ == "__main__":
    main()
