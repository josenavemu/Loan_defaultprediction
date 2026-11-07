"""
LAB 5 RUNNER
"""

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_script(script_name):

    print("\n" + "=" * 60)

    print(
        f"Running: {script_name}"
    )

    print("=" * 60)

    result = subprocess.run(

        [
            sys.executable,

            str(
                PROJECT_ROOT /
                "src" /
                script_name
            )
        ],

        cwd=PROJECT_ROOT,

        check=False
    )

    if result.returncode != 0:

        raise SystemExit(
            f"{script_name} failed."
        )


def main():

    print("=" * 60)

    print(
        "LAB 5 - DATA VALIDATION "
        "AND REPRODUCIBLE PIPELINE"
    )

    print("=" * 60)

    # Step 1
    run_script(
        "validate_data.py"
    )

    # Step 2
    run_script(
        "preprocess_pipeline.py"
    )

    # Step 3
    run_script(
        "validate_outputs.py"
    )

    print("\n" + "=" * 60)

    print(
        "LAB 5 COMPLETED SUCCESSFULLY"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()