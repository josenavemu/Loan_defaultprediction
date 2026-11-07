"""
LAB 6 RUNNER
"""

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)


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
        "LAB 6 - MODEL REGISTRY "
        "AND LIFECYCLE"
    )

    print("=" * 60)

    # Step 1
    run_script(
        "train_registry.py"
    )

    # Step 2
    run_script(
        "automate_lifecycle.py"
    )

    # Step 3
    run_script(
        "generate_registry_report.py"
    )

    print("\n" + "=" * 60)

    print(
        "LAB 6 COMPLETED SUCCESSFULLY"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()