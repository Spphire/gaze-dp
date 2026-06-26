import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from diffusion_policy.scripts.review_gaze_wam_data_onboarding import main


if __name__ == "__main__":
    main()
