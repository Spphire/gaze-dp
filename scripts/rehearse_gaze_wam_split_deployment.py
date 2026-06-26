import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from diffusion_policy.scripts.rehearse_gaze_wam_split_deployment import main


if __name__ == "__main__":
    main()
