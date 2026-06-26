import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from diffusion_policy.scripts.compare_gaze_wam_ablation_metrics import main


if __name__ == "__main__":
    main()
