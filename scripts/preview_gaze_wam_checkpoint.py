import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion_policy.scripts.preview_gaze_wam_checkpoint import main


if __name__ == "__main__":
    main()
