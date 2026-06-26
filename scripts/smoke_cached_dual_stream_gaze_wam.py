import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from diffusion_policy.scripts.smoke_cached_dual_stream_gaze_wam import main


if __name__ == "__main__":
    raise SystemExit(main())
