import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from diffusion_policy.scripts.inspect_open_video_gaze_metadata import main


if __name__ == "__main__":
    main()
