import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from diffusion_policy.scripts.convert_hot3d_processed_to_open_zarr import main


if __name__ == "__main__":
    main()
