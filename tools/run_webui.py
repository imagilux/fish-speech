import os
from argparse import ArgumentParser

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tools.webui import build_app
from tools.webui.inference import get_inference_wrapper


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.environ.get("API_URL", "http://localhost:8080"),
        help="Base URL of the Fish Speech API server",
    )
    parser.add_argument("--theme", type=str, default="light")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    inference_fct = get_inference_wrapper(args.api_url)

    app = build_app(inference_fct, args.theme, api_url=args.api_url)
    app.launch()
