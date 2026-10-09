import json
import os
import re

import numpy as np


def main():
    pattern = re.compile(r"^ams_\d+$")
    values = []

    for root, dirs, _files in os.walk("."):
        dirs.sort()
        if pattern.match(os.path.basename(root)):
            checkpoint = os.path.join(root, "ams_checkpoint.txt")
            if not os.path.exists(checkpoint):
                continue
            try:
                with open(checkpoint, "r", encoding="utf-8") as stream:
                    current_p = json.load(stream).get("current_p")
                if current_p is not None:
                    values.append(float(current_p))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

    if not values:
        raise RuntimeError("No valid current_p values found in ams_* checkpoints.")
    np.savetxt("results_prob.txt", np.asarray(values, dtype=float))


if __name__ == "__main__":
    main()
