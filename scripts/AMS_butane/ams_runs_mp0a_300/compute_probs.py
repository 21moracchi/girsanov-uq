import json
import os
import re
import numpy as np


def main():
    pattern = re.compile(r"^ams_\d+$")
    values = []

    for root, dirs, files in os.walk("."):
        dirs.sort()
        dirname = os.path.basename(root)
        if pattern.match(dirname):
            file_path = os.path.join(root, "ams_checkpoint.txt")
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        current_p = data.get("current_p")
                        if current_p is not None:
                            values.append(float(current_p))
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

    if not values:
        raise RuntimeError("No valid current_p values found in ams_* checkpoints.")

    np.savetxt("results_prob.txt", np.asarray(values, dtype=float))

if __name__ == "__main__":
    main()
