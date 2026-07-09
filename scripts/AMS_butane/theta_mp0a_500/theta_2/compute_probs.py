import os
import json
import re
import numpy as np

def main():
    root_dir = "."
    pattern = re.compile(r"^ams_\d+$")
    values = []

    for root, dirs, files in os.walk(root_dir):
        dirname = os.path.basename(root)
        if pattern.match(dirname):
            file_path = os.path.join(root, "ams_checkpoint.txt")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if "current_p" in data:
                            values.append(float(data["current_p"]))
                except (json.JSONDecodeError, ValueError, KeyError):
                    pass

    if values:
        mean_val = sum(values) / len(values)
        std = np.std(np.array(values))/np.sqrt(len(values))
        with open("results_prob.txt", "w") as out:
            out.write("Values:\n")
            for v in values:
                out.write(f"{v}\n")
            out.write(f"\nMean: {mean_val}, Std : {std}\n")

if __name__ == "__main__":
    main()
