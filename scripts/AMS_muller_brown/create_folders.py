#!/usr/bin/env python3
"""Create sensitivity-analysis folders from template_ams.

For each parameter (d_0 ... d_4), this script creates:
- one parent folder named after the parameter (e.g. d_0)
- one subfolder per relative factor (e.g. 1.1d_0_ref, 0.7d_0_ref)

Each variation folder is a full copy of template_ams where only the
corresponding parameter value in common/model_block.py is modified.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


# Relative factors applied to the reference value from template_ams.
# Example: 1.1 means 1.1 * d_i_ref.
# You can change these lists freely.
PARAM_FACTORS = {
    "d_0": [0.9,0.95,1.05,1.1],
    "d_1": [0.9,0.95,1.05,1.1],
    "d_2": [0.9,0.95,1.05,1.1],
    "d_3": [0.9,0.95,1.05,1.1],
    "d_4": [0.9,0.95,1.05,1.1],
}

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = SCRIPT_DIR / "template_ams"
MODEL_BLOCK_RELATIVE_PATH = Path("common") / "model_block.py"


def extract_base_values(model_block_text: str) -> dict[str, float]:
    """Read d_0..d_4 base values from model_block.py."""
    values: dict[str, float] = {}
    for i in range(5):
        name = f"d_{i}"
        match = re.search(
            rf"^\s*{name}\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$",
            model_block_text,
            flags=re.MULTILINE,
        )
        if not match:
            raise ValueError(f"Parameter {name} not found in model_block.py")
        values[name] = float(match.group(1))
    return values


def format_number(value: float) -> str:
    """Format numbers cleanly for folder names and file content."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def apply_parameter_value(model_block_text: str, parameter: str, new_value: float) -> str:
    """Replace one parameter line (e.g. d_0 = ...) with the new value."""
    pattern = rf"^(\s*{parameter}\s*=\s*)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(\s*)$"
    replacement = rf"\g<1>{format_number(new_value)}\g<3>"
    new_text, n_replaced = re.subn(pattern, replacement, model_block_text, flags=re.MULTILINE)

    if n_replaced != 1:
        raise ValueError(f"Expected one replacement for {parameter}, found {n_replaced}")
    return new_text


def create_sensitivity_folders() -> None:
    if not TEMPLATE_DIR.exists():
        raise FileNotFoundError(f"Template folder not found: {TEMPLATE_DIR}")

    model_block_template = TEMPLATE_DIR / MODEL_BLOCK_RELATIVE_PATH
    if not model_block_template.exists():
        raise FileNotFoundError(f"Missing file in template: {model_block_template}")

    template_text = model_block_template.read_text(encoding="utf-8")
    base_values = extract_base_values(template_text)

    print(f"Template: {TEMPLATE_DIR}")
    print("Base values:", {k: format_number(v) for k, v in base_values.items()})

    for parameter, factors in PARAM_FACTORS.items():
        parameter_folder = SCRIPT_DIR / parameter
        parameter_folder.mkdir(exist_ok=True)

        for factor in factors:
            subfolder_name = f"{format_number(float(factor))}{parameter}_ref"
            destination = parameter_folder / subfolder_name

            if destination.exists():
                action = "update"
            else:
                shutil.copytree(TEMPLATE_DIR, destination)
                action = "create"

            model_block_copy = destination / MODEL_BLOCK_RELATIVE_PATH
            updated_value = base_values[parameter] * float(factor)
            # Rebuild from the clean template each run so partially failed runs are recoverable.
            updated_text = apply_parameter_value(template_text, parameter, updated_value)
            model_block_copy.write_text(updated_text, encoding="utf-8")

            print(
                f"[ok:{action}] {destination} -> {parameter} = {format_number(updated_value)} "
                f"(base {format_number(base_values[parameter])}, factor {format_number(float(factor))})"
            )


if __name__ == "__main__":
    create_sensitivity_folders()
