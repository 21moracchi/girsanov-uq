import sys
import os
import numpy as np
import ase.units as units
from ase.build import molecule
from ase.io import read
from ase.constraints import FixCom
from girsanov_uq.descriptors.mace import MACELinear
from girsanov_uq.post_processing.reweighting import Reweightor


def resolve_ams_id() -> int:
    if len(sys.argv) > 1:
        return int(sys.argv[1])

    cwd_name = os.path.basename(os.getcwd())
    if cwd_name.startswith("ams_"):
        suffix = cwd_name.split("_", 1)[1]
        if suffix.isdigit():
            return int(suffix)

    raise ValueError("ams_id not provided and current folder is not ams_#")


ams_id = resolve_ams_id()
input_directory = (
    f"../../../ams_{ams_id}"
)
output_directory = (
    "."
)

model_path = "../../../../models/mace_mp0a_ft.model"

os.makedirs(output_directory, exist_ok=True)
files = sorted([f for f in os.listdir(input_directory) if f.endswith(".traj")])
device = "cuda"
calc = MACELinear(model_paths=model_path, device=device, enable_cueq=False, default_dtype="float32")

temperature_K = 300
atoms = molecule('trans-butane')

atoms.calc = calc
atoms.set_constraint(FixCom())
masses = atoms.get_masses()

reweightor = Reweightor(
    calc,
    mode="linear",
    friction=0.01 / units.fs,
    temperature_K=temperature_K,
    masses=masses,
)

for file_name in files:
    file_path = os.path.join(input_directory, file_name)
    traj = read(file_path, index=":")
    print(file_path, flush=True)
    score, fim = reweightor.reweight_one_trajectory_linear(traj, as_list=False)

    base_name = file_name.replace(".traj", "")
    np.save(os.path.join(output_directory, f"{base_name}_score.npy"), score)
    np.save(os.path.join(output_directory, f"{base_name}_fim.npy"), fim)
