import shutil
from pathlib import Path

n_ams = 70
offset = 0

root = Path(__file__).resolve().parent
reweight_src = root / "reweight.py"
job_topaze_src = root / "job_reweight_topaze"

for i in range(offset, offset + n_ams):
    target_dir = root / f"results/ams_{i}"
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(reweight_src, target_dir / "reweight.py")
    shutil.copy2(job_topaze_src, target_dir / "job_reweight_topaze")
