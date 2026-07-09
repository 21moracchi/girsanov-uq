import os
import shutil
from pathlib import Path

def process_trajectories(directory_path, N):
    base_dir = Path(directory_path)
    
    used_files = list(base_dir.glob("*_used.extxyz"))
    for f in used_files:
        new_name = f.name.replace("_used", "")
        f.rename(base_dir / new_name)

    source_files = sorted(
        [f for f in base_dir.glob("*.extxyz") if f.stem.isdigit()],
        key=lambda x: int(x.stem)
    )
    
    M = len(source_files)
    if M == 0:
        return

    for j in range(N):
        subdir = base_dir / str(j)
        subdir.mkdir(exist_ok=True)
        
        for source_path in source_files:
            target_path = subdir / source_path.name
            
            if not target_path.exists():
                shutil.copy(source_path, target_path)

if __name__ == "__main__":
    import sys
    ini_conds_dir = sys.argv[1]
    if len(sys.argv) > 1:
        process_trajectories(sys.argv[1], int(sys.argv[2]))