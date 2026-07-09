import os, time 
import re
from pathlib import Path
import numpy as np 

ROOT = Path(__file__).resolve().parents[1]
COMMON_BLOCK_PATTERN = re.compile(
    r"(?ms)^([ \t]*)# >>> COMMON:\s*([^\n]+?)\s*$\n^\1# <<< COMMON\s*$"
)


def expand_common_blocks(text: str) -> str:
    def _replace(match):
        indent = match.group(1)
        rel_path = match.group(2).strip()
        snippet_path = ROOT / rel_path
        snippet_text = snippet_path.read_text()
        lines = snippet_text.splitlines()
        return "\n".join(f"{indent}{line}" if line else "" for line in lines)

    return COMMON_BLOCK_PATTERN.sub(_replace, text)

n_walkers = 10
dyn_seeds = np.random.randint(10**6, size=n_walkers)
sampler_seeds = np.random.randint(10**6, size=n_walkers) 
for i in range(n_walkers):
    os.system('mkdir FV_replica_' + str(i))
    os.system('cp job* FV_replica_' + str(i))
    os.system('cp run_ini_conds.py FV_replica_' + str(i) + '/')
    f = open("run_ini_conds.py", "rt")
    f_run = open("FV_replica_" + str(i) + "/run_ini_conds.py", "wt")
    rendered = expand_common_blocks(f.read())
    for line in rendered.splitlines(keepends=True):
        if "walker_index" in line:
            line = line.replace("walker_index=0", "walker_index=" + str(i))
        
        if "sampler_seed" in line:
            line = line.replace("sampler_seed = None", "sampler_seed = " + str(sampler_seeds[i]))
        if "dyn_seed" in line:
            line = line.replace("dyn_seed = None", "dyn_seed = " + str(dyn_seeds[i]))
        
        f_run.write(line)
    f.close()
    f_run.close()
   
