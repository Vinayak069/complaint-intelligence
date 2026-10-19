# Runs inside a Kaggle notebook session: fetches this repo's code and data, then runs one job.
# The GitHub workflow fills in JOB, REPO and REF before pushing it to Kaggle.
import os
import subprocess
import sys

JOB, REPO, REF = "__JOB__", "__REPO__", "__REF__"
OUT = "/kaggle/working/out"


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


os.makedirs(OUT, exist_ok=True)
sh(f"git clone -q --depth 1 -b {REF} https://github.com/{REPO} /kaggle/tmp/repo")
sh(f"git clone -q --depth 1 -b data https://github.com/{REPO} /kaggle/tmp/repo/data")
sh("pip install -q -e /kaggle/tmp/repo[nlp] 2>&1 | tail -2")
os.chdir("/kaggle/tmp/repo")
code = subprocess.run([sys.executable, "-u", f"kaggle/{JOB}.py", "--data", "data", "--out", OUT]).returncode
sys.exit(code)
