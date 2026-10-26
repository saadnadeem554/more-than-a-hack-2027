"""Kaggle GPU job: 7-fold leave-one-match-out CV per feature set, then full-data seeds.

Input: the private dataset built by train_build_dataset.py from external (CC-BY) matches only.
Output: models/<name>_all_s<seed>.pt and results.txt (one RESULT line per fold x decoding cost).
"""
import glob, os, shutil, subprocess, sys

src = os.path.dirname(glob.glob("/kaggle/input/**/train_seq.py", recursive=True)[0])
work = "/kaggle/working"
for f in os.listdir(src):
    if f.endswith((".py", ".npz")):
        shutil.copy(os.path.join(src, f), os.path.join(work, f))
os.chdir(work)
os.makedirs("models", exist_ok=True)

BASE = ["--hidden", "64", "--layers", "2", "--chunk", "400", "--iters", "4000"]
DATA = ["--features", "features_v4.npz", "--exclude", "7", "8"]  # 7, 8 = Metrica (dropped)
RUNS = {
    "dfl": DATA + BASE,
    "dfl8k": DATA + BASE[:-1] + ["8000"],
}
FINAL = {"dfl": (0, 1, 2, 3, 4)}

with open("results.txt", "w") as log:
    def run(name, extra):
        p = subprocess.run([sys.executable, "train_seq.py", *extra], capture_output=True, text=True)
        print(name, p.stdout[-800:], p.stderr[-800:], flush=True)
        for line in p.stdout.splitlines():
            if line.startswith("RESULT"):
                log.write("{} {}\n".format(name, line))
        log.flush()
        if p.returncode:
            raise SystemExit(p.returncode)

    for name, cfg in RUNS.items():
        for fold in range(7):
            run(name, cfg + ["--val", str(fold), "--out", "models/{}_val{}.pt".format(name, fold)])
        for seed in FINAL.get(name, ()):
            run(name, cfg + ["--val", "-1", "--seed", str(seed),
                             "--out", "models/{}_all_s{}.pt".format(name, seed)])
for f in glob.glob("models/*_val*.pt"):
    os.remove(f)
