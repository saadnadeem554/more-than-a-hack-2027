"""Kaggle GPU job: 7-fold leave-one-match-out sweep over model configs, then full-data seeds.

Input: the private dataset built by train_build_dataset.py from external (CC-BY) matches only.
Output: models/<config>_all_s<seed>.pt and results.txt (one RESULT line per fold x cost).
"""
import glob, os, shutil, subprocess, sys

src = os.path.dirname(glob.glob("/kaggle/input/**/train_seq.py", recursive=True)[0])
work = "/kaggle/working"
for f in ("train_seq.py", "possession_metrics.py", "features.npz"):
    shutil.copy(os.path.join(src, f), os.path.join(work, f))
os.chdir(work)
os.makedirs("models", exist_ok=True)

CONFIGS = {
    "base": ["--hidden", "64", "--layers", "2", "--chunk", "400", "--iters", "4000"],
    "wide": ["--hidden", "128", "--layers", "2", "--chunk", "400", "--iters", "4000"],
    "long": ["--hidden", "64", "--layers", "2", "--chunk", "1000", "--iters", "4000"],
    "deeplong": ["--hidden", "128", "--layers", "3", "--chunk", "1000", "--iters", "6000"],
}

with open("results.txt", "w") as log:
    def run(name, extra):
        p = subprocess.run([sys.executable, "train_seq.py", "--features", "features.npz", *extra],
                           capture_output=True, text=True)
        print(p.stdout[-1500:], p.stderr[-1500:], flush=True)
        for line in p.stdout.splitlines():
            if line.startswith("RESULT"):
                log.write("{} {}\n".format(name, line))
        log.flush()
        if p.returncode:
            raise SystemExit(p.returncode)

    for name, cfg in CONFIGS.items():
        for fold in range(7):
            run(name, cfg + ["--val", str(fold), "--out", "models/{}_val{}.pt".format(name, fold)])
        for seed in (0, 1, 2):
            run(name, cfg + ["--val", "-1", "--seed", str(seed),
                             "--out", "models/{}_all_s{}.pt".format(name, seed)])
for f in glob.glob("models/*_val*.pt"):
    os.remove(f)
