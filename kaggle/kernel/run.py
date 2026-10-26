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
V6 = ["--features", "features_v6_lab.npz"]   # Y/YF = 3 s merge; Y_m2/YF_m2, Y_m4/YF_m4 extra
V6T = ["--features", "features_v6t_lab.npz"]  # same degradation + touch features
RUNS = {
    "base": V6 + BASE,
    "tw3": V6 + BASE + ["--trans-weight", "3"],
    "touch": V6T + BASE,
    "train2": V6 + BASE + ["--ykey", "Y_m2"],
    "train4": V6 + BASE + ["--ykey", "Y_m4"],
    "eval2": V6 + BASE + ["--yfkey", "YF_m2"],
    "eval4": V6 + BASE + ["--yfkey", "YF_m4"],
}
FINAL = {}
DUMP = False

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
            run(name, cfg + ["--val", str(fold), "--out", "models/{}_val{}.pt".format(name, fold)]
                + (["--dump", "dump_{}_val{}.npz".format(name, fold)] if DUMP else []))
        for seed in FINAL.get(name, ()):
            run(name, cfg + ["--val", "-1", "--seed", str(seed),
                             "--out", "models/{}_all_s{}.pt".format(name, seed)])
for f in glob.glob("models/*_val*.pt"):
    os.remove(f)
