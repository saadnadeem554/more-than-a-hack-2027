import glob, os, shutil, subprocess, sys

src = os.path.dirname(glob.glob("/kaggle/input/**/train_seq.py", recursive=True)[0])
work = "/kaggle/working"
for f in ("train_seq.py", "possession_metrics.py", "features.npz"):
    shutil.copy(os.path.join(src, f), os.path.join(work, f))
os.chdir(work)
os.makedirs("models", exist_ok=True)

def run(*args):
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, "train_seq.py", "--features", "features.npz", *args], check=True)

run("--val", "0", "1", "--iters", "4000", "--out", "models/seq_val01.pt")
run("--val", "2", "3", "--iters", "4000", "--out", "models/seq_val23.pt")
run("--val", "4", "5", "6", "--iters", "4000", "--out", "models/seq_val456.pt")
for seed in (0, 1, 2):
    run("--val", "-1", "--iters", "4000", "--seed", str(seed), "--out", "models/seq_all_s%d.pt" % seed)
