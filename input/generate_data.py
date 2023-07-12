import numpy as np
import random
import subprocess

def generate(file_name, data):
    with open(f"{file_name}.in", "wb") as f:
        f.write(data.view(f"S{len(data)}")[0])

    # for m in ["zstd", "lz4", "zlib"]:
    for m in ["zstd"]:
        subprocess.run(["./cpu_root_comp", "-f", f"{file_name}.in", "-t", m, "-o", f"{file_name}.root.{m}"])

if __name__ == "__main__":
    k = 5
    s = 50000000
    for n in np.logspace(1, k, base=10, num=k).astype(int):
        # generate(f"uniform_random_50m.{n}", np.around(np.random.uniform(0, k, s)).astype(np.uint8))
        generate(f"gauss_random_50m.{n}", np.around(np.minimum(255, np.maximum(1, np.random.normal(128, n, size=s)))).astype(np.uint8))
