import numpy as np
import random
import subprocess

def generate(file_name, n, func):
    with open(file_name, "w") as f:
        for i in range(n):
            f.write(chr(func(i)))

    for m in ["zstd", "lz4", "zlib"]:
        subprocess.run(["./cpu_root_comp", "-f", file_name, "-t", m, "-o", f"{file_name}.root.{m}"])

if __name__ == "__main__":
    n = 100000000
    generate("uniform_random_100m", n, lambda x: random.randrange(32, 126))
    generate("gauss_random_100m", n, lambda x: min(0, max(255, int(random.gauss(128, 128)))))
    generate("ones_100", 100, lambda x: 49)
