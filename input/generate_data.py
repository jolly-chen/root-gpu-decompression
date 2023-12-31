import numpy as np
import random
import subprocess

def generate(file_name, data):
    with open(f"{file_name}.in", "wb") as f:
        f.write(data.view(f"S{len(data)}")[0])

    # for m in ["zstd", "lz4", "zlib"]:
    for m in ["zstd"]:
        subprocess.run(
            [
                "./cpu_root_comp",
                "-f",
                f"{file_name}.in",
                "-t",
                m,
                "-o",
                f"{file_name}.root.{m}",
            ]
        )


def generate_floats(file_name, data):
    with open(f"{file_name}.in", "wb") as f:
        data.tofile(f)

    # for m in ["zstd", "lz4", "zlib"]:
    for m in ["zstd"]:
        subprocess.run(
            [
                "./cpu_root_comp",
                "-f",
                f"{file_name}.in",
                "-t",
                m,
                "-o",
                f"{file_name}.root.{m}",
                "-p"
            ]
        )


if __name__ == "__main__":
    k = 1
    np.random.seed(123)

    for size in [
        1000 * s for s in np.logspace(2, 7, base=2, num=6).astype(int)
    ]:  # 4KB to 128 KB
        # Maximum compression ratio
        generate(f"max_compression.{size}", np.ones(size, dtype=np.uint8) * 49)

        # Low compression ratio
        generate(
            f"low_compression.{size}",
            np.around(
                np.minimum(255, np.maximum(-1, np.random.normal(128, 32, size=size)))
            ).astype(np.uint8),
        )

        # Medium compression ratio
        generate(
            f"mid_compression.{size}",
            np.around(
                np.minimum(255, np.maximum(-1, np.random.normal(128, 1, size=size)))
            ).astype(np.uint8),
        )

    size = 16000
    generate_floats(
        f"packed_floats.{size * 4}",
        np.random.normal(0, 1000, size=size).astype(np.float32),
    )
