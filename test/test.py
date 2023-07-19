import os
import glob
import subprocess
import unittest
import numpy as np
from parameterized import parameterized

os.chdir("../input")
test_files = glob.glob("*.root.*")
os.chdir("..")

class TestSequence(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 1
        self.w = 1

        self.input_folder = "input/"
        self.output_folder = "output/"
        self.bin="./cpu_root_decomp"

    @parameterized.expand([*test_files])
    def test_single_cpu(self, f):
        method = f.split(".")[-1]
        input_file = f"{self.input_folder}/{f}"
        output_file = f"{self.output_folder}/{f}.out"
        expected = f"{self.input_folder}/{f.removesuffix(f'.root.{method}')}.in"
        CMD = f"./cpu_root_decomp -n {self.n} -w {self.w} -f {input_file} -o {output_file} -s {f.split('.')[1]} -c 1"
        if "pack" in f:
            CMD += " -p"

        print(f"\n{CMD}")
        result = subprocess.run([*CMD.split()], stdout=subprocess.DEVNULL)
        self.assertEqual(result.stderr, None)
        result = subprocess.run(["diff", f"{expected}", f"{output_file}"])
        self.assertEqual(result.returncode, 0)

    @parameterized.expand([*test_files])
    def test_single_gpu(self, f):
        method = f.split(".")[-1]
        input_file = f"{self.input_folder}/{f}"
        output_file = f"{self.output_folder}/{f}.out"
        expected = f"{self.input_folder}/{f.removesuffix(f'.root.{method}')}.in"
        CMD = f"./gpu_root_decomp -n {self.n} -w {self.w} -f {input_file} -o {output_file} -t {method}"
        if "pack" in f:
            CMD += " -p"

        print(f"\n{CMD}")
        result = subprocess.run([*CMD.split()], stdout=subprocess.DEVNULL)
        self.assertEqual(result.stderr, None)
        result = subprocess.run(["diff", f"{expected}", f"{output_file}"])
        self.assertEqual(result.returncode, 0)

    @parameterized.expand([*test_files])
    def test_multiple_cpu(self, f):
        method = f.split(".")[-1]
        input_file = f"{self.input_folder}/{f}"
        output_file = f"{self.output_folder}/{f}.out"
        expected = f"{self.input_folder}/{f.removesuffix(f'.root.{method}')}.in"

        m = 100
        with open(expected, "rb") as fe:
            with open("tmp", "wb") as ft:
                ft.write(fe.read() * m)

        CMD = f"./cpu_root_decomp -n {self.n} -w {self.w} -f {input_file} -o {output_file} -s {f.split('.')[1]} -m {m}"
        if "pack" in f:
            CMD += " -p"

        print(f"\n{CMD}")
        result = subprocess.run([*CMD.split()], stdout=subprocess.DEVNULL)
        self.assertEqual(result.stderr, None)
        result = subprocess.run(["diff", "tmp", f"{output_file}"])
        self.assertEqual(result.returncode, 0)
        result = subprocess.run(["rm", "tmp"])

    @parameterized.expand([*test_files])
    def test_multiple_gpu(self, f):
        method = f.split(".")[-1]
        input_file = f"{self.input_folder}/{f}"
        output_file = f"{self.output_folder}/{f}.out"
        expected = f"{self.input_folder}/{f.removesuffix(f'.root.{method}')}.in"

        m = 100
        with open(expected, "rb") as fe:
            with open("tmp", "wb") as ft:
                ft.write(fe.read() * m)

        CMD = f"./gpu_root_decomp -n {self.n} -w {self.w} -f {input_file} -o {output_file} -t {method} -m {m}"
        if "pack" in f:
            CMD += " -p"

        print(f"\n{CMD}")
        result = subprocess.run([*CMD.split()], stdout=subprocess.DEVNULL)
        self.assertEqual(result.stderr, None)
        result = subprocess.run(["diff", "tmp", f"{output_file}"])
        self.assertEqual(result.returncode, 0)
        result = subprocess.run(["rm", "tmp"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
