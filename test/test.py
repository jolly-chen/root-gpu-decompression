import os
import glob
import subprocess
import unittest
import numpy as np
from parameterized import parameterized

INPUT_FOLDER = "input/"
OUTPUT_FOLDER = "output/"
EXE="./gpu_root_decomp"

os.chdir("../input")
TEST_FILES = glob.glob("*.root.*")
EXPECTED_FILES = glob.glob("*.in")
os.chdir("..")


class TestSequence(unittest.TestCase):
    @parameterized.expand([*TEST_FILES])
    def test_single(self, f):
        method = f.split(".")[-1]
        input_file = f"{INPUT_FOLDER}/{f}"
        output_file = f"{OUTPUT_FOLDER}/{f}.out"
        expected = f"{INPUT_FOLDER}/{f.removesuffix(f'.root.{method}')}.in"
        CMD = f"{EXE} -n 1 -f {input_file} -o {output_file} -t {method}"

        print(CMD)
        subprocess.run([*CMD.split()])
        result = subprocess.run(["diff", f"{expected}", f"{output_file}"])
        self.assertEqual(result.returncode, 0)

    @parameterized.expand([*TEST_FILES])
    def test_multiple(self, f):
        method = f.split(".")[-1]
        input_file = f"{INPUT_FOLDER}/{f}"
        output_file = f"{OUTPUT_FOLDER}/{f}.out"
        expected = f"{INPUT_FOLDER}/{f.removesuffix(f'.root.{method}')}.in"

        m = 100
        with open(expected, "rb") as fe:
            with open("tmp", "wb") as ft:
                ft.write(fe.read() * m)

        CMD = f"{EXE} -n 1 -f {input_file} -o {output_file} -t {method} -m {m}"

        print(CMD)
        subprocess.run([*CMD.split()])
        result = subprocess.run(["diff", "tmp", f"{output_file}"])
        self.assertEqual(result.returncode, 0)
        result = subprocess.run(["rm", "tmp"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
