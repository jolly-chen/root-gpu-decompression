import os
import glob
import subprocess

INPUT_FOLDER="input/"
OUTPUT_FOLDER="output/"

def Compare(input_file, method, output_file, expected):
    CMD=f"./cpu_root_decomp -n 1 -f {input_file} -o {output_file} -s {50000000} && diff {expected} {output_file}"
    CMD=f"./gpu_root_decomp -n 1 -f {input_file} -o {output_file} -t {method} && diff {expected} {output_file}"
    subprocess.run([*CMD.split()])



os.chdir("../input")
TEST_FILES = glob.glob("*.root.*")
EXPECTED_FILES= glob.glob("*.in")
os.chdir("..")

for f, e in zip(TEST_FILES, EXPECTED_FILES):
   Compare(f"{INPUT_FOLDER}/{f}", "zstd", f"{OUTPUT_FOLDER}/{f}.out", f"{e}")
   Compare(f"{INPUT_FOLDER}/{f}", "zlib", f"{OUTPUT_FOLDER}/{f}.out", f"{e}")
   Compare(f"{INPUT_FOLDER}/{f}", "lz4", f"{OUTPUT_FOLDER}/{f}.out", f"{e}")

