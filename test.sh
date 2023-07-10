#!/bin/bash

Compare() {
    local input_file=$1
    local method=$2
    local output_file=$3
    local expected=$4
    ./gpu_root_decomp -f ${input_file} -t ${method} -o ${output_file} && diff ${expected} ${output_file}
}

INPUT_FOLDER="input/"
OUTPUT_FOLDER="output/"
TEST_FILES=(
    "uniform_random_100m"
    "gauss_random_100m"
    "ones_100"
)

for FILE in "${TEST_FILES[@]}"
do
   Compare "${INPUT_FOLDER}/${FILE}.root.zst" "zstd" "${OUTPUT_FOLDER}/${FILE}.zst.out" "${INPUT_FOLDER}/${FILE}"
done
