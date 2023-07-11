#!/bin/bash

Compare() {
    local input_file=$1
    local method=$2
    local output_file=$3
    local expected=$4
    CMD="./gpu_root_decomp -v -f ${input_file} -t ${method} -o ${output_file} && diff ${expected} ${output_file}"
    echo "${CMD}"
    eval ${CMD}
    echo $'\n'$'\n'
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
   Compare "${INPUT_FOLDER}/${FILE}.root.zstd" "zstd" "${OUTPUT_FOLDER}/${FILE}.zstd.out" "${INPUT_FOLDER}/${FILE}"
   Compare "${INPUT_FOLDER}/${FILE}.root.zlib" "zlib" "${OUTPUT_FOLDER}/${FILE}.zlib.out" "${INPUT_FOLDER}/${FILE}"
   Compare "${INPUT_FOLDER}/${FILE}.root.lz4" "lz4" "${OUTPUT_FOLDER}/${FILE}.lz4.out" "${INPUT_FOLDER}/${FILE}"
done
