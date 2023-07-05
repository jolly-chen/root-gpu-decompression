#!/bin/bash

compare() {
    local input_file=$1
    local method=$2
    local output_file=$3
    local expected=$4
    ./ll_decomp -f ${input_file} -t ${method} -o ${output_file} && diff ${expected} ${output_file}
}

compare "input/zeroes_11m.zst" "zstd" "output/zeroes_11m.zstd" "input/zeroes_11m"

compare "input/zeroes_11m.zlib" "zlib" "output/zeroes_11m.zlib" "input/zeroes_11m"

compare "input/zeroes_11m.lz4" "lz4" "output/zeroes_11m.lz4" "input/zeroes_11m"
