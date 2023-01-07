#!/bin/bash

set -ue

BIN_LIST=(
    "1-rgb_to_bgr"
    "2-rgb_to_gray"
    "3-binarization"
    "4-otsu_binarization"
    "5-rgb_to_hsv"
    "6-color_reduction"
    "7-average_pooling"
    "8-max_pooling"
    "9-gaussian_filter"
    "11-mean_filter"
    "47-dilation"
)

# default params
BUILD_DIR="build"
TARGET="all"
NUM_ITERATIONS=100
OPTIONS="--cpp"

# parse args
while (( $# > 0 )); do
    case $1 in
    --build | -b)
        ;;
    --target | -t)
        if [ $# -lt 2 ]; then
            echo "Missing argument for --target."
            exit -1
        fi
        TARGET=$2
        # check target
        CHECK_RES="NG"
        if [ ${TARGET} = "all" ]; then
            CHECK_RES="OK"
        fi
        for BIN in ${BIN_LIST[@]}; do
            if [ ${TARGET} = ${BIN} ]; then
                CHECK_RES="OK"
                break
            fi
        done
        if [ ${CHECK_RES} = "NG" ]; then
            echo "Invalid target."
            exit -1
        fi
        ;;
    --iterations | -itr)
        if [ $# -lt 2 ]; then
            echo "Missing argument for --iterations."
            exit -1
        fi
        NUM_ITERATIONS=$2
        ;;
    --simd | --neon)
        OPTIONS="${OPTIONS} --simd"
        ;;
    --cuda)
        OPTIONS="${OPTIONS} --cuda"
        ;;
    --help | -h)
        echo "Usage: bash run_target.bash [--target(-t) TARGET] [--itrations(-itr) ITR]  [--simd] [--cuda] [--help(-h)]"
        exit 0
        ;;
    esac
    shift
done

# print params
echo "Build dir      : ${BUILD_DIR}"
echo "Target         : ${TARGET}"
echo "Iterations     : ${NUM_ITERATIONS}"
echo "Passed options : ${OPTIONS}"
echo ""

if [ ${TARGET} = "all" ]; then
    for BIN in ${BIN_LIST[@]}; do
        echo ${BIN}
        ./${BUILD_DIR}/${BIN}/${BIN} ${NUM_ITERATIONS} --simd --cuda
    done
else
    echo ${TARGET}
    ./${BUILD_DIR}/${TARGET}/${TARGET} ${NUM_ITERATIONS} --simd --cuda
fi
