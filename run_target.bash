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
    "10-median_filter"
    "11-mean_filter"
    "47-dilation"
)

# default params
BUILD_DIR="build"
TARGET="all"
NUM_ITERATIONS=100
OPTIONS="--cpp"

function print_usage () {
    echo "Usage: bash run_target.bash [--build(-b) BUILD_DIR] [--target(-t) TARGET] [--itrations(-itr) ITR]"
    echo "                            [--simd] [--cuda] [--dump] [--help(-h)]"
}

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
        shift
        ;;
    --iterations | -itr)
        if [ $# -lt 2 ]; then
            echo "Missing argument for --iterations."
            exit -1
        fi
        NUM_ITERATIONS=$2
        shift
        ;;
    --simd | --neon)
        OPTIONS="${OPTIONS} --simd"
        ;;
    --cuda)
        OPTIONS="${OPTIONS} --cuda"
        ;;
    --dump)
        OPTIONS="${OPTIONS} --dump"
        ;;
    --help | -h)
        print_usage
        exit 0
        ;;
    *)
        echo "Unknown argument $1"
        echo ""
        print_usage
        exit -1
        ;;
    esac
    shift
done

OPTIONS="--itr ${NUM_ITERATIONS} ${OPTIONS}"

# print params
echo "Build dir      : ${BUILD_DIR}"
echo "Target         : ${TARGET}"
echo "Passed options : ${OPTIONS}"
echo ""

if [ ${TARGET} = "all" ]; then
    for BIN in ${BIN_LIST[@]}; do
        echo ${BIN}
        ./${BUILD_DIR}/${BIN}/${BIN} ${NUM_ITERATIONS} ${OPTIONS}
    done
else
    echo ${TARGET}
    ./${BUILD_DIR}/${TARGET}/${TARGET} ${NUM_ITERATIONS} ${OPTIONS}
fi
