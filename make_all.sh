#!/bin/bash

set -ue

DIRS=`find -maxdepth 1 -type d | grep -x "\./[0-9]*" | sort --version-sort`

for DIR in $DIRS; do
    echo $DIR
    cd $DIR
    make
    cd - > /dev/null
done
