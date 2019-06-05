#!/bin/bash

# Script to list the R2 and RMSE statistics from model folders

get_stats_file() {
    foo=$(ls "$1"/model.ckpt-*.test_stats 2>/dev/null 1>&1)
    if [ ! -z "$foo" ]; then
        echo $foo
    else
        echo ""
    fi
}

read_stats_file() {
    r2line=$(head -n 1 $1)
    printf "r2: "
    for a in $r2line; do
        printf "%0.4f " "$a"
    done
    printf "\n"
    rmseline=$(sed '2q;d' $1) 
    printf "RMSE (mT): "
    for a in $rmseline ; do
        awk '{printf "%2.3f ", 1000 * $0}' <<< "$a"
    done
    printf "\n"
    maeline=$(cat $1 | tail -n 1)
    printf "MAE (mT): "
    for a in $maeline; do
        awk '{printf "%2.3f ", 1000 * $0}' <<< "$a"
    done
    printf "\n\n"
}

print_all_stats() {
    models=$(ls -d $1/*)
    for model in $models; do
        stats=$(get_stats_file $model) 
        if [ "$stats" != "" ]; then
            echo $stats
            read_stats_file $stats
        fi
    done
}


if [ "$1" != "" ]; then
    echo "there is an argument"
    for dataset in $1; do
        print_all_stats $dataset
    done

else
    datasets=$(ls -d log/*)
    for dataset in $datasets; do
        print_all_stats $dataset
    done
fi
