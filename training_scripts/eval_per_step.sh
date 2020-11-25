#! /bin/bash

STEPS=(84600 86400 88200 90000 91800 93600 95400 97200 99000 100800 102600 104400 106200 108000 109800 111600 113400 115200)

for step in "${STEPS[@]}"
do
    echo $step
    # modify eval script
     sed  -i "s/Step=[0-9]*/Step=$step/g" eval.ini
    # run eval
    python evaluate.py eval.ini
done
