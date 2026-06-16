#!/bin/bash
sizes=(10 100 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
for size in "${sizes[@]}"
do
    ./compile.sh $size
    for it in {1..30}
    do
        echo "Iteration $it"
        echo ""
        ./mainCuda
        echo ""
        ./main
        echo ""
    done
done
