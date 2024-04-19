#!/bin/bash

echo "==== running on CPU ===="
python3 /app/gpu_cpu_tester.py, , 

echo "==== running on GPU ===="
python3 gpu_cpu_tester.py gpu

echo "==== done... sleeping for 4h ===="
sleep 4h