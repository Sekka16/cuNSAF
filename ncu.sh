#!/bin/bash

ncu --set full --target-processes all -f -o report ./build/test 

ncu-ui report.ncu-rep

# ncu --metrics sm__warps_active.avg.pct_of_peak_sustained,dram__throughput.avg.pct_of_peak_sustained ./build/test
