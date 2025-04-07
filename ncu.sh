#!/bin/bash

ncu --set full --target-processes all -f -o report ./build/test 

ncu-ui report.ncu-rep
