#!/bin/bash
cd ../microwaveopt/lib_example/ex1/ADS_Parametric_Layout
export HPEESOF_DIR=/usr/local/ADS2015_01
export ADS_LICENSE_FILE=27000@license.intec.ugent.be

export PATH=$HPEESOF_DIR/bin:$PATH
adsMomWrapper -O -3D proj proj
