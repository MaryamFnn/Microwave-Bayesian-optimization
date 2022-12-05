#!/bin/bash


cd /media/data/Users/fgarb/Documents/PycharmProjects/MicrowaveOpt/microwaveopt/lib_example/ex10/ADS_Original_Files_2015/
export SIMARCH=linux_x86_64
export HOME=/media/data/Users/fgarb/Documents/PycharmProjects/MicrowaveOpt/microwaveopt/lib_example/ex10/ADS_Original_Files_2015/
export HPEESOF_DIR=/usr/local/ADS2015_01
export COMPL_DIR=$HPEESOF_DIR
export LD_LIBRARY_PATH=$HPEESOF_DIR/lib/linux_x86_64:$HPEESOF_DIR/adsptolemy/lib.linux_x86_64:$HPEESOF_DIR/SystemVue/2014.10/linux_x86_64/lib
export PATH=$HPEESOF_DIR/bin/$SIMARCH:$HPEESOF_DIR/bin:$HPEESOF_DIR/lib/$SIMARCH:$HPEESOF_DIR/circuit/lib.$SIMARCH:$HPEESOF_DIR/adsptolemy/lib.$SIMARCH:
export ADS_LICENSE_FILE=27000@license.intec.ugent.be

hpeesofsim netlist.log
