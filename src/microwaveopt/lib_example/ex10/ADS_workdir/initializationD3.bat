set SIMARCH=win32_64
set HOME=C:\Users\idlab261\Documents\Matlab\Ongoing\Research_Projects\Matlab_ADS_Param_Circuit_Simulations\ADS_files
set HPEESOF_DIR=C:\Program Files\Keysight\ADS2017_Update1
set COMPL_DIR=%HPEESOF_DIR%
set PATH=%HPEESOF_DIR%\bin\%SIMARCH%;%HPEESOF_DIR%\bin;%HPEESOF_DIR%\lib\%SIMARCH%;%HPEESOF_DIR%\circuit\lib.%SIMARCH%;%HPEESOF_DIR%\adsptolemy\lib.%SIMARCH%;%SVECLIENT_DIR%/bin/MATLABScript/runtime/win64;%SVECLIENT_DIR%/sveclient;%PATH%
set ADS_LICENSE_FILE=27000@license.intec.ugent.be

 

hpeesofsim netlist.log 