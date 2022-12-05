set HPEESOF_DIR=C:\Program Files\Keysight\ADS2015_01
set ADS_LICENSE_FILE=27000@license.intec.ugent.be
set PATH=%HPEESOF_DIR%\bin;%PATH%
cd C:\\Users\\Administrator\\Desktop\\Sources\\MicrowaveOptProject\\MyCodeForAntennaPatch\\MicrowaveOpt-new\\microwaveopt\\lib_example\\ADS_ex\\PatchAntenna_ex_param
adsMomWrapper -O -3D proj proj
