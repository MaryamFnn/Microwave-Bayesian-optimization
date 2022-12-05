set HPEESOF_DIR=C:\Program Files\Keysight\ADS2019

set ADS_LICENSE_FILE=27000@license.intec.ugent.be
set PATH=%HPEESOF_DIR%\bin;%PATH%
cd ADS_Parametric_Layout
adsMomWrapper -O -3D proj proj
