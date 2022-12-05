
clear all
close all
clc

%%% Only two files are needed: the neltlist.log and the *.cs project file

%%% Add line  ASCII_Rawfile=yes in the option line of the netlist.log file
%%% Options ResourceUsage=yes UseNutmegFormat=no ASCII_Rawfile=yes EnableOptim=no TopDesignName="MyLibrary_lib:cell_1:schematic" DcopOutputNodeVoltages=yes DcopOutputPinCurrents=yes DcopOutputAllSweepPoints=no DcopOutputDcopType=0

 
projDir='C:\Users\idlab261\Documents\Matlab\Ongoing\Research_Projects\Matlab_ADS_Param_Circuit_Simulations\ADS_files';
pinPath=fullfile(projDir,'netlist.log');
netlistData=importdata(pinPath);

 
%set value resistance 
Rnew=30;
netlistData{4}=['R:R6  Vout_PAM_2 0 R=' num2str(Rnew,20) ' Ohm Noise=yes '];
 

%set value capacitance
Cnew=0.15;
netlistData{44}=['C:C3  Vout_PAM_2 0 C=' num2str(Cnew,20) ' pF '];

%set new value microstrip spacing
Sp_new=92;
netlistData{42}=['MCLIN:CLin2  Vin_PAM_1 Vin_PAM_2 Vout_PAM_2 Vout_PAM_1 Subst="MSub1" W=50 um S=' num2str(Sp_new,20) ' um L=5 cm '];

%write port location file
fid = fopen(pinPath, 'w+');
fprintf(fid, '%s\n', netlistData{:});
fclose(fid);

 
system('C:\Users\idlab261\Documents\Matlab\Ongoing\Research_Projects\Matlab_ADS_Param_Circuit_Simulations\ADS_files\initializationD3.bat')

% resPath=fullfile(projDir,'spectra.raw');
% result = read_raw(resPath) 
