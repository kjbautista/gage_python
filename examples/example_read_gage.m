%% EXAMPLE: READ GAGE OUTPUT FILES
% using read_gage_output.m to read A-line, M-mode, and beammap data from Gage acquisition files.

clear all
close all
clc

% add read_gage path to directory
addpath(genpath('read_gage'))

%% READ A-LINE (if hydrophone IS NOT defined during the acquisition)

aline_file = "examples\example_data\example_aline_without_cal_file.npy"; % data to read
cal_file = "read_gage\calibrations\h1344_p2040_rightangle_atten.txt"; % calibration file for the hydrophone used during the acquisition
[data,info] = read_gage_output(aline_file, cal_file); % read the data and info, convert voltage to pressure

% plot data
figure(1);
subplot(1,2,1)
plot(data.timeScale',data.pressureData'*1e-6)
axis tight;
grid on;
xlabel('Time (us)')
ylabel('Pressure (MPa)')
subplot(1,2,2)
plot(data.timeScale',data.voltageData*1e3)
axis tight;
grid on;
xlabel('Time (us)')
ylabel('Voltage (mV)')

%% READ A-LINE (if hydrophone IS defined during the acquisition)

aline_file = "examples\example_data\example_aline_with_cal_file.npy"; % data to read
[data,info] = read_gage_output(aline_file); % read the data and info, convert voltage to pressure

% plot data
figure(2);
subplot(1,2,1)
plot(data.timeScale',data.pressureData'*1e-6)
axis tight;
grid on;
xlabel('Time (us)')
ylabel('Pressure (MPa)')
subplot(1,2,2)
plot(data.timeScale',data.voltageData*1e3)
axis tight;
grid on;
xlabel('Time (us)')
ylabel('Voltage (mV)')

%% READ M-MODE

% TODO: need an example file

%% READ BEAMMAP - 2D
beammap_folder = "C:\Users\dayton-lab\Documents\20260421\beammap_20260421_155852_1p8V_at_foc25_pulse1ms_2D";
tic
[data,info] = read_gage_output(beammap_folder); % read the data and info, convert voltage to pressure
toc

figure(4);
subplot(1,2,1)
imagesc(data.dim1,data.dim2,data.pressureData_peakPos'.*1e-3)
xlabel('Lateral (mm)')
ylabel('Axial (mm)')
c = colorbar;
c.Label.String = 'Peak Pos. Pressure (kPa)';
axis image;
subplot(1,2,2)
imagesc(data.dim1,data.dim2,abs(data.pressureData_peakNeg'.*1e-3))
axis image;
xlabel('Lateral (mm)')
ylabel('Axial (mm)')
c = colorbar;
c.Label.String = 'Peak Neg. Pressure (kPa)';

%% READ BEAMMAP - 3D

beammap_folder = "C:\Users\dayton-lab\Documents\20260421\beammap_20260421_143743_1p8V_at_foc25_pulse0p5ms_3D";
tic
[data,info] = read_gage_output(beammap_folder); % read the data and info, convert voltage to pressure
toc

figure(5);
for k0 = 1:length(data.dim3)
    tmp = data.pressureData_peakNeg(:,:,k0).*1e-3;
    imagesc(data.dim1,data.dim2,squeeze(abs(tmp))')
    xlabel('Lateral (mm)')
    ylabel('Axial (mm)')
    c = colorbar;
    c.Label.String = 'Peak Neg. Pressure (kPa)';
    axis image;
    title(sprintf('Elev. position: %.2f mm',data.dim3(k0)),"FontWeight","normal")
    drawnow;
    pause(.5)
end


