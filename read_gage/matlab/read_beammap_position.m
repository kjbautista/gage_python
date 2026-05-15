function [data, info] = read_beammap_position(scanDirectory, positionIndex, calFilePath)
%READ_BEAMMAP_POSITION Load all averaged A-lines for one beammap position.
%
%   [data, info] = read_beammap_position(scanDirectory, positionIndex)
%   [data, info] = read_beammap_position(scanDirectory, positionIndex, calFilePath)
%
%   scanDirectory  - path to a beammap save folder (the one containing the
%                    acquisition_parameters_<timestamp>.json sidecar).
%   positionIndex  - 1-based scan position number. Matches the file
%                    voltage_data_pos<NNNN>_<timestamp>.npy and the row of
%                    coordinates_mm_<timestamp>.npy.
%   calFilePath    - (optional) full path to a .txt calibration file that
%                    overrides the cal_<timestamp>.cal stored alongside the
%                    scan data.
%
%   Returns a struct with the same fields produced by read_gage_output for
%   A-line data (timeScale, voltageData, pressureData, dataFile) plus the
%   coordinate and position index for this point. info is the JSON sidecar.

if nargin < 3
    calFilePath = '';
end
if ~isfolder(scanDirectory)
    error('read_beammap_position:badDirectory', ...
        '%s is not a folder.', scanDirectory);
end
if ~isnumeric(positionIndex) || ~isscalar(positionIndex) || ...
        positionIndex < 1 || positionIndex ~= floor(positionIndex)
    error('read_beammap_position:badIndex', ...
        'positionIndex must be a positive integer (1-based), got %s.', ...
        mat2str(positionIndex));
end

metadataFiles = dir(fullfile(scanDirectory, 'acquisition_parameters_*.json'));
if isempty(metadataFiles)
    error('read_beammap_position:missingSidecar', ...
        'No acquisition_parameters_*.json found in %s', scanDirectory);
end
parametersPath = fullfile(metadataFiles(1).folder, metadataFiles(1).name);
info = jsondecode(fileread(parametersPath));
if isfield(info, 'timestamp') && ~isempty(info.timestamp)
    timestamp = char(string(info.timestamp));
else
    [~, stem, ext] = fileparts(parametersPath);
    token = regexp([stem ext], '^acquisition_parameters_(\d{8}_\d{6})\.json$', 'tokens', 'once');
    if isempty(token)
        error('read_beammap_position:badTimestamp', ...
            'Cannot determine timestamp from %s', parametersPath);
    end
    timestamp = token{1};
end

coordinatesPath = fullfile(scanDirectory, sprintf('coordinates_mm_%s.npy', timestamp));
coordinates = read_npy(coordinatesPath);
nPositions = size(coordinates, 1);
if positionIndex > nPositions
    error('read_beammap_position:indexOutOfRange', ...
        'positionIndex %d exceeds the %d scan positions in %s.', ...
        positionIndex, nPositions, coordinatesPath);
end

positionFile = fullfile(scanDirectory, ...
    sprintf('voltage_data_pos%04d_%s.npy', positionIndex, timestamp));
if ~isfile(positionFile)
    error('read_beammap_position:missingPositionFile', ...
        'Cannot find %s', positionFile);
end
voltageData = read_npy(positionFile);

timeAxisPath = fullfile(scanDirectory, sprintf('time_axis_us_%s.npy', timestamp));
if ~isfile(timeAxisPath)
    error('read_beammap_position:missingTimeAxis', ...
        'Cannot find %s', timeAxisPath);
end
timeScale = read_npy(timeAxisPath);

acqCfg = local_optional_field(info, 'acquisition_config', struct());
sampleRateHz = local_optional_field(acqCfg, 'sample_rate_hz', 0);
calSidecarPath = fullfile(scanDirectory, sprintf('cal_%s.cal', timestamp));
pressureData = local_voltage_to_pressure(voltageData, sampleRateHz, ...
    calFilePath, calSidecarPath);

data = struct();
data.timeScale      = timeScale(:);
data.voltageData    = voltageData;
data.pressureData   = pressureData;
data.coordinate_mm  = coordinates(positionIndex, :);
data.position_index = positionIndex;
data.dataFile       = string(positionFile);
end

% -------------------------------------------------------------------------
function pressureData = local_voltage_to_pressure(voltageData, sampleRateHz, calOverridePath, calSidecarPath)
pressureData = [];
[hydroCals, calLabel] = local_resolve_calibration(calOverridePath, calSidecarPath);
if isempty(hydroCals) || sampleRateHz <= 0
    return;
end
f = hydroCals(:, 1) * 1e6;               % MHz -> Hz
pressSens = 1 ./ hydroCals(:, 2);        % V/Pa -> Pa/V
nyquistHz = sampleRateHz / 2;
calMaxHz = max(f);
calMinHz = min(f);
nSamples = size(voltageData, 1);
NFFT = 2^nextpow2(nSamples);
% Remove DC offset before FFT, matching the Python reader.
voltageData = voltageData - mean(voltageData, 1);
fdata = fft(voltageData, NFFT, 1);
freqOfFFT = nyquistHz * linspace(0, 1, NFFT / 2 + 1);
fMagOneSided = mean(abs(fdata(1:NFFT/2+1, :)), 2);
[~, peakIdx] = max(fMagOneSided);
peakFreqHz = freqOfFFT(peakIdx);
pressSensInterp = interp1(f, pressSens, freqOfFFT, 'linear', 0);
sens2 = [pressSensInterp(1:end-1) fliplr(pressSensInterp)]';
sens2(end) = [];
fcorr = fdata .* repmat(sens2, 1, size(fdata, 2));
pressureData = real(ifft(fcorr, NFFT, 1));
pressureData = pressureData(1:nSamples, :);
if peakFreqHz < calMinHz || peakFreqHz > calMaxHz
    warning('read_beammap_position:freqOutOfRange', ...
        'Peak signal frequency (%.2f MHz) is outside the calibration frequency range (%.2f-%.2f MHz) for "%s". Sensitivity is set to zero outside the calibration range.', ...
        peakFreqHz / 1e6, calMinHz / 1e6, calMaxHz / 1e6, calLabel);
end
end

% -------------------------------------------------------------------------
function [calData, calLabel] = local_resolve_calibration(calOverridePath, calSidecarPath)
calData = [];
calLabel = '';
if ~isempty(calOverridePath) && isfile(calOverridePath)
    overridePath = char(calOverridePath);
    [~, calLabel] = fileparts(overridePath);
    fid = fopen(overridePath, 'r');
    line = fgetl(fid);
    while ischar(line) && ~startsWith(strtrim(line), 'HEADER_END')
        line = fgetl(fid);
    end
    rawData = textscan(fid, '%f %f %f %f', 'CommentStyle', '#');
    fclose(fid);
    calData = [rawData{1}, rawData{3}];   % FREQ_MHZ and SENS_VPERPA
    return;
end
if ~isempty(calSidecarPath) && isfile(calSidecarPath)
    [~, calLabel] = fileparts(char(calSidecarPath));
    calData = read_npy(char(calSidecarPath));
    return;
end
[~, sidecarName, sidecarExt] = fileparts(char(calSidecarPath));
warning('read_beammap_position:missingCalibration', ...
    'Calibration sidecar not found (%s%s). Voltage data will not be converted to pressure.', ...
    sidecarName, sidecarExt);
end

% -------------------------------------------------------------------------
function value = local_optional_field(s, fieldName, defaultValue)
if isfield(s, fieldName)
    value = s.(fieldName);
else
    value = defaultValue;
end
end
