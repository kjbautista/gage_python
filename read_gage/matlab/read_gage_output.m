function [data, info] = read_gage_output(pathInput, calFilePath)
%READ_GAGE_OUTPUT Read Python GUI acquisition data (A-line, M-mode, or Beammap).
%
%   [data, info] = read_gage_output(pathInput)
%   [data, info] = read_gage_output(pathInput, calFilePath)
%
%   pathInput   - path to a data .npy file (A-line or M-mode) or a scan
%                 folder (Beammap). The data_type is read automatically from
%                 the JSON sidecar's acquisition_config.data_type field.
%   calFilePath - (optional) full path to a .txt calibration file that
%                 overrides the value stored in the JSON sidecar.

if nargin < 2
    calFilePath = '';
end

if isfolder(pathInput)
    [data, info] = local_read_beammap(pathInput, calFilePath);
else
    info = local_read_sidecar(pathInput);
    acqCfg = local_optional_field(info, 'acquisition_config', struct());
    dataType = local_optional_field(acqCfg, 'data_type', 'aline');
    switch dataType
        case 'aline'
            data = local_read_aline(pathInput, info, calFilePath);
        case 'mmode'
            data = local_read_mmode(pathInput, info, calFilePath);
        otherwise
            error('read_gage_output: unknown data_type ''%s''', dataType);
    end
end
end

% -------------------------------------------------------------------------
function data = local_read_aline(filename, info, calFilePath)
matrix = local_read_npy(filename);
voltageData = matrix(:, 2:end);
data = struct();
data.timeScale    = matrix(:, 1);
data.voltageData  = voltageData;
data.pressureData = local_voltage_to_pressure(voltageData, info, calFilePath, ...
    local_aline_cal_sidecar_path(filename));
data.dataFile     = string(filename);
end

% -------------------------------------------------------------------------
function data = local_read_mmode(filename, info, calFilePath)
% The Python M-mode worker now writes A-lines in display order, so no
% circular-buffer reordering is needed on read-back.
matrix = local_read_npy(filename);
voltageData = matrix(:, 2:end);
data = struct();
data.timeScale    = matrix(:, 1);
data.voltageData  = voltageData;
data.pressureData = local_voltage_to_pressure(voltageData, info, calFilePath, ...
    local_aline_cal_sidecar_path(filename));
data.dataFile     = string(filename);
end

% -------------------------------------------------------------------------
function sidecarPath = local_aline_cal_sidecar_path(filename)
[folderPath, stem] = fileparts(filename);
sidecarPath = fullfile(folderPath, strcat(stem, '.cal'));
end

% -------------------------------------------------------------------------
function [data, info] = local_read_beammap(scanDirectory, calFilePath)
metadataFiles = dir(fullfile(scanDirectory, 'acquisition_parameters_*.json'));
parametersPath = fullfile(metadataFiles(1).folder, metadataFiles(1).name);
info = jsondecode(fileread(parametersPath));
if isfield(info, 'timestamp')
    timestamp = char(string(info.timestamp));
else
    [~, stem, ext] = fileparts(parametersPath);
    token = regexp([stem ext], '^acquisition_parameters_(\d{8}_\d{6})\.json$', 'tokens', 'once');
    timestamp = token{1};
end

coordinates = local_read_npy(fullfile(scanDirectory, sprintf('coordinates_mm_%s.npy', timestamp)));

positionFiles = dir(fullfile(scanDirectory, sprintf('voltage_data_pos*_%s.npy', timestamp)));
fileIndices = zeros(numel(positionFiles), 1);
for k = 1:numel(positionFiles)
    token = regexp(positionFiles(k).name, '^voltage_data_pos(\d{4,})_', 'tokens', 'once');
    fileIndices(k) = str2double(token{1});
end
[~, sortOrder] = sort(fileIndices);
positionFiles = positionFiles(sortOrder);
nPositions = numel(positionFiles);

% Coordinate axes centered around zero
rawDim1 = unique(coordinates(:, 1));
rawDim2 = unique(coordinates(:, 2));
rawDim3 = unique(coordinates(:, 3));
dim1 = rawDim1 - mean(rawDim1);
dim2 = rawDim2 - mean(rawDim2);
dim3 = rawDim3 - mean(rawDim3);
n1 = numel(dim1); n2 = numel(dim2); n3 = numel(dim3);

% Resolve calibration once for the whole scan; per-position FFTs reuse it.
% Avoids holding an nPositions x nSamples matrix in memory at any point.
calSidecarPath = fullfile(scanDirectory, sprintf('cal_%s.cal', timestamp));
[hydroCals, calLabel] = local_resolve_calibration(calFilePath, calSidecarPath);
hasCal = ~isempty(hydroCals);
acqCfg = local_optional_field(info, 'acquisition_config', struct());
sampleRateHz = local_optional_field(acqCfg, 'sample_rate_hz', 0);

% Peak maps: n1 x n2 x n3 (only thing we keep across positions)
voltageData_peakPos = zeros(n1, n2, n3);
voltageData_peakNeg = zeros(n1, n2, n3);
if hasCal
    pressureData_peakPos = zeros(n1, n2, n3);
    pressureData_peakNeg = zeros(n1, n2, n3);
else
    pressureData_peakPos = [];
    pressureData_peakNeg = [];
end

warnedFreqOutOfRange = false;
for k = 1:nPositions
    posMatrix = local_read_npy(fullfile(positionFiles(k).folder, positionFiles(k).name));
    voltageTrace = mean(posMatrix, 2);   % nSamples x 1
    i1 = find(abs(rawDim1 - coordinates(k, 1)) < 1e-9);
    i2 = find(abs(rawDim2 - coordinates(k, 2)) < 1e-9);
    i3 = find(abs(rawDim3 - coordinates(k, 3)) < 1e-9);
    voltageData_peakPos(i1, i2, i3) = max(voltageTrace);
    voltageData_peakNeg(i1, i2, i3) = min(voltageTrace);
    if hasCal
        [pressureTrace, peakFreqHz, calMinHz, calMaxHz] = ...
            local_voltage_to_pressure_with_cal(voltageTrace, sampleRateHz, hydroCals);
        if ~warnedFreqOutOfRange && (peakFreqHz < calMinHz || peakFreqHz > calMaxHz)
            warning('read_gage_output:freqOutOfRange', ...
                'At scan position %d the peak signal frequency (%.2f MHz) is outside the calibration frequency range (%.2f-%.2f MHz) for "%s". Sensitivity is set to zero outside the calibration range.', ...
                k, peakFreqHz / 1e6, calMinHz / 1e6, calMaxHz / 1e6, calLabel);
            warnedFreqOutOfRange = true;
        end
        pressureData_peakPos(i1, i2, i3) = max(pressureTrace);
        pressureData_peakNeg(i1, i2, i3) = min(pressureTrace);
    end
end

data = struct();
data.coordinates_mm       = coordinates;
data.voltageData_peakPos  = voltageData_peakPos;
data.voltageData_peakNeg  = voltageData_peakNeg;
data.pressureData_peakPos = pressureData_peakPos;
data.pressureData_peakNeg = pressureData_peakNeg;
data.dim1 = dim1;
data.dim2 = dim2;
data.dim3 = dim3;
data.scan_directory = string(scanDirectory);
end

% -------------------------------------------------------------------------
function info = local_read_sidecar(filename)
[folderPath, stem] = fileparts(filename);
metadataPath = fullfile(folderPath, strcat(stem, '_parameters.json'));
info = jsondecode(fileread(metadataPath));
end

% -------------------------------------------------------------------------
function pressureData = local_voltage_to_pressure(voltageData, info, calFilePathOverride, calSidecarPath)
pressureData = [];
if ~isfield(info, 'acquisition_config')
    return;
end
[hydroCals, calLabel] = local_resolve_calibration(calFilePathOverride, calSidecarPath);
if isempty(hydroCals)
    return;
end
sampleRateHz = info.acquisition_config.sample_rate_hz;
[pressureData, peakFreqHz, calMinHz, calMaxHz] = ...
    local_voltage_to_pressure_with_cal(voltageData, sampleRateHz, hydroCals);
if peakFreqHz < calMinHz || peakFreqHz > calMaxHz
    warning('read_gage_output:freqOutOfRange', ...
        'Peak signal frequency (%.2f MHz) is outside the calibration frequency range (%.2f-%.2f MHz) for "%s". Sensitivity is set to zero outside the calibration range.', ...
        peakFreqHz / 1e6, calMinHz / 1e6, calMaxHz / 1e6, calLabel);
end
end

% -------------------------------------------------------------------------
function [pressureData, peakFreqHz, calMinHz, calMaxHz] = ...
    local_voltage_to_pressure_with_cal(voltageData, sampleRateHz, hydroCals)
%LOCAL_VOLTAGE_TO_PRESSURE_WITH_CAL Pure FFT-based pressure conversion using
% a pre-loaded calibration matrix. Returns the pressure trace plus the peak
% input frequency and calibration min/max so the caller can decide whether
% to emit a freq-out-of-range warning. No I/O, no warnings — safe to call
% in tight loops (e.g. per-position beammap reads).
f = hydroCals(:, 1) * 1e6;               % MHz -> Hz
pressSens = 1 ./ hydroCals(:, 2);        % V/Pa -> Pa/V
nyquistHz = sampleRateHz / 2;
calMaxHz = max(f);
calMinHz = min(f);
NFFT = 2^nextpow2(size(voltageData, 1));
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
pressureData = pressureData(1:size(voltageData, 1), :);
end

% -------------------------------------------------------------------------
function [calData, calLabel] = local_resolve_calibration(calOverridePath, calSidecarPath)
%LOCAL_RESOLVE_CALIBRATION Return Nx2 [freq_mhz, sens_v_per_pa] calibration.
%
% Resolution order:
%   1. calOverridePath - full path to an Onda .txt file (highest priority).
%   2. calSidecarPath  - the <stem>_cal.npy / cal_<ts>.npy saved next to the data.
%   3. Neither -> warn and return empty.
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
    calData = local_read_npy(char(calSidecarPath));
    return;
end
[~, sidecarName, sidecarExt] = fileparts(char(calSidecarPath));
warning('read_gage_output:missingCalibration', ...
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

% -------------------------------------------------------------------------
function array = local_read_npy(filename)
%LOCAL_READ_NPY Minimal NumPy .npy v1/v2/v3 reader for the dtypes that the
% acquisition workers actually emit (float64 and other common numeric
% scalars). Returns a double array. Handles both C and Fortran array order
% by reading the bytes column-major into MATLAB and permuting if needed.
fid = fopen(filename, 'rb', 'ieee-le');
if fid < 0
    error('read_gage_output:cannotOpenNpy', 'Cannot open %s', filename);
end
cleanupObj = onCleanup(@() fclose(fid));

magic = fread(fid, 6, 'uint8=>uint8')';
expected = uint8([147 78 85 77 80 89]);   % \x93 N U M P Y
if numel(magic) ~= 6 || ~isequal(magic, expected)
    error('read_gage_output:badNpyMagic', 'Not a .npy file: %s', filename);
end

majorVer = fread(fid, 1, 'uint8');
fread(fid, 1, 'uint8');                    % minor version (unused)
switch majorVer
    case 1
        headerLen = fread(fid, 1, 'uint16');
    case {2, 3}
        headerLen = fread(fid, 1, 'uint32');
    otherwise
        error('read_gage_output:unsupportedNpyVersion', ...
            'Unsupported .npy version %d in %s', majorVer, filename);
end

header = fread(fid, headerLen, 'uint8=>char')';

descrToken = regexp(header, '''descr''\s*:\s*''([^'']+)''', 'tokens', 'once');
if isempty(descrToken)
    error('read_gage_output:badNpyHeader', 'Cannot parse descr in %s', filename);
end
descr = descrToken{1};

fortranToken = regexp(header, '''fortran_order''\s*:\s*(True|False)', 'tokens', 'once');
fortranOrder = ~isempty(fortranToken) && strcmp(fortranToken{1}, 'True');

shapeToken = regexp(header, '''shape''\s*:\s*\(([^)]*)\)', 'tokens', 'once');
if isempty(shapeToken)
    error('read_gage_output:badNpyHeader', 'Cannot parse shape in %s', filename);
end
shapeStr = strtrim(shapeToken{1});
if isempty(shapeStr)
    shape = [1, 1];
else
    parts = strsplit(shapeStr, ',');
    shape = zeros(1, 0);
    for k = 1:numel(parts)
        token = strtrim(parts{k});
        if isempty(token)
            continue;
        end
        shape(end+1) = str2double(token); %#ok<AGROW>
    end
    if isscalar(shape)
        shape = [shape, 1];   % column vector for 1-D arrays
    end
end

switch descr
    case {'<f8', '=f8', 'f8'}
        precision = 'double=>double';
    case {'<f4', '=f4', 'f4'}
        precision = 'single=>double';
    case {'<i8', '=i8'}
        precision = 'int64=>double';
    case {'<i4', '=i4'}
        precision = 'int32=>double';
    case {'<i2', '=i2'}
        precision = 'int16=>double';
    case {'|i1', 'i1'}
        precision = 'int8=>double';
    case {'<u8', '=u8'}
        precision = 'uint64=>double';
    case {'<u4', '=u4'}
        precision = 'uint32=>double';
    case {'<u2', '=u2'}
        precision = 'uint16=>double';
    case {'|u1', 'u1', '|b1'}
        precision = 'uint8=>double';
    otherwise
        error('read_gage_output:unsupportedNpyDtype', ...
            'Unsupported .npy dtype "%s" in %s', descr, filename);
end

nElements = prod(shape);
raw = fread(fid, nElements, precision);
if numel(raw) ~= nElements
    error('read_gage_output:truncatedNpy', 'Truncated .npy data in %s', filename);
end

if isscalar(shape)
    array = raw(:);
elseif fortranOrder
    array = reshape(raw, shape);
else
    % NumPy default is C (row-major); MATLAB reshape is column-major. Reading
    % into the reversed shape and permuting back recovers the logical array.
    array = reshape(raw, fliplr(shape));
    array = permute(array, numel(shape):-1:1);
end
end
