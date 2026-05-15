function array = read_npy(filename)
%READ_NPY Minimal NumPy .npy v1/v2/v3 reader.
%
%   array = read_npy(filename) returns a double array loaded from the given
%   .npy file. Useful for inspecting beammap sidecar files directly, e.g.
%   coordinates = read_npy('coordinates_mm_20260330_120000.npy');
%
%   Supports the dtypes that the acquisition workers actually emit
%   (float64 and other common numeric scalars) and handles both C and
%   Fortran array order.

fid = fopen(filename, 'rb', 'ieee-le');
if fid < 0
    error('read_npy:cannotOpen', 'Cannot open %s', filename);
end
cleanupObj = onCleanup(@() fclose(fid));

magic = fread(fid, 6, 'uint8=>uint8')';
expected = uint8([147 78 85 77 80 89]);   % \x93 N U M P Y
if numel(magic) ~= 6 || ~isequal(magic, expected)
    error('read_npy:badMagic', 'Not a .npy file: %s', filename);
end

majorVer = fread(fid, 1, 'uint8');
fread(fid, 1, 'uint8');                    % minor version (unused)
switch majorVer
    case 1
        headerLen = fread(fid, 1, 'uint16');
    case {2, 3}
        headerLen = fread(fid, 1, 'uint32');
    otherwise
        error('read_npy:unsupportedVersion', ...
            'Unsupported .npy version %d in %s', majorVer, filename);
end

header = fread(fid, headerLen, 'uint8=>char')';

descrToken = regexp(header, '''descr''\s*:\s*''([^'']+)''', 'tokens', 'once');
if isempty(descrToken)
    error('read_npy:badHeader', 'Cannot parse descr in %s', filename);
end
descr = descrToken{1};

fortranToken = regexp(header, '''fortran_order''\s*:\s*(True|False)', 'tokens', 'once');
fortranOrder = ~isempty(fortranToken) && strcmp(fortranToken{1}, 'True');

shapeToken = regexp(header, '''shape''\s*:\s*\(([^)]*)\)', 'tokens', 'once');
if isempty(shapeToken)
    error('read_npy:badHeader', 'Cannot parse shape in %s', filename);
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
        error('read_npy:unsupportedDtype', ...
            'Unsupported .npy dtype "%s" in %s', descr, filename);
end

nElements = prod(shape);
raw = fread(fid, nElements, precision);
if numel(raw) ~= nElements
    error('read_npy:truncated', 'Truncated .npy data in %s', filename);
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
