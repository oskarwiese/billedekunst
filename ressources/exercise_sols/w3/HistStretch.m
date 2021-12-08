function Io = HistStretch(I)
% Function that stretches the histogram of an image
% Input:
%   I = Image you want to be stretched
% Output:
%   Io = The histogram-stretched image

Itemp = double(I);  %First we have to convert to a format that can contain decimals

vmind = 0;  % General minimum pixel value defines (256 bit image)
vmaxd = 255;    % General maximum pixel value defined (256bit image)
vmin = min(min(Itemp));   % Min pixel value for the image defined
vmax = max(max(Itemp));   % Max pixel value for the image defined

Im = (vmaxd-vmind)/(vmax - vmin) *(I-vmin)+vmind; %Formula for hist-stretch
Io = uint8(Im);
