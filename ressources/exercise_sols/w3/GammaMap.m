function gIm = GammaMap(Im,gamma)
% Function that takes an image an creates its gamma map
% Input:
%   Im = image you want to gamme map
%   gamma = the gamme value
% Output:
%   gIm = a graph of the gamma map

Itemp = double(Im); %First we convert the image to double
vmax = 255; % Max pixel value

NyIm = (Itemp/vmax);    % Convertes down to pixelvalues between [0,1]

NyIm2 = NyIm.^gamma;   %The graph is computed

ko = 255*(NyIm2);   %We convert the gamma map back

gIm = uint8(ko); % Now it has to be converted back to integer data type
imshow(gIm);


