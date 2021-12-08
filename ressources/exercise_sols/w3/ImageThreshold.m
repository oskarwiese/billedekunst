function Thres = ImageThreshold(Im,T)
% Function creates a treshold image
% Input:
%   Im = image you want to treshold
%   T = The treshold value
% Output:
%   Tres = treshold image

[m,n] = size(Im);   %First we define the size of the image

% Now we loop over number of rows and columns and change each pixel to
% either 1 or 0 in correlation to the T-value:
for i = 1:m 
    for j = 1:n
        if Im(i,j) <= T
            Im(i,j) = 0;
        else 
            Im(i,j) = 1;
        end
    end
end
Thres = Im*255;  % Important to multiply back up so the range is [0,255]


