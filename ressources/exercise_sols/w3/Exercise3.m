% Clear Everything first
close all; clear; clc;

%% Initial Analysis
vb = imread('vertebra.png');
figure;
subplot(1,2,1),
imshow(vb);
subplot(1,2,2),
imhist(vb);
% The image histogram is indeed a bimodal histogram with
% peaks in 70 and 233.
% The vertebrae can be made out in the neck area but below
% they are blended in with the rest of the body.
% The skull itself is also quite uniform in intensity.

vbmin = min(vb(:)); % The minimum intensity in the image
% is 57. 
vbmax = max(vb(:)); % The maximum intensity in the image
% is 235.

%% Exercise 1

vb = imread('vertebra.png');
% Mean computes a 1-D mean of a vector while
% mean2 computes a 2-D mean of an entire array.

vbmean = mean2(vb); % The mean is 156.17 = 156.

%% Exercise 2

vb = imread('vertebra.png');
% Use the histogram stretch function on the image
Io = HistStretch(vb);

% HISTSTRETCH.M CODE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function Io = HistStretch(I)
% % Stretches an image histogram to improve contrast
% Itemp = double(I);
% 
% % Stretch the image intensities to span the entire
% % spectrum:
% vmax_d = 255;
% vmin_d = 0;
% 
% % Get vmax and vmin
% vmin = min(Itemp(:));
% vmax = max(Itemp(:));
% 
% Itemp = ((vmax_d - vmin_d) / (vmax - vmin)) * (Itemp - vmin) + vmin_d;
% Io = uint8(Itemp);
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The contrast increases and the bone structure becomes
% more accentuated.
figure;
subplot(1,2,1), title('Original')
imshow(vb)
subplot(1,2,2), title('Stretched')
imshow(Io)

%% Exercise 3

% Choose an interval to plot gamma in, here from
% 0 to 1 in intervals of 0.001
interval = (0:0.001:1);
gam = [0.48 1 1.52];

figure;
hold on;
grid on;
plot(interval,interval.^gam(1));
plot(interval,interval.^gam(2));
plot(interval,interval.^gam(3));
xlabel('Value in')
ylabel('Value out')
legend('\gamma = 0.48','\gamma = 1', '\gamma = 1.52');
clear interval

%% Exercise 4

% Apply the gamma values from exercise 3 to the image
% and display side-by-side

% GAMMAMAP.M CODE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function GammaIm = GammaMap(I, gamma)
% % Creates a gamma-mapped image with the
% % desired gamma value
% 
% % Convert image to double
% Itemp = double(I);
% 
% % Scale to [0 1] by dividing each element by 255
% Itemp = Itemp./255;
% 
% % Apply the gamma mapping
% Itemp = Itemp.^gamma;
% 
% %Scale back to [0 255] and convert to uint8
% GammaIm = uint8(255.*Itemp);
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vb = imread('vertebra.png');
gam = [0.48 1 1.52];

IGammaLow = GammaMap(vb,gam(1));
IGammaNone = GammaMap(vb,gam(2));
IGammaHigh = GammaMap(vb,gam(3));
figure;
subplot(2,3,1)
imshow(IGammaLow)
ylabel('Unstretched Hist.')
subplot(2,3,2)
imshow(IGammaNone)
subplot(2,3,3)
imshow(IGammaHigh)

% In this case, we just divide and multiply by 255
% to get the image in the desired range. If we instead
% use a linear mapping method with vmax_d = 1 and vmin_d = 0
% this would also stretch the histogram of the image to 
% improve contrast. We can stretch the histogram before 
% passing it to GammaMap to see the difference

Io = HistStretch(vb);
IGammaLow = GammaMap(Io,gam(1));
IGammaNone = GammaMap(Io,gam(2));
IGammaHigh = GammaMap(Io,gam(3));

subplot(2,3,4)
imshow(IGammaLow)
title('\gamma=0.48')
ylabel('Stretched Hist.')
subplot(2,3,5)
imshow(IGammaNone)
title('\gamma=1')
subplot(2,3,6)
imshow(IGammaHigh)
title('\gamma=1.52')

% It can be seen on all these 6 images that the
% stretched histogram vertebrae image with a gamma of 1.52
% seems to provide the best visibility of the bones while
% a gamma of 0.48 washes out the bone structure.

clear IGammaHigh IGammaLow IGammaNone

%% Exercise 5

% doc imadjust
% The following values seem to increase the contrast
% on regions of interest in the image by mapping the
% background values to black and saturating the max
% intensity at 0.9*255 = 230

vb = imread('vertebra.png');
vbAdjusted = imadjust(vb,[0.5, 0.9]);

figure;
imshow(vbAdjusted)

%% Exercise 6

% ImageThreshold.M CODE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function imBin = ImageThreshold(im, T)
% % Thresholds an input image with threshold T and
% % returns a binary image imBin
% 
% % Pre-allocate the binary image
% imBin = false(size(im));
% 
% % Go through each pixel and check if it is
% % above or below the threshold
% for m = 1:size(im,1)
%      for n = 1:size(im,2)
%          if im(m,n) < T
%              imBin(m,n) = 0;
%          else
%              imBin(m,n) = 1;
%          end
%      end
%  end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vb = imread('vertebra.png');
T = 175;
vbThreshold = ImageThreshold(vb, T);
vbAdjusted = imadjust(vb,[0.5, 0.9]);
vbThreshold_adj = ImageThreshold(vb, T);
figure;
subplot(1,3,1)
imshow(vb)
subplot(1,3,2)
imshow(vbThreshold), title('non-adjusted')
subplot(1,3,3)
imshow(vbThreshold_adj), title('adjusted')

% With the unadjusted vertebra image it is impossible
% to seperate the vertebrae as they blend in with
% the skin in the X-ray. Using the adjusted image makes
% it slightly better but still does not isolate the
% vertebrae

%% Exercise 7

% doc graythresh
% Graythresh automatically finds a threshold level
% between 0 and 1 that is deemed optimal by Otsu's
% method

vb = imread('vertebra.png');
T = 175;
vbThreshold = ImageThreshold(vb, T);

level = graythresh(vb)
imGraythresh = ImageThreshold(vb,255*level);

figure;
subplot(1,3,1)
imshow(vb)
subplot(1,3,2)
imshow(vbThreshold), title('Threshold = 175')
subplot(1,3,3)
imshow(imGraythresh), title(['Auto Threshold = ',num2str(255*level)])

% The found threshold is 255*0.5804 = 148 which
% makes a nice silhouette of the patient but
% completely removes bone structure.

figure;
imhist(vb)

% The algorithm is likely good for segmentation between
% foreground and background based on the histogram but tries
% to minimize variance in foreground and background pixels 
% which creates the silhouette of the patient.

%% Exercise 8
clc; clear;

% Read the image and save it as grayscale image
im = rgb2gray(imread('dark_background.png'));

% Create an automatically thresholded image
imGraythresh = ImageThreshold(im,graythresh(im)*255);

% Try to find your own threshold
T = 5;
imGrayCustom = ImageThreshold(im, T);

figure;
subplot(1,3,1)
imshow(im)
title('Gray')
subplot(1,3,2)
imshow(imGraythresh)
title('Automatic Thresh.')
subplot(1,3,3)
imshow(imGrayCustom)
title('Custom Thresh.')

%% Exercise 9

vb = imread('vertebra.png');
imtool(vb)

%% Color Thresholding in RGB space

clear; clc;
im = imread('DTUSigns2.jpg');
figure;
imshow(im);

Rcomp = im(:,:,1);
Gcomp = im(:,:,2);
Bcomp = im(:,:,3);

% Segmenting out the blue sign by checking if the RGB 
% color falls within limits below
segm = Rcomp < 10 & Gcomp > 85 & Gcomp < 105 & Bcomp >...
    180 & Bcomp < 200;
figure;
imshow(segm);

% By using the data cursor, it appears that the RGB composition
% of the DTU sign roughly falls within:
% R: 160 - 175
% G: 45 - 65
% B: 45 - 75

segmDTU = Rcomp > 160 & Rcomp < 175 & Gcomp > 45 & Gcomp...
    < 65 & Bcomp > 45 & Bcomp < 75;

figure;
imshow(segmDTU)

clear Rcomp Gcomp Bcomp segm segmDTU
%% Color Thresholding in HSI/HSV space

im = imread('DTUSigns2.jpg');
HSV = rgb2hsv(im);
figure;
imshow(HSV)
Hcomp = HSV(:,:,1);
Scomp = HSV(:,:,2);
Vcomp = HSV(:,:,3);
figure;
subplot(1,3,1)
imshow(Hcomp)
title('H comp.')
subplot(1,3,2)
imshow(Scomp)
title('S comp.')
subplot(1,3,3)
imshow(Vcomp)
title('Vcomp')

% By inspecting the H-, S-, and V-compositions of the 
% image it is possible to make a single segmentation
% that segments both the blue sign and the red DTU sign.
% The limits are:
% H: 0.45 - 1
% S: 0.6 - 1
% V: 0.58 - 0.83

segm = Hcomp > 0.45 & Scomp > 0.6 & Vcomp > 0.58 & Vcomp < 0.83;
figure;
imshow(segm)