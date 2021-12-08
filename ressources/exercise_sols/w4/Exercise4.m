clc; clear; close all;

%% Exercise 1
f = [zeros(5,3), ones(5,2)]
h = ones(3,3)

g = imfilter(f,h);
g(2,4) % r = 2, c=4 is the center pixel of the filter
% There is 3 zeros and 6 ones covered by the kernel. 3*0+6*1 = 6

%% Exercise 2: Border Handling
im1 = imread('Gaussian.png');
imshow(im1);

fsize = 5;
h = ones(fsize)/fsize^2;
% Odd filter size ensures the kernel can be centered around one pixel

%% Exercise 3
meanim1 = imfilter(im1, h);

figure
subplot(1,2,1);
imshow(im1), colormap gray, axis image off;
title('Original image')
subplot(1,2,2);
imshow(meanim1), colormap gray, axis image off;
title('Filtered image, mean filter')

colormap(gca, jet);

%% Exercise 4: Border replication
meanim2 = imfilter(im1, h, 'replicate');

figure
subplot(1,2,1);
imshow(meanim1), colormap gray, axis image off;
title('Zero-padding')
subplot(1,2,2);
imshow(meanim2), colormap gray, axis image off;
title('Replicate')

% Difference at the border - no dark border around image. 

% The two images are only different along the borders. When filtering
% border pixels, the filter will exceed the image borders and cover
% "non-existing" pixes. These pixels needs to be given a value in order to
% be able to calculate the filtered pixel value. When using zero padding,
% all non-existing pixels are given the value 0, resulting in the dark
% border as described in exercise 3. When using border replication, the
% non-existing pixels are given the value of the nearest actual border 
% pixel. Thus, in the light areas, the pixels outside of the image are
% given high values (corresponding to light colors) and in the dark areas
% these pixels are given low values (corresponding to dark colors). This
% prevents the black border we get when filtering using zero padding. 
%% Noise Reduction 
% Exercise 5:
fsize = 5;
h = ones(fsize)/fsize^2;

meanim1 = imfilter(im1, h,'replicate');

figure
subplot(1,2,1);
imshow(im1), colormap gray, axis image off;
title('Original image')
subplot(1,2,2);
imshow(meanim1), colormap gray, axis image off;
title('Filtered image, mean filter')

% Try zooming in on the dark circle in the middle of both images. Although
% we register this as a dark circle, the circle in the original image
% actually consists of many white/light pixels as well. This is noise. In
% the filtered image we see how the pixel values vary less, i.e. there is
% less noise in this image. This is due to the mean filter functioning as
% a smoothing filter, by giving all pixels the average pixel value of the
% surrounding pixels (and the pixel itself). 

%% Exercise 6: 
fsize = 15;
h = ones(fsize)/fsize^2;

meanim1 = imfilter(im1, h,'replicate');

figure
subplot(1,2,1);
imshow(im1), colormap gray, axis image off;
title('Original image')
subplot(1,2,2);
imshow(meanim1), colormap gray, axis image off;
title('Filtered image, mean filter 15')

% Filtered image gets very blurry - hard to see borders especially when
% stripes are close (lower left corner). 
% Larger regions (ie. middle circle) gets more uniform. 

%% Exercise 8: Median filter
fsize = 5;
medianim1 = medfilt2(im1,[fsize fsize]);

figure
subplot(1,2,1);
imshow(meanim1), colormap gray, axis image off;
title('Filtered image, mean filter')
subplot(1,2,2);
imshow(medianim1), colormap gray, axis image off;
title('Filtered image, median filter')

% Less blurry -> Less subject to outliers. But the regions do not get as
% uniform. 

%% Exercise 9: 
fsize = 15;
medianim2 = medfilt2(im1,[fsize fsize]);

figure
subplot(1,2,1);
imshow(medianim1), colormap gray, axis image off;
title('Filtered image, median filter 5')
subplot(1,2,2);
imshow(medianim2), colormap gray, axis image off;
title('Filtered image, median filter 15')

% Larger kernel makes the image more blurry but regions more uniform. Also
% notice the dark border gets larger. 

% Here it gets more clear, that the mean filter results in more blurry
% edges than the median filter. This is due to the fact that the value
% applied to a pixel when using a median filter is the median value among
% the surrounding pixels. Thus when applying the filter to e.g. a light 
% pixel near an edge, the few dark pixels covered by the kernel will not
% (directly) have an impact on the resulting pixel value. 

%% Exercise 11: Salt and pepper. 
im2 = imread('SaltPepper.png');
figure; imshow(im2);

fsize = 5; 
h = ones(fsize)/fsize^2;
meanim2 = imfilter(im2, h);
medianim2 = medfilt2(im2,[fsize fsize]);

figure
subplot(1,2,1);
imshow(meanim2), colormap gray, axis image off;
title('Filtered image, mean filter 5')
subplot(1,2,2);
imshow(medianim2), colormap gray, axis image off;
title('Filtered image, median filter 5')

% Salt and pepper noise is extreme outliers. The median is more robust to
% extreme observations than the mean. 

% The results could for example be compared using imhist:
figure
subplot(1,3,1);
imhist(im2); 
title('Original image'); 
subplot(1,3,2);
imhist(meanim2); 
title('Filtered image, mean filter (5x5)');
subplot(1,3,3);
imhist(medianim2); 
title('Filtered image, median filter (5x5)');

% The original image contains only three values. When filtered with the
% median kernel, the resulting image consists of the same three pixel
% values, while the image resulting from the mean filter contains many
% different pixel values due to the mean computations. 
%% Standard filter routines
% Exercise 12
hsize = 5; 
h = fspecial('average', hsize)

%% Edge Detection

clc; clear; close all;

I = imread('ElbowCTSlice.png');
imshow(I);

%% Exercise 13:
h = fspecial('sobel');
g= imfilter(I, h);

figure;
imshow(g, []);
colormap(gca, jet);
% Horizontal edges

%% Exercise 14: 
g= imfilter(I, h');
figure;
imshow(g, []);
colormap(gca, jet);
% Vertical edges with highest gradient from the left

g= imfilter(I, -h');
figure;
imshow(g, []);
colormap(gca, jet);
% Vertical edges with highest gradient from the right

%% Exercise 15: 
h = fspecial('sobel');
g= imfilter(I, h);

hmean = fspecial('average',5);
Imean = imfilter(I,hmean);
h = fspecial('sobel');
g5 = imfilter(Imean,h);

hmean = fspecial('average',13);
Imean = imfilter(I,hmean);
h = fspecial('sobel');
g13 = imfilter(Imean,h);

figure
subplot(1,3,1);
imshow(g), colormap gray, axis image off;
title('Edge No filter')
subplot(1,3,2);
imshow(g5), colormap gray, axis image off;
title('Edge mean 5')
subplot(1,3,3);
imshow(g13), colormap gray, axis image off;
title('Edge mean 13')
% The more mean-filtered the less influence from small fluctuations within
% the regions but less well localised edges. 

%% Exercise 16: 
h = fspecial('sobel');
g= imfilter(I, h);

Imedian = medfilt2(I,[5 5]);
h = fspecial('sobel');
g5 = imfilter(Imedian,h);

Imedian = medfilt2(I,[13 13]);
h = fspecial('sobel');
g13 = imfilter(Imedian,h);

figure
subplot(1,3,1);
imshow(g), colormap gray, axis image off;
title('Edge No filter')
subplot(1,3,2);
imshow(g5), colormap gray, axis image off;
title('Edge median 5')
subplot(1,3,3);
imshow(g13), colormap gray, axis image off;
title('Edge median 13')
% Preserved the localisation of the edges while still removing noise in the
% smooth regions. 

%% Exercise 17:
edgeIM = edge(I,'sobel'); % This is how you use it!

figure;
subplot(2,3,1)
imshow(edge(I,'sobel')), colormap gray, axis image off;
title('Sobel')
subplot(2,3,2)
imshow(edge(I,'prewitt')), colormap gray, axis image off;
title('prewitt')
subplot(2,3,3)
imshow(edge(I,'roberts')), colormap gray, axis image off;
title('roberts')
subplot(2,3,4)
imshow(edge(I,'log')), colormap gray, axis image off;
title('log')
subplot(2,3,5)
imshow(edge(I,'zerocross')), colormap gray, axis image off;
title('zerocross')
subplot(2,3,6)
imshow(edge(I,'canny')), colormap gray, axis image off;
title('canny')

% Further specifications: 
% Direction: Horizontal, vertical, both
% Threshold: Return all edges that are stronger than threshold
figure; 
imshow(edge(I,'log',0.01))


%% Gaussian filter
%% Exercise 18
clc; clear; close all;
hsize = 17;
sigma=3;
G = fspecial('gaussian',hsize, sigma);
surf(G);

%% Exercise 19: 
I = imread('ElbowCTSlice.png');
hsize = 17; sigma = 1; 
G1 = fspecial('gaussian',hsize, sigma);
I1 = imfilter(I,G1);

hsize = 17; sigma = 3; 
G2 = fspecial('gaussian',hsize, sigma);
I2 = imfilter(I,G2);

hsize = 51; sigma = 11; 
G3 = fspecial('gaussian',hsize, sigma);
I3 = imfilter(I,G3);

figure; 
subplot(2,3,1)
imshow(I1), colormap gray, axis image off;
title('sigma = 1, hsize = 17')
subplot(2,3,4)
surf(G1)

subplot(2,3,2)
imshow(I2), colormap gray, axis image off;
title('sigma = 3, hsize = 17')
subplot(2,3,5)
surf(G2)

subplot(2,3,3)
imshow(I3), colormap gray, axis image off;
title('sigma = 11, hsize = 51')
subplot(2,3,6)
surf(G3)

% The larger sigma the more blurry.

%% Filtering your own image
clear all; close all; clc

% The image DTUSigns2.jpg from exercise 3 is used.

my1 = imread('DTUSign1.jpg'); 
my2 = rgb2gray(my1); 
my3 = imresize(my2,[1000 NaN]); 

%% Exercise 21
prewitt = fspecial('prewitt'); 
sobel = fspecial('sobel'); 
G = fspecial('gaussian', 17, 3); 
mySobel = imfilter(my3,sobel); 
myPrewitt = imfilter(my3,prewitt);
myEdge = edge(my3); 
myGaussian = imfilter(my3,G); 

figure(1)
subplot(2,2,1)
imshow(mySobel), colormap(gca, gray), axis image off; 
title('Sobel'); 
subplot(2,2,2)
imshow(myPrewitt), colormap(gca, gray), axis image off; 
title('Prewitt'); 
subplot(2,2,3)
imshow(myEdge), colormap(gca, gray), axis image off; 
title('Edge'); 
subplot(2,2,4)
imshow(myGaussian), colormap(gca, gray), axis image off; 
title('Gaussian'); 

