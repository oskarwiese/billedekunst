%% Ex 1
close all; clear; clc;
%Read image
mc = imread('metacarpals.png');

%Check size
size(mc)

%Check contents of workspace
whos

%% Ex 2
%Display image
figure
imshow(mc);

%The bone edges appear lighter in image. This can be explained by the fact
%that more dense bone material is present in those areas of the image, 
%while the bones are less dense in the center of the bone, where it is
%hollow and contains bone marrow. The ends of the the bones primarily
%consist of trabecular (spongy) bone and is therefore less bright than the
%dense bone along the length of the bone.
%Bone absorbs more x-rays than softer or less dense tissue, and thus appear
%very bright on an x-ray image.

%% Ex 3
%Histogram of metacarpal image
figure;
imhist(mc);
ylim([0 6000])
%Most frequent pixel value found is 113

%Histogram counts (how many pixels have a specific value)
[counts,x] = imhist(mc);
%Find idx of most frequent pixel value
idx_freq = find(counts == max(counts))
%Most frequent pixel value is found at idx 114, corresponding to the pixel
%value 113. Because the image values ranges from 0 to 255, while Matlab
%indices are from 1 to 256.

%% Ex 4
%Pixel value at (r,c)=(100,90)
pix_val = mc(100,90); %118

%% Ex 5
imtool(mc);

%Background has lots of pixel values equal to 113

%% Ex 6
close all; clear; clc;
I = imread('horns.jpg');

%Resize image to be 0.25 times the original size
I2 = imresize(I, 1/4);
imshow(I2);

%Calculate scaling factor automatically:
max_dim = max(size(I));
scale_factor = 1000/max_dim %Desired max dimension divided by original max dimension
I3 = imresize(I, scale_factor);
%% Ex 7
%Examine single pixel value (two different ways of obtaining the same pixel
%value). impixel uses (x,y) coordinates. When accessing the pixel from the
%image directly (r,c) coordinate system is used.
impixel(I2, 500, 400)

I2(400, 500, :) %(R,G,B)=(47,52,49)

%Transform to gray-level image:
Igr = rgb2gray(I2);

figure
imshow(Igr);

Igr(400, 500, :) %50

%The gray level value is calculated with the following formula (see rgb2gray):
%0.2989 * R + 0.5870 * G + 0.1140 * B 

%% Ex 8
figure;

imhist(Igr);

[counts,x] = imhist(Igr);

%% Ex 9
%Histograms of very dark images will have most counts in bins at low pixel 
%values (to the left of the histogram), while histograms of very light 
%images will have most counts in bins at high pixel values (to the right of
%the histogram).

clear;clc;
clouds = imread('clouds.jpg');
im_light_gr = rgb2gray(clouds);
figure
imhist(im_light_gr);

horns = imread('horns.jpg');
im_dark_gr = rgb2gray(horns);
figure
imhist(horns);

%% Ex 10
close all; clear; clc;

%DICOM header information
ctInf = dicominfo('CTangio.dcm') %Manufacturer: Toshiba

%Read DICOM image
ct = dicomread('CTangio.dcm');

whos %Pixels are stored as int16

figure;
imshow(ct); %Example with better contrast: imshow(ct, [-500 max(max(ct))])

imtool(ct) 

%% Ex 11
close all; clear; clc;
mc = imread('metacarpals.png');
imshow(mc);
colormap(gca, jet);
colorbar;

%% Colour channels
close all; clear; clc;
im1 = imread('DTUSign1.jpg');
figure;
imshow(im1);

Rcomp = im1(:,:,1);
figure;
imshow(Rcomp);
%colormap(gca, gray);

Gcomp = im1(:,:,2);
figure;
imshow(Gcomp);
%colormap(gca, gray);

Bcomp = im1(:,:,3);
figure;
imshow(Bcomp);
%colormap(gca, gray);

%The sign looks bright on the R-channel image because the red channel has
%high values at the pixels where the sign is, because the sign is red.
%Because the sign is red, the G- and B-channels have low values at the
%sign-pixels and thus appear dark in the images of the G- and B-channels.

%The walls are bright on all channel images, because white has high values
%in both the R,G and B channels. (The walls very bright in the original
%image).

%% Simple image manipulation
close all; clear; clc;
im1 = imread('DTUSign1.jpg');
figure;
imshow(im1);

im1(500:1000,800:1500,:)=0; %Create black rectangle in image (row 500-1000 and column 800-1500)
figure;
imshow(im1);
%imwrite(im1,'DTUSign1-marked.jpg'); %save image

%CReate blue rectangle around DTU Compute sign
im1(1533:1738,2277:2770,3)=255; 
figure;
imshow(im1);
%imwrite(im1,'DTUSign1-blue_rect.jpg'); %save image

%This exercise can be interpreted in different ways. The purpose is not to
%create a perfect rectangle, but to let the students figure out how to
%color some pixels blue in a specific area of the image. In this solution
%the rectangle covers the whole sign, but it could also be made with 4
%boxes - creating a rectangle around the sign and not covering the sign.

%% Advanced image visualization
close all; clear; clc;
fing = imread('finger.png');
imshow(fing);

%%
figure;
imcontour(fing, 5); %Shows 5 levels of equally spaced contour lines
%Noisy background gives rise to contours

%%
%Look at profile across the bone
imshow(fing);
improfile;

%%
%Image plotted as landscape
mesh(double(fing));