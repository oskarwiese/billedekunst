%% Intro
clear; close all; clc
load Image1.mat
imagesc(Image1);
imagegrid(gca,size(Image1));
colormap(gca,hot);

%% Erosion
se1 = strel('square',3)
se2 = strel('disk',1)

subplot(2,2,1);
imagesc(Image1);
imagegrid(gca,size(Image1));
colormap(hot);
title('Original');

e1 = imerode(Image1, se1);
e2 = imerode(Image1, se2);

subplot(2,2,3)
imagesc(e1)
imagegrid(gca,size(Image1));
colormap(hot);
title('3x3 square erosion')

subplot(2,2,4)
imagesc(e2)
imagegrid(gca,size(Image1));
colormap(hot);
title('3x3 disk erosion')

%% Dilation
subplot(2,2,1);
imagesc(Image1);
imagegrid(gca,size(Image1));
colormap(hot);
title('Original');

d1 = imdilate(Image1, se1);
d2 = imdilate(Image1, se2);

subplot(2,2,3)
imagesc(d1)
imagegrid(gca,size(Image1));
colormap(hot);
title('3x3 square dilation')

subplot(2,2,4)
imagesc(d2)
imagegrid(gca,size(Image1));
colormap(hot);
title('3x3 disk dilation')

%% Opening
load Image1.mat

subplot(2,2,1);
imagesc(Image1);
imagegrid(gca,size(Image1));
colormap(hot);
title('Original');

o1 = imopen(Image1, se1);
%o2 = imopen(Image1, se2);
o2 = mopen(Image1, se1);

subplot(2,2,3)
imagesc(o1)
imagegrid(gca,size(o1));
colormap(hot);
title('3x3 square opening')

subplot(2,2,4)
imagesc(o2)
imagegrid(gca,size(o2));
colormap(hot);
title('3x3 square opening (own function)')

%subplot(2,2,4)
%imagesc(o2)
%imagegrid(gca,size(o2));
%colormap(hot);
%title('3x3 disk opening')

%% Closing
load Image1.mat

subplot(2,2,1);
imagesc(Image1);
imagegrid(gca,size(Image1));
colormap(hot);
title('Original');

c1 = imclose(Image1, se1);
c2 = mclose(Image1, se1);

subplot(2,2,3)
imagesc(c1)
imagegrid(gca,size(c1));
colormap(hot);
title('3x3 square closing')

subplot(2,2,4)
imagesc(c2)
imagegrid(gca,size(c2));
colormap(hot);
title('3x3 square closing (own function)')

% The difference in the borders of the images is due to the different ways
% that Matlab uses padding. Imerode pads with the maximum value in the
% image (1), while imdilate pads with the minimum value in the image (0).
% It was not found how imclose pads, but this variation between the
% functions are due to different padding.
%% Rects
load rects.mat

subplot(2,1,1);
imagesc(rects);
colormap(hot);
title('Original');

se3 = strel('square',9);
o3 = imopen(rects, se3);

subplot(2,1,2)
imagesc(o3)
colormap(hot);
title('9x9 square opening')

%% Brain
clear all,close all,clc;
I = imread('BrainCT.png');
figure;
imshow(I);
%%
% Show the histogram
figure;
imhist(I)
%%
% Remove intensities below 90
I2 = I > 90;
figure;
imshow(I2);
%%
% Define structural elements
se4 = strel('square',9);
se5 = strel('square',5);
se6 = strel('square',3);

Imask = imclose(I2, se4);
I4 = imopen(Imask, se5);
I5 = imerode(I4, se6);
I6 = imdilate(I5,se4);

figure;
subplot(2,2,1);
imshow(I);
title('Original with threshold');

subplot(2,2,2)
imshow(I6)
title('3x3 square erosion, 5x5 square dilation')

subplot(2,2,3)
imshow(Imask)
title('9x9 square closing')

subplot(2,2,4)
imshow(I4)
title('5x5 square opening')

%% Compute boundary with dilation
I_dil = imdilate(Imask,se6);
I_boundary = I_dil - Imask;
figure;
subplot(1,2,1)
imshow(I_boundary);
title('Boundary found with dilation')

% Compute boundary with erosion
I_ero = imerode(Imask,se6);
I_boundary2 = Imask - I_ero;
subplot(1,2,2)
imshow(I_boundary2);
title('Boundary found with erosion')

% When the boundary is found using dilation, the object will appear bigger,
% than when it is found with erosion. It changes since the boundary is
% either found as the outline outside or inside the "real" boundary of the
% bones.