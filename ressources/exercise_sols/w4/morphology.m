%% Intro
clear,close all,clc
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
%o2 = imopen(Image1, se2);
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

%% Rects
load rects.mat

subplot(2,2,1);
imagesc(rects);
colormap(hot);
title('Original');

se3 = strel('square',9);
o3 = imopen(rects, se3);

subplot(2,2,3)
imagesc(o3)
colormap(hot);
title('9x9 square opening')


%% Brain
clear all,close all,clc;
I = imread('BrainCT.png');
imshow(I);

I2 = I > 90;
imshow(I2);

se4 = strel('square',9);
se5 = strel('square',5);

I3 = imclose(I2, se4);

figure;
imshow(I3);

I4 = imopen(I3, se5);
figure;
imshow(I4);

se6 = strel('square',3);

I5 = imerode(I4, se6);

I6 = I4-I5;
figure;
imshow(I6);



