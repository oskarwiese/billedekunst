%% Ex 1 & 2
set(0,'defaultaxesfontsize',15);
clear all, close all,clc
load Image1.mat

subplot(1,3,1)
imagesc(Image1);
imagegrid(gca,size(Image1));
colormap(hot);
title('Original image')

subplot(1,3,2);
% 4 connectiviy
L4 = bwlabel(Image1,4);
imagesc(L4);
imagegrid(gca,size(L4));
colormap(hot);
title('4 connectiviy')

subplot(1,3,3);
% 8 connectiviy
L8 = bwlabel(Image1,8);
imagesc(L8);
imagegrid(gca,size(L8));
colormap(hot);
title('8 connectiviy')
% The difference between 4 and 8 connectivity is that regions where the
% cornors of pixels touch are the same region in 8 connectivity but not in
% 4.
%% Ex 3
RGB4 = label2rgb(L4);
RGB8 = label2rgb(L8);

%RGB4 = label2rgb(L4, 'spring', 'c', 'shuffle');
%RGB8 = label2rgb(L8, 'spring', 'c', 'shuffle');

subplot(1,3,1)
imagesc(label2rgb(Image1));
imagegrid(gca,size(Image1));
colormap(hot);
title('Original image')

subplot(1,3,2);
% 4 connectiviy
imagesc(RGB4);
imagegrid(gca,size(RGB4));
title('4 connectiviy')

subplot(1,3,3);
% 8 connectiviy
imagesc(RGB8);
imagegrid(gca,size(RGB8));
title('8 connectiviy')

%% Ex 4 & 5 
 stats8 = regionprops(L8, 'Area');
 val1= stats8(1).Area
 val2= stats8(2).Area
 val3= stats8(3).Area
%% Ex 6
allArea = [stats8.Area]
%% Ex 7
idx = find([stats8.Area] > 16);
BW2 = ismember(L8,idx);
 
imagesc(BW2);
imagegrid(gca,size(BW2));
colormap(hot);

%% Ex 8
 stats8 = regionprops(L8, 'All');
 
allPerimeter = [stats8.Perimeter]
perimeter_20 = sum(allPerimeter>20)
plot(allArea, allPerimeter, '*'), xlabel('Area'), ylabel('Perimeter');


%% Chemometec U2OS cell analysis - raw images
clear all,close all,clc;
%I16 = imread('CellData\Sample G1 - COS7 cells DAPI channel.tiff');
%I16c = imcrop(I16, [900 900 500 500]);
%I16c = imcrop(I16, [300 0 500 500]); % Can only be handled with both size and area
%I16c = imcrop(I16, [0 700 500 500]); % Can only be handled with both size and area

I16 = imread('CellData\Sample E2 - U2OS DAPI channel.tiff');
I16c = imcrop(I16, [500 700 500 500]);
%I16c = imcrop(I16, [700 900 500 500]);
%I16c = imcrop(I16, [0 0 500 500]);

%Im = imread('CellData\Sample E2 - U2OS DAPI channel_selection.tif');
Im = im2uint8(I16c);
%min(min(Im))
%max(max(Im))
%C=unique(Im);
%size(C)
%imshow(Im, [0 10000])

%imhist(Im)
%% Ex 12
imshow(Im, [0 150]); title('DAPI Stained U2OS cell nuclei');
%imshow(Im, [0 150]); title('DAPI Stained COS7 cell nuclei');
%% Ex 13
imhist(Im)
% 10 is choosen, zoom in to see the values in the histogram
BW = (Im > 10);
figure, imshow(BW); title('Thresholded image');
%% Ex 14
BWc = imclearborder(BW);
figure, imshow(BWc); title('Thresholded image - border cells removed');
%% Ex 15 
se = strel('disk',3);        
BWe = imopen(BWc,se);
L = bwlabel(BWe,8);
L1 = label2rgb(L);
subplot(1,3,1),
imshow(BWc), title('Original image - border cells removed')
subplot(1,3,2),
imshow(BWe), title('Opened image');
subplot(1,3,3)
imagesc(L1); axis image; title('Regions labeled with RGB colors');
%% Ex 16

cellStats = regionprops(L, 'All');
 
cellPerimeter = [cellStats.Perimeter];
cellArea = [cellStats.Area];

figure, plot(cellPerimeter, cellArea, '.');  xlabel('Perimeter'); ylabel('Area');

figure, hist(cellArea); title('Cell Area Histogram');
%% Ex 17


idx = find([cellStats.Area] > 200);
BW2 = ismember(L,idx);
figure, imagesc(BW2); axis image; title('Object with area > 200');

idx = find([cellStats.Area] < 200);
BW2 = ismember(L,idx);
figure, imagesc(BW2); axis image; title('Object with area < 200');
countA = numel(idx);

minArea = 20;
maxArea = 200;
idx = find([cellStats.Area] < maxArea & [cellStats.Area] > minArea );
BW2 = ismember(L,idx);
figure, imagesc(BW2); axis image; title('Object with area < 200 and area > 20');
countA2 = numel(idx);
%% Ex 18

circularity =  (4 * pi * [cellStats.Area]) ./ ([cellStats.Perimeter].^2);
hist(circularity);
idx = find([circularity] > 0.9);
BW2 = ismember(L,idx);
figure, imagesc(BW2); axis image; title('Circularity > 0.9')
countC = numel(idx);
%% Ex 19
idx = find([circularity] > 0.9 & [cellStats.Area] < 200 & [cellStats.Area] > 50);
BW2 = ismember(L,idx);
countCA = numel(idx);
tit = sprintf('Circularity and Area filtered : %i cells',countCA);
figure, imagesc(BW2); axis image; title(tit)


% Cell counts
countA
countC
countCA

%% Ex 20
% Test CountCellNuclei.m
[I, N] = CountCellNuclei(Im);
sprintf('CountCellNuclei says %i cells ',N);
tit = sprintf('CountCellNuclei output with %i cells',N);
figure, imagesc(I); axis image; title(tit)


%% Chemometec U2OS cell analysis
% same procedure but another image
clear all,close all,clc;
addpath CellDataWorking
Im = imread('Sample E2 - U2OS DAPI channel_selection2_8bit.png');
imshow(Im); title('Original image');

%% Ex 24
se = strel('disk',3);        
BWe = imopen(Im,se);
imshow(BWe)
BW = (BWe > 75);
figure;
imshow(BW), title('Thresholded (grayscale) image')
%%
L = bwlabel(BW,8);
L1 = label2rgb(L);
figure, imagesc(L1); axis image; title('Regions labeled with RGB colors');
%%
cellStats = regionprops(L, 'All');
 
cellPerimeter = [cellStats.Perimeter]
cellArea = [cellStats.Area];

plot(cellPerimeter, cellArea, '.');

sort(cellArea);
hist(cellArea);


%% Identify outliers
idx = find([cellStats.Area] > 200);
BW2 = ismember(L,idx);
figure, imagesc(BW2); axis image; title('Object with area > 200');

idx = find([cellStats.Area] < 200);
BW2 = ismember(L,idx);
figure, imagesc(BW2); axis image; title('Object with area < 200');
%%
cellEcc = [cellStats.Eccentricity];
hist(cellEcc);

idx = find([cellStats.Eccentricity] > 0.70);
BW2 = ismember(L,idx);
figure, imagesc(BW2); axis image; title('Eccentricity > 0.70')

cellShape = [cellStats.Perimeter] ./ [cellStats.Area];
hist(cellShape);

idx = find([cellShape] < 0.20);
BW2 = ismember(L,idx);
figure, imagesc(BW2); axis image; title('Perimeter / Area < 0.20')
%% Ex 24
se = strel('disk',3);        
BWe = imopen(BWc,se);