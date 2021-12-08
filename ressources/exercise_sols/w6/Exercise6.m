% Initial commands and loading
clc; close all; clear all;
ct2 = dicomread('CTangio2.dcm');
I2 = imread('CTAngio2Scaled.png');
imshow(I2);

%% Exercise 1

% Use attached LiverROI.png/FatROI.png/... images
% or annotate by hand with the code below:

%%%%%%%%%% FOR FIRST RUN THROUGH %%%%%%%%%%
%LiverROI = roipoly(I2);
%imwrite(LiverROI, 'LiverROI.png');
%LiverVals = double(ct2(LiverROI));

    % Two kidneys so we make two ROIs and 
    % do an 'OR' operation on them
%KidneyROI1 = roipoly(I2);
%KidneyROI2 = roipoly(I2);
%KidneyROI = KidneyROI1 | KidneyROI2;
%imwrite(KidneyROI, 'KidneyROI.png');
%KidneyVals = double(ct2(KidneyROI));

%SpleenROI = roipoly(I2);
%imwrite(SpleenROI, 'SpleenROI.png');
%SpleenVals = double(ct2(SpleenROI));

%TrabROI = roipoly(I2);
%imwrite(TrabROI, 'TrabROI.png');
%TrabVals = double(ct2(TrabROI));

%BoneROI = roipoly(I2);
%imwrite(BoneROI, 'BoneROI.png');
%BoneVals = double(ct2(BoneROI));

%BackgroundROI = roipoly(I2);
%imwrite(BackgroundROI, 'BackgroundROI.png');
%BackgroundVals = double(ct2(BackgroundROI));

%FatROI = roipoly(I2);
%imwrite(FatROI, 'FatROI.png');
%FatVals = double(ct2(FatROI));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

LiverROI = imread('LiverROI.png');
LiverVals = double(ct2(LiverROI));

KidneyROI = imread('KidneyROI.png');
KidneyVals = double(ct2(KidneyROI));

SpleenROI = imread('SpleenROI.png');
SpleenVals = double(ct2(SpleenROI));

TrabROI = imread('TrabROI.png');
TrabVals = double(ct2(TrabROI));

BoneROI = imread('BoneROI.png');
BoneVals = double(ct2(BoneROI));

BackgroundROI = imread('BackgroundROI.png');
BackgroundVals = double(ct2(BackgroundROI));

FatROI = imread('FatROI.png');
FatVals = double(ct2(FatROI));

%% Exercise 2
% Plotting histograms with one bin for each HU
% value:
figure;
subplot(3,3,1)
hist(LiverVals,length(LiverVals));
title('Liver')

subplot(3,3,2)
hist(KidneyVals,length(KidneyVals));
title('Kidney')

subplot(3,3,3)
hist(SpleenVals,length(SpleenVals));
title('Spleen')

subplot(3,3,4)
hist(TrabVals,length(TrabVals));
title('Trabecular')

subplot(3,3,5)
hist(BoneVals,length(BoneVals));
title('Hard Bone')

subplot(3,3,6)
hist(BackgroundVals,length(BackgroundVals));
title('Background')

subplot(3,3,7)
hist(FatVals,length(FatVals));
title('Fat')

% Of these plots, it appears that the liver,
% kidney, and spleen all have a gaussian distribution.
% Fat can be included too though it does have a slight
% skew.

% Getting the basic statistics of the classifications:
sprintf('Liver mean %g, std %g, min %g, max %d',...
    mean(LiverVals), std(LiverVals), ...
    min(LiverVals), max(LiverVals))

sprintf('Kidney mean %g, std %g, min %g, max %d',...
    mean(KidneyVals), std(KidneyVals), ...
    min(KidneyVals), max(KidneyVals))

sprintf('Spleen mean %g, std %g, min %g, max %d',...
    mean(SpleenVals), std(SpleenVals), ...
    min(SpleenVals), max(SpleenVals))

sprintf('Trabecular mean %g, std %g, min %g, max %d',...
    mean(TrabVals), std(TrabVals), ...
    min(TrabVals), max(TrabVals))

sprintf('Hard bone mean %g, std %g, min %g, max %d',...
    mean(BoneVals), std(BoneVals), ...
    min(BoneVals), max(BoneVals))

sprintf('Background mean %g, std %g, min %g, max %d',...
    mean(BackgroundVals), std(BackgroundVals), ...
    min(BackgroundVals), max(BackgroundVals))

sprintf('Fat mean %g, std %g, min %g, max %d',...
    mean(FatVals), std(FatVals), ...
    min(FatVals), max(FatVals))

%% Exercise 3
figure;
xrange = -1200:0.1:1200; % Fit over the complete Hounsfield range
pdfFitLiver = normpdf(xrange, mean(LiverVals), std(LiverVals));
S = length(LiverVals); % Scale factor

hold on;
hist(LiverVals, xrange);
plot(xrange, pdfFitLiver * S, 'r');
hold off;
xlim([-10, 100]);

%% Exercise 4

% Creating PDFs for the values:
xrange = -1200:0.1:1200; % Fit over the complete Hounsfield range
pdfFitLiver = normpdf(xrange, mean(LiverVals), std(LiverVals));
Sliver = length(LiverVals); % Scale factor

pdfFitKidney = normpdf(xrange, mean(KidneyVals), std(KidneyVals));
Skidney = length(KidneyVals); % Scale factor

pdfFitSpleen = normpdf(xrange, mean(SpleenVals), std(SpleenVals));
Sspleen = length(SpleenVals); % Scale factor

pdfFitTrab = normpdf(xrange, mean(TrabVals), std(TrabVals));
Strab = length(TrabVals); % Scale factor

pdfFitBone = normpdf(xrange, mean(BoneVals), std(BoneVals));
Sbone = length(BoneVals); % Scale factor

pdfFitBackground = normpdf(xrange, mean(BackgroundVals), std(BackgroundVals));
Sbackground = length(BackgroundVals); % Scale factor

pdfFitFat = normpdf(xrange, mean(FatVals), std(FatVals));
Sfat = length(FatVals); % Scale factor

figure;
plot(xrange, pdfFitLiver * Sliver, xrange, pdfFitKidney * Skidney, ...
    xrange, pdfFitSpleen * Sspleen, xrange, pdfFitTrab * Strab, ...
    xrange, pdfFitBone * Sbone, xrange, pdfFitBackground * Sbackground, ...
    xrange, pdfFitFat * Sfat);

legend('Liver','Kidney','Spleen','Trabecular','Hard bone', ...
    'Background','Fat');

% It appears that background, fat, trab. bone, and hard bone
% can be identified. Kidney and spleen overlap a lot with liver. 
% As such, it is likely a good idea to collapse liver, spleen,
% and kidney into a "Soft Tissue" category.

%% Exercise 5
% If we assume that all pixel classes have a gaussian distribution,
% we can use the means of each class to determine the 
% classification.

T1 = (mean(BackgroundVals) + mean(FatVals)) / 2;
T2 = (mean(FatVals) + mean(LiverVals)) / 2;
T3 = (mean(LiverVals) + mean(TrabVals)) / 2;
T4 = (mean(TrabVals) + mean(BoneVals)) / 2;

TableRange = [-255,T1,T2,T3,T4 ; T1,T2,T3,T4,255]'

%% Exercise 6
% Use the LabelImage function which goes through
% each pixel in the image and assigns it to a class
% depending on the value of the pixel in focus. If
% the pixel value is less than T1, the pixel is 
% assigned class 0 (background), if the pixel value
% is more than T1 but less than T2, it is assigned 
% class 1 (fat), etc. Function needs 6 inputs in total.
ILabel = LabelImage(ct2, T1, T2, T3, T4, T4);

%% Exercise 7
figure
imagesc(ILabel)
hcb = colorbar;
set(hcb, 'YTick',[0, 1, 2, 3, 5]);
set(hcb, 'YTickLabel', {'Background', 'Fat', 'Soft Tissue', ...
    'Trab. Bone', 'Hard Bone'});

%% Exercise 8
% Looking at the plot of the PDFs again:
figure;
plot(xrange, pdfFitLiver * Sliver, xrange, pdfFitKidney * Skidney, ...
    xrange, pdfFitSpleen * Sspleen, xrange, pdfFitTrab * Strab, ...
    xrange, pdfFitBone * Sbone, xrange, pdfFitBackground * Sbackground, ...
    xrange, pdfFitFat * Sfat);

legend('Liver','Kidney','Spleen','Trabecular','Hard bone', ...
    'Background','Fat');

% We can see that:
% Intersection - Background/Fat: -161.2
% Intersection - Fat/Soft tissue: -15.8
% Intersection - Soft tissue/Trab. bone: 97.4
% Intersection - Trab. bone/Hard bone: 241.9

% Setting this as limits:
T1 = -161.2;
T2 = -15.8;
T3 = 97.4;
T4 = 241.9;

%% Exercise 9
% Use the LabelImage function to 
ILabelParam = LabelImage(ct2, T1, T2, T3, T4, T4);

figure
subplot(1,2,1)
imagesc(ILabel)
hcb = colorbar;
set(hcb, 'YTick',[0, 1, 2, 3, 4]);
set(hcb, 'YTickLabel', {'Background', 'Fat', 'Soft Tissue', ...
    'Trab. Bone', 'Hard Bone'});
title('Minimum Distance Classification')

subplot(1,2,2)
imagesc(ILabelParam)
hcb = colorbar;
set(hcb, 'YTick',[0, 1, 2, 3, 4]);
set(hcb, 'YTickLabel', {'Background', 'Fat', 'Soft Tissue', ...
    'Trab. Bone', 'Hard Bone'});
title('Parametric Classification')

% The images are quite similar. However, some parts inside the
% body is classified as background in the parametric 
% classification which, going by color, is likely more correct.
% Also, the parametric classification finds more hard bone 
% than the minimum distance classification.

%% Exercise 10
clc; clear; close all;
% We define the following classes:
% Class 1: Blue sign
% Class 2: Red sign
% Class 3: White car
% Class 4: Green leaves
% Class 5: Yellow grass

I = imread('DTUSigns055.jpg');
figure;
imshow(I);
Ired = I(:,:,1);
Igreen = I(:,:,2);
Iblue = I(:,:,3);

% Use attached BSignROI.png/RSignROI.png ... images
% or annotate by hand with the code below:

%%%%%%%%%% FOR FIRST RUN THROUGH %%%%%%%%%%
%BSignROI = roipoly(I);
%imwrite(BSignROI, 'BSignROI.png');

%RSignROI = roipoly(I);
%imwrite(RSignROI, 'RSignROI.png');

%WCarROI = roipoly(I);
%imwrite(WCarROI, 'WCarROI.png');

%GLeavesROI = roipoly(I);
%imwrite(GLeavesROI, 'GLeavesROI.png');

%YGrassROI = roipoly(I);
%imwrite(YGrassROI, 'YGrassROI.png');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Loading the ROIs for the classes
BSignROI = imread('BSignROI.png');
RSignROI = imread('RSignROI.png');
WCarROI = imread('WCarROI.png');
GLeavesROI = imread('GLeavesROI.png');
YGrassROI = imread('YGrassROI.png');

%% Exercise 11
% Here, a histogram inspection and some manual trial-and-error
% is done for each class:

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Blue sign:
redVals = double(Ired(BSignROI));
greenVals = double(Igreen(BSignROI));
blueVals = double(Iblue(BSignROI));


figure;
totVals = [redVals greenVals blueVals];
nbins = 255;
hist(totVals,nbins);
h = findobj(gca,'Type','patch');
set(h(3),'FaceColor','r','EdgeColor','r','FaceAlpha',0.3,'EdgeAlpha',0.3);
set(h(2),'FaceColor','g','EdgeColor','g','FaceAlpha',0.3,'EdgeAlpha',0.3);
set(h(1),'FaceColor','b','EdgeColor','b','FaceAlpha',0.3,'EdgeAlpha',0.3);
xlim([0 255]);

Rlo = 0;
Rhi = 7;
Glo = 87;
Ghi = 107;
Blo = 179;
Bhi = 197;

figure;
BlueSign = Ired > Rlo & Ired < Rhi & Igreen > Glo & ...
Igreen < Ghi & Iblue > Blo & Iblue < Bhi;
imshow(BlueSign);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Red sign:
redVals = double(Ired(RSignROI));
greenVals = double(Igreen(RSignROI));
blueVals = double(Iblue(RSignROI));


figure;
totVals = [redVals greenVals blueVals];
nbins = 255;
hist(totVals,nbins);
h = findobj(gca,'Type','patch');
set(h(3),'FaceColor','r','EdgeColor','r','FaceAlpha',0.3,'EdgeAlpha',0.3);
set(h(2),'FaceColor','g','EdgeColor','g','FaceAlpha',0.3,'EdgeAlpha',0.3);
set(h(1),'FaceColor','b','EdgeColor','b','FaceAlpha',0.3,'EdgeAlpha',0.3);
xlim([0 255]);

Rlo = 130;
Rhi = 178;
Glo = 30;
Ghi = 66;
Blo = 35;
Bhi = 72;

figure;
RedSign = Ired > Rlo & Ired < Rhi & Igreen > Glo & ...
Igreen < Ghi & Iblue > Blo & Iblue < Bhi;
imshow(RedSign);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% White car:
redVals = double(Ired(WCarROI));
greenVals = double(Igreen(WCarROI));
blueVals = double(Iblue(WCarROI));


figure;
totVals = [redVals greenVals blueVals];
nbins = 255;
hist(totVals,nbins);
h = findobj(gca,'Type','patch');
set(h(3),'FaceColor','r','EdgeColor','r','FaceAlpha',0.3,'EdgeAlpha',0.3);
set(h(2),'FaceColor','g','EdgeColor','g','FaceAlpha',0.3,'EdgeAlpha',0.3);
set(h(1),'FaceColor','b','EdgeColor','b','FaceAlpha',0.3,'EdgeAlpha',0.3);
xlim([0 255]);

Rlo = 103;
Rhi = 158;
Glo = 106;
Ghi = 161;
Blo = 105;
Bhi = 161;

figure;
WhiteCar = Ired > Rlo & Ired < Rhi & Igreen > Glo & ...
Igreen < Ghi & Iblue > Blo & Iblue < Bhi;
imshow(WhiteCar);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Green leaves:
redVals = double(Ired(GLeavesROI));
greenVals = double(Igreen(GLeavesROI));
blueVals = double(Iblue(GLeavesROI));


figure;
totVals = [redVals greenVals blueVals];
nbins = 255;
hist(totVals,nbins);
h = findobj(gca,'Type','patch');
set(h(3),'FaceColor','r','EdgeColor','r','FaceAlpha',0.3,'EdgeAlpha',0.3);
set(h(2),'FaceColor','g','EdgeColor','g','FaceAlpha',0.3,'EdgeAlpha',0.3);
set(h(1),'FaceColor','b','EdgeColor','b','FaceAlpha',0.3,'EdgeAlpha',0.3);
xlim([0 255]);

Rlo = 18;
Rhi = 68;
Glo = 24;
Ghi = 72;
Blo = 15;
Bhi = 40;

figure;
GreenLeaves = Ired > Rlo & Ired < Rhi & Igreen > Glo & ...
Igreen < Ghi & Iblue > Blo & Iblue < Bhi;
imshow(GreenLeaves);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Yellow grass:
redVals = double(Ired(YGrassROI));
greenVals = double(Igreen(YGrassROI));
blueVals = double(Iblue(YGrassROI));


figure;
totVals = [redVals greenVals blueVals];
nbins = 255;
hist(totVals,nbins);
h = findobj(gca,'Type','patch');
set(h(3),'FaceColor','r','EdgeColor','r','FaceAlpha',0.3,'EdgeAlpha',0.3);
set(h(2),'FaceColor','g','EdgeColor','g','FaceAlpha',0.3,'EdgeAlpha',0.3);
set(h(1),'FaceColor','b','EdgeColor','b','FaceAlpha',0.3,'EdgeAlpha',0.3);
xlim([0 255]);

Rlo = 154;
Rhi = 202;
Glo = 131;
Ghi = 176;
Blo = 87;
Bhi = 130;

figure;
YellowGrass = Ired > Rlo & Ired < Rhi & Igreen > Glo & ...
Igreen < Ghi & Iblue > Blo & Iblue < Bhi;
imshow(YellowGrass);

%% Exercise 12
clc; close all;
figure;
subplot(2,3,1)
imshow(BlueSign);
title('Blue Sign')

subplot(2,3,2)
imshow(RedSign);
title('Red Sign')

subplot(2,3,3)
imshow(WhiteCar);
title('White Car')

subplot(2,3,4)
imshow(GreenLeaves);
title('Green Leaves')

subplot(2,3,5)
imshow(YellowGrass);
title('Yellow Grass')

% It can be seen that overall, the segmentations does
% find at least some of their intended objects. The 
% classifications that work the best are the sign 
% classifications. The grass and leaves classifications
% does also find some of the grasses and leaves but 
% because they are large patches that vary a lot in color
% based on lighting conditions, they are not completely
% classified. Also, some false positives are found here 
% as well. Lastly, the white car classification is the
% hardest type as white of course contains a mix of every
% color, meaning that trying to classify a white car 
% will also end up classifying a lot of other objects 
% wrongly.
% It is of course possible to improve on the 
% classification range limits to get a better 
% classification of objects.

%% Exercise 13 + 14
clc; clear; close all;
% Trying the limits with an image from a former exercise:
I = imread('DTUSigns055.jpg');
% This image contains a red sign and some green leaves
% so this is what we will try to classify
Ired = I(:,:,1);
Igreen = I(:,:,2);
Iblue = I(:,:,3);

Rlo = 18;
Rhi = 68;
Glo = 24;
Ghi = 72;
Blo = 15;
Bhi = 40;

figure;
GreenLeaves = Ired > Rlo & Ired < Rhi & Igreen > Glo & ...
Igreen < Ghi & Iblue > Blo & Iblue < Bhi;
subplot(1,2,1)
imshow(GreenLeaves);
title('Green Leaves')

Rlo = 130;
Rhi = 178;
Glo = 30;
Ghi = 66;
Blo = 35;
Bhi = 72;

RedSign = Ired > Rlo & Ired < Rhi & Igreen > Glo & ...
Igreen < Ghi & Iblue > Blo & Iblue < Bhi;
subplot(1,2,2)
imshow(RedSign);
title('Red Sign')

% On this image, the limits do not work as well at all.
% Small lighting differences cause large variances in
% classification accuracy going from color alone. It does
% catch some leaves and parts of the sign but not a very
% good classification. It goes to show how important lighting
% conditions are when doing classification this way.