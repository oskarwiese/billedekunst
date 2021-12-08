% Start by clearing the workspace
clc; clear; close all;

% Test that the image is loaded correctly
ct = dicomread('1-105.dcm');
imshow(ct, [-100, 200]);

%% Exercise 1

% Creating the mask, use first time:
    %imshow(ct,[-100,200]);
    %SpleenROI = roipoly;
    %imwrite(SpleenROI, 'SpleenROI.png');
% Otherwise load the mask:
SpleenROI = imread('SpleenROI.png');
SpleenVals = double(ct(SpleenROI));

LiverROI = imread('LiverROI.png');
LiverVals = double(ct(LiverROI));

KidneyROI = imread('KidneyROI.png');
KidneyVals = double(ct(KidneyROI));

%% Exercise 2
figure;
histogram(SpleenVals)
title('Spleen Histogram')
meanSpleen = mean(SpleenVals)
stdSpleen = std(SpleenVals)

figure;
histogram(LiverVals)
title('Liver Histogram')
meanLiver = mean(LiverVals)
stdLiver = std(LiverVals)

figure;
histogram(KidneyVals)
title('Kidney Histogram')
meanKidney = mean(KidneyVals)
stdKidney = std(KidneyVals)
%% Exercise 3
% Checking against exercise 6, the HU of the
% spleen are higher here. This could be because
% the image from the previous exercise
% used a histogram-scaled version.
% It could also be due to varying factors
% in the patients' bodies.

% PDF for spleen
xrange = -500:0.1:500;
pdfFitSpleen = normpdf(xrange, mean(SpleenVals),...
    std(SpleenVals));
figure;
S = length(SpleenVals); % Scale factor
hold on
histogram(SpleenVals, xrange);
plot(xrange, pdfFitSpleen * S, 'r');
hold off
xlim([40, 180]);
title('Spleen PDF');
% The fitted Gaussian distribution does
% not fit particularly well on the data
% but it will do

% PDF for liver
xrange = -500:0.1:500;
pdfFitLiver = normpdf(xrange, mean(LiverVals),...
    std(LiverVals));
figure;
S = length(LiverVals);
hold on
histogram(LiverVals, xrange);
plot(xrange, pdfFitLiver * S, 'r');
hold off
xlim([0, 200]);
title('Liver PDF'); 
% This has a good fit on the data

% PDF for kidney
xrange = -500:0.1:500;
pdfFitKidney = normpdf(xrange, mean(KidneyVals),...
    std(KidneyVals));
figure;
S = length(KidneyVals);
hold on
histogram(KidneyVals, xrange);
plot(xrange, pdfFitKidney * S, 'r');
hold off
xlim([-50, 200]);
title('Kidney PDF'); 
% As the kidney has almost a bimodal
% distribution, a Gaussian PDF will
% likely not have a very nice fit on the
% data
%% Exercise 4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THE FOLLOWING EXERCISES 4-9 CONSIDER ONLY THE
% SEGMENTATION OF THE SPLEEN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% We disregard the tails of the PDFs and choose
% the following limits:
T1 = 90; % Lower limit
T2 = 140; % Upper limit

binI = (ct > T1) & (ct < T2);
figure
imshow(binI);

%% Exercise 5
% There are a lot of other organs kept in
% the image. This is due to soft tissue
% generally being in the same range as the 
% spleen. This makes it hard/impossible to
% seperate the spleen only using this kind of
% binary classification.

%% Exercise 5
% We start by creating the disk SE,
% we use a disk SE with a radius of 1 pixels
SE1 = strel('disk',1);
ImClosed = imclose(binI, SE1);

% Upon inspection, the liver seems to be 
% solid and seperatable
imshow(ImClosed);

%% Exercise 6
% Here, we use again use a disk SE with
% a radius of 3 pixels to open the image
SE2 = strel('disk', 3);
IMorph = imopen(ImClosed, SE2);

% Inspecting this, the image seems to have
% a solid spleen that is seperated from the
% rest of the organs/tissue
imshow(IMorph);

%% Exercise 7
L8 = bwlabel(IMorph, 8);
RGB8 = label2rgb(L8);
figure
% Showing the labeled image, the spleen
% has a seperate label from the other
% organs/tissue in the image which means 
% the labelling works
imagesc(RGB8);
%% Exercise 9
% We start by getting the BLOB features:
stats8 = regionprops(L8,'all');
% Display the perimeters for the found
% BLOBs:
Perimeters = [stats8.Perimeter]

% We select BLOBs with an area bigger than
% 1200 pixels, a perimeter less than 400
% pixels, and a major axis that is at least
% twice as long as the minor axis
idx = find([stats8.Area] > 1200 &...
    [stats8.Perimeter] < 400 &...
    ([stats8.MajorAxisLength] ./ [stats8.MinorAxisLength]) > 2);
ISpleenFinal = ismember(L8, idx);
figure;
imagesc(ISpleenFinal);
axis image;
title('Spleen candidate');

% Now, only the spleen is shown
%%
% Repeating the process for the liver and
% kidney gives the following:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LIVER PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Limits:
% T1 = 80; % Lower limit
% T2 = 120; % Upper limit
% 
% % Morphology:
% SE1 = strel('disk',2);
% ImClosed = imclose(binI, SE1);
% SE2 = strel('disk', 9);
% IMorph = imopen(ImClosed, SE2);
% 
% % BLOB Analysis:
% idx = find([stats8.Area] > 8000 &...
%     [stats8.Perimeter] > 400);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% KIDNEY PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Limits:
% T1 = 65; % Lower limit
% T2 = 170; % Upper limit
% 
% % Morphology:
% SE1 = strel('disk',2);
% ImClosed = imclose(binI, SE1);
% SE2 = strel('disk', 15);
% IMorph = imopen(ImClosed, SE2);
% 
% % BLOB Analysis:
% idx = find([stats8.Eccentricity] < 0.7);

%% Exercise 10
% The "TestSegmentation.m" script should now be run
% with the training image "1-105.dcm" and the test image
% "1-083.dcm". You can set these in the "TestSegmentation.m"
% script. The script uses the function in
% "AbdominalSegmentation.m" for labelling the image.
TrainingImage = imread('training.png');
ValidationImage = imread('validation.png');

figure;
subplot(2,1,1)
imshow(TrainingImage);
title('Training image')
subplot(2,1,2)
imshow(ValidationImage);
title('Validation Image')

% The DICE scores for the liver labelling
% in the images are quite good as it is easy
% to threshold. Additionally, it is easy to 
% isolate the liver in BLOB analysis due to
% its large area and perimeter. Also, as the
% liver is so large, you can use some larger
% structural elements for the morphology

% The DICE scores for the spleen is also good.
% The spleen is also rather uniform in color so
% it can be thresholded. This means that you
% do not have to use large SE's which lets
% you retain more information about the spleen.

% The kidney in the training image is expectedly
% found but in the test image the label is way 
% too large leading to a large false positive,
% thus reducing the DICE score. This is because
% it is very non-uniform in color so a rather wide
% threshold is used so there is still a lot 
% of soft tissue included when doing the morphology