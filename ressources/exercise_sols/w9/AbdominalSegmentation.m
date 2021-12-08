function [ISpleen, ILiver, IKidney] = AbdominalSegmentation(ct)

T1Spleen = 90;
T2Spleen = 140;

T1Liver = 80;
T2Liver = 120;

T1Kidney = 65;
T2Kidney = 170;

binISpleen = (ct > T1Spleen) & (ct < T2Spleen);
binILiver = (ct > T1Liver) & (ct < T2Liver);
binIKidney = (ct > T1Kidney) & (ct < T2Kidney);

% Spleen Morphology
SE1Spleen = strel('disk',1);
ImClosedSpleen = imclose(binISpleen, SE1Spleen);

SE2Spleen = strel('disk',3);
IMorphSpleen = imopen(ImClosedSpleen, SE2Spleen);

% Liver Morphology
SE1Liver = strel('disk',2);
ImClosedLiver = imclose(binILiver, SE1Liver);

SE2Liver = strel('disk',9);
IMorphLiver = imopen(ImClosedLiver, SE2Liver);

% Kidney Morphology
SE1Kidney = strel('disk',2);
ImClosedKidney = imclose(binIKidney, SE1Kidney);

SE2Kidney = strel('disk',15);
IMorphKidney = imopen(ImClosedKidney, SE2Kidney);

% Spleen Labelling:
L8Spleen = bwlabel(IMorphSpleen,8);
RGB8Spleen = label2rgb(L8Spleen);

stats8Spleen = regionprops(L8Spleen, 'All');

idx = find([stats8Spleen.Area] > 1200 &...
    [stats8Spleen.Perimeter] < 400 &...
    ([stats8Spleen.MajorAxisLength] ./...
    [stats8Spleen.MinorAxisLength]) > 2);
ISpleen = ismember(L8Spleen,idx);

% Liver Labelling:
L8Liver = bwlabel(IMorphLiver,8);
RGB8Liver = label2rgb(L8Liver);

stats8Liver = regionprops(L8Liver, 'All');

idx = find([stats8Liver.Area] > 8000 &...
    [stats8Liver.Perimeter] > 400);
ILiver = ismember(L8Liver,idx);

% Kidney Labelling:
L8Kidney = bwlabel(IMorphKidney,8);
RGB8Kidney = label2rgb(L8Kidney);

stats8Kidney = regionprops(L8Kidney, 'All');

idx = find([stats8Kidney.Eccentricity] < 0.7);
IKidney = ismember(L8Kidney,idx);