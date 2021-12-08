%% Geometric Transformations, week 8
%% Exercise 1
% The first step is to create the 2D points of the grid. Here the
% function meshgrid is useful. Look at the documentation in Matlab:

[X,Y] = meshgrid(-6:6,-6:6);

% Look at the elements of the matrices. Does it make sense, in relation to the
% input vectors?

% Yes it does, our x vector is -6,-5,...,6 and the X is therefore a row
% wise copy

%% Exercise 2
% Rearrange the 2D points into a 2×n matrix, such that each column
% in the matrix corresponds to a 2D point
XY = [X(:) Y(:)]'

% Why is it beneficial to apply this rearrangement?

% Its beneficial since each column represents a point in the grid.

%% Exercise 3
% A function PlotGrid has been provided, which plots the 2D points
% as a grid. Show the original 2D points:
PlotGrid(XY)

% Compare the plot with the elements of the matrices X and Y
% The elements of X represents the x coordinates and the elements of Y represents
% the y coordinates

%% Exercise 4
% Apply a simple scaling to the 2D points, where Sx = 0.7 and Sy =
% 1.3. Consider how this can easily be implemented in Matlab, without using 
% forloops. Plot the result and compare with the original grid.

Sx = 0.7;
Sy =1.3;

scaleMat = [Sx 0; 0 Sy]; %Eq. 10.4, page 140
XYscale = scaleMat*XY;

figure;
subplot(1,2,1)
PlotGrid(XYscale);
title('Scaled')
subplot(1,2,2)
PlotGrid(XY)
title('Original grid')

%% Exercise 5
% Rotate the 2D grid 20 degrees counterclockwise and show the result.
% Does the grid rotate as expected? Why/Why not?

theta = (20*pi)/180; % MATLAB operates with radians so we convert 20 degrees to radians

rotMat = [cos(theta) -sin(theta); sin(theta) cos(theta)]; %Eq. 10.5

XYrot = rotMat*XY;

figure; 
subplot(1,2,1)
PlotGrid(XYrot);
title('Rotated 20 degrees counter')
subplot(1,2,2)
PlotGrid(XY)
title('Original grid')

% Yes it did rotate as exepected

%% Extra transformations
% Shear the grid using Bx = 0.5 and By = 0.1. Show the result and
% compare with the original grid.
Bx = 0.5;
By = 0.1;

shearMat = [1 Bx; By 1]; %Eq. 10.7

XYshear = shearMat*XY;

subplot(1,2,1)
PlotGrid(XYshear);
title('Shear grid')
subplot(1,2,2)
PlotGrid(XY);
title('Original')

%% Extra transformations 
% Try to apply a translation using homogeneous coordinates.(see page 143)

homoMat = [1 0 8; 0 1 8; 0 0 1];
affMat = [X(:) , Y(:) , ones(size(XY,2),1)]';

XYtrans = homoMat*affMat;

figure;
PlotGrid(XYtrans);
%% Extra transformations 
% Play around with the transformation parameters and try to apply
% multiple transformations to the grid.

homoMat = [1 0 8; 0 1 8; 0 0 1];
scalmatrix = [0.7 0 0 ; 0 1.3 0 ; 0 0 1];
rotmatrix = [cos(theta) -sin(theta) 0 ; sin(theta) cos(theta) 0; 0 0 1];
shearmatrix = [1 0.5 0 ; 0.1 1 0 ; 0 0 1];

affMat = [X(:) , Y(:) , ones(size(XY,2),1)]';

combTrans = (homoMat*scalmatrix*rotmatrix*shearmatrix);

XYcombtrans = combTrans * affMat;

subplot(1,2,1)
PlotGrid(XYcombtrans);
title('Combining transformations')
subplot(1,2,2)
PlotGrid(XY)
title('Original')

%% Exericse 6
% Start by loading the image Im.png into Matlab and show the image
clc 
clear all
Im = imread('Im.png');

imshow(Im)

%% Exercise 7
% Scale the image with Sx = 3 and Sy = 2, using forward mapping
% (and zero-order interpolation) and show the resulting image. What is the result
% of forward mapping.
Sx = 3; 
Sy = 2; 

[X,Y] = meshgrid(1:size(Im,2),1:size(Im,1));
XY = [X(:) Y(:)]';

scaleMat = [Sx 0; 0 Sy]; %Eq. 10.4
XYscale = scaleMat*XY;

ImScale = zeros(size(Im,1)*Sy,size(Im,2)*Sx);

for idx=1:size(XYscale,2)
    ImScale(XYscale(2,idx),XYscale(1,idx)) = Im(idx);
    %In this case it is not necessary to use interpolation as we are
    %scaling with positive integers
end

ImScale = uint8(ImScale);
figure; imagesc(ImScale); colormap('gray')
% It is evident there are holes in the image 

%% Exercise 8
% Scale the image again, but now using backward mapping using for
% example interp2. Compare the result with the previous image.

[X,Y] = meshgrid(1:size(Im,2),1:size(Im,1));
[Xscale,Yscale] = meshgrid(1:size(Im,2)*Sx,1:size(Im,1)*Sy);
XYscale = [Xscale(:) Yscale(:)]';

invScaleMat = [1/Sx 0; 0 1/Sy];%eq.10.11

XYinv = invScaleMat *XYscale;
Xinv = reshape(XYinv(1,:),size(Xscale));
Yinv = reshape(XYinv(2,:),size(Yscale));

Vq = interp2(X,Y,im2double(Im),Xinv,Yinv);

figure; imagesc(Vq),colormap('gray')
%pros: No holes in image
%cons:  We need to inverse the matrices and perform interpolation

%% Exercise 9
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Registration %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all;

hand1 = imread('Hand1.jpg');
hand2 = imread('Hand2.jpg');

subplot(1,2,1)
imshow(hand1)
title('hand 1')
subplot(1,2,2)
imshow(hand2)
title('hand 2')

% cpselect(hand2, hand1);

% movingpoints = cpstruct.basePoints;
% fixedpoints = cpstruct.inputPoints;
% 
% save('fixedPoints.mat','fixedpoints');
% save('movingPoints.mat','movingpoints');

load('fixedPoints.mat');
load('movingPoints.mat');
%% Exercise 10

figure;
plot(fixedpoints(:,1), fixedpoints(:,2), 'b*-', ...
movingpoints(:,1), movingpoints(:,2), 'r*-');
legend('Hand 1 - The fixed image', 'Hand 2 - The moving image');
axis ij; % This reverses the direction of the axis

%% Exercise 11
% Calculate F from your annotations. It can for example be done
% like this:

ex = fixedpoints(:,1) - movingpoints(:,1);
errorX = ex' * ex;
ey = fixedpoints(:,2) - movingpoints(:,2);
errorY = ey' * ey;

% F is a measure of how well the landmarks are aligned. More precisely F is
% the sum of squared line lengths between matching LMs.
F = errorX + errorY %1.332*10^5

%% Exercise 12
% Compute the centre of mass of the two point sets. Store them in
% two variables called fixed_COM and moving_COM.

fixed_COM = 1/length(fixedpoints(:,1)) * sum(fixedpoints);
moving_COM = 1/length(movingpoints(:,1)) * sum(movingpoints);

%% Exercise 13
% Create two translated landmark sets by subtracting the computed
% centre-of-masses:

fixed_trans = [fixedpoints(:,1) - fixed_COM(1) ...
fixedpoints(:,2) - fixed_COM(2)];
moving_trans = [movingpoints(:,1) - moving_COM(1) ...
movingpoints(:,2) - moving_COM(2)];

% Plot the two translated landmark sets in the same plot. What do you observe?
% Do the hands match better than before?

figure;
plot(fixed_trans(:,1), fixed_trans(:,2), 'b*-', ...
moving_trans(:,1), moving_trans(:,2), 'r*-');
legend('Hand 1 - The fixed image', 'Hand 2 - The moving image');
axis ij; % This reverses the direction of the axis

%The two hands match better - but there is some missing rotation
%The two center of masses are moved to (0,0)

%% Exercise 14
% Compute the transform that makes hand2 fit hand1:
mytform = fitgeotrans(movingpoints, fixedpoints,'NonreflectiveSimilarity');

%% Exercise 15
% Compute the transformed points from hand2 using:
forward = transformPointsForward(mytform,movingpoints);

% plot them together with the points from hand1. What do you observe?
figure;
plot(fixedpoints(:,1), fixedpoints(:,2), 'b*-', ...
forward(:,1), forward(:,2), 'r*-');
legend('Hand 1 - The fixed image', 'Hand 2 - The moving image');
axis ij; % This reverses the direction of the axis

% The hands now match very well!

%% Exercise 16
% Use Equation (1) to calculate F after alignment. Do this by substituting
% bi so they are now the transformed coordinates (forward). What has
% happened to the value of F. Can you explain it?

exnew = fixedpoints(:,1) - forward(:,1);
errorXnew = exnew' * exnew;
eynew = fixedpoints(:,2) - forward(:,2);
errorYnew = eynew' * eynew;
Fnew = errorXnew + errorYnew %8.757*10^3

% The value of F has decreased significantly. This is because the hands has
% been aligned. F is a measure of the distance between the points so it
% makes sense that F is smaller now when the points almost match

%% Exercise 17
% Transform the image of hand2 using:
hand2t = imwarp(hand2, mytform);

%Show the transformed version of hand2 together with hand1. What do you observe?
subplot(1,2,1)
imshow(hand2t)
title('hand2 transformed')
subplot(1,2,2)
imshow(hand1)
title('Hand1')

%% Exercise 18
clc;clear all;
% Find out what a glioma is and how it is typically treated.
% Glioma = Brain tumor. (A glioma is a type of tumor that starts in the
% glial cells of the brain or the spine)
% Treatment: surgery, radiation therapy, and chemotherapy. 
%% Exercise 19
%  Use cpselect with BT1 as the fixed image and BT2 as the moving
% image. Choose at least 4 anatomical landmarks and place them around the brain
% (not the tumor). Save them to BT1points.mat and BT2points.mat.

brain1 = imread('BT1.png');
brain2 = imread('BT2.png');

% cpselect(brain1, brain2);
% BT2points = cpstruct1.basePoints;
% BT1points = cpstruct1.inputPoints;
% 
% save('BT2points.mat','BT2points');
% save('BT1points.mat','BT1points');

load('BT1points.mat');
load('BT2points.mat');

figure
subplot(1,2,1)
imshow(brain1); 
hold on 
plot(BT1points(:,1),BT1points(:,2),'o')
hold off
subplot(1,2,2)
imshow(brain2); 
hold on 
plot(BT2points(:,1),BT2points(:,2),'o')
hold off
%% Exercise 20
% Use cp2tform and imtransform to create a similarity transformed
% version of BT2. The transformed image should keep the exact same size as the
% fixed image. This can be done by:

mytform = fitgeotrans(BT2points, BT1points,'nonreflectivesimilarity');
brain2t = imwarp(brain2, mytform,'OutputView', imref2d( size(brain2) ));

% Show the image of BT1 together with transformed version of BT2. What do you
% observe?

figure;
subplot(1,3,1);
imshow(brain1);
title('Brain 1');

subplot(1,3,2);
imshow(brain2);
title('Brain 2');

subplot(1,3,3);
imshow(brain2t);
title('Brain 2 transformed');

% It can be seen the brain 2 image has been translated

%% Exercise 21
% Use roipoly to draw precisely around the edge of the tumor in
% BT1 and the transformed version of BT2. Store the two binary images as
% BT1_annot.png and BT2_annot.png. Show the two binary tumor images side
% by side and comment.

%% Drawing tumor 1
% Tumor1 = roipoly(brain1); 
% imwrite(Tumor1, 'Tumor1.png');
%% Drawing tumor 2
% Tumor2 = roipoly(TI); 
% imwrite(Tumor2, 'Tumor2.png');
%%
T1 = imread('Tumor1.png');
T2 = imread('Tumor2.png');

subplot(1,2,1)
imshow(T1)
title('Tumor before surgery')
subplot(1,2,2)
imshow(T2)
title('Tumor after surgery')

% Its clear to see the tumor has decreased in size

%% Exercise 22
% Add the two binary images together (BT1 and the transformed
% version of BT2). Use label2RGB to visualise the differences. Try to add two
% times the second image and use label2RGB again. Explain the output image
compare1 = label2rgb(T1+T2);
compare2 = label2rgb(T1+2*T2);

figure; 
subplot(1,2,1);
imshow(compare1);
subplot(1,2,2);
imshow(compare2);

% 1) Three classes:
%0 = never tumor, 1 = tumor before OR after, 2 = tumor before AND after

% 2) Four classes. It now distinguish between areas where the tumor has been removed 
%and areas to which the tumor has spread. 
%0 = never tumor, 1 = tumor before (have been removed), 2 = tumor after (spread to), 3 = tumor before AND after


%% Exercise 23
%Calculate how much of the tumor that was removed during treatment.
%You can do this by comparing the cross sectional area of the tumor before
%and after.

Area1 = sum(T1(:));
Area2 = sum(T2(:));

(Area1-Area2)/Area1*100 %The percentage of the tumor that has been removed
% The tumor is 79.4% smaller than it used to be!

