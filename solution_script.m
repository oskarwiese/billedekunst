%%% utils for matlab
%%
clc, clear,  close all
path_folder = 'C:\Users\ander\Desktop\02502 Image Analysis\exam' ;
cd(path_folder)
addpath data

%% Load data and subtract mean for PCA
M = load('data\irisdata.txt');
M = M(:,1:4);                           % Husk at tjekke om data faktisk passer med viden der gives
M = M - mean(M,1);                      %subtracting mean

%% Q1
% The irisdata.txt file contains measurements from 150 iris flowers. The
% measurements are the sepal length, sepal width, petal length and petal
% width. So you have M=4 features, N=150 observations. A principal
% component analysis (PCA) should be done on these data. How many
% percent of the total variation do the two first principal components
% explain?

Cx = cov(M);
[PC, V] = eig(Cx);                      % Calculate principal components and eigenvectors
V = diag(V);                            % Turn V into vector
[junk, rindices] = sort(-1*V);          % Sort from largest to smallest
V = V(rindices);                        % Sort eigenvalues
PC = PC(:,rindices);                    % Sort eigenvectors
Vnorm = V / sum(V) * 100;               % Find variance explained
plot(Vnorm, '*-')
ylabel('Percent explained variance')
xlabel('Principal component')
Vnorm(1)+Vnorm(2)                       % How many
% percent of the total variation do the two first principal components
% explain?

%% Q2
% The irisdata.txt file contains measurements from 150 iris flowers. The
% measurements are the sepal length, sepal width, petal length, and petal
% width. So you have M=4 features, N=150 observations. A principal
% component analysis (PCA) should be done on these data.
% Aer the PCA, the flower data (sepal lengths and widths, petal lengths
% and widths) are projected into PCA space. What are the projected values
% of the first flower?

signals = PC' * M';                     % Project data into PCA space
signals(:,1)                            % Show the projected values of first flower

%%
% The irisdata.txt file contains measurements from 150 iris flowers. The
% measurements are the sepal length, sepal width, petal length, and petal
% width. So you have M=4 features, N=150 observations. A principal
% component analysis (PCA) should be done on these data.
% Aer the PCA, the flower data (sepal lengths and widths, petal lengths
% and widths) are projected into PCA space. What are the projected values
% of the first flower?

[vectors,values,psi] = pc_evectors(M',4);

sigs = vectors'*M';
sigs(:,1)

%% Q3
% The photo called sky_gray.png is loaded and a linear histogram
% stretching is performed so the new image has a maximum pixel value of
% 200 and a minimum pixel value of 10. What is the average pixel value of
% the new image?

SG = double(imread('data\sky_gray.png'));
%imshow(SG)
mind = 10;
maxd = 200;
st_SG = (maxd-mind) /(max(SG(:)) - min(SG(:))) * (SG - min(SG(:))) + mind;  % Use linear histogram stretching
mean(st_SG(:))

%% Q4
% The photo called sky.png is loaded and an RGB threshold is performed
% with the limits R < 100, G > 85, G < 200, and B > 150. Pixels with values
% within these limits are set to foreground and the rest of the pixels are set
% to background.
% The resulting 2D binary image is morphologically eroded using a diskshaped structuring element with radius=5.
% When doing an erosion the pixels beyond the image border are assigned
% a value of 1 (the default Matlab behavior).
% How many foreground pixels are there in the final image?

SG_RGB = double(imread('data\sky.png'));    % Load image
lgcl_SG = SG_RGB(:,:,1) < 100 & SG_RGB(:,:,2) > 85 & SG_RGB(:,:,2) < 200 & SG_RGB(:,:,3) > 150; % Threshold image
se1 = strel('disk',5);                      % Define structuring element
out = imerode(lgcl_SG,se1);                 % Use erosing with disk SE
sum(out(:))                                 % Find number of foreground pixels

%% Q5
% The photo called flower.png is loaded and it is converted from the RGB
% color space to the HSV color space. Secondly, a threshold is performed
% on the HSV values with the limits H < 0.25, S > 0.8 and V > 0.8. Pixels with
% values within these limits are set to foreground and the rest of the pixels
% are set to background.
% Finally, a morphological opening is performed on the binary image using
% a disk-shaped structuring element with radius=5. When doing a dilation,
% pixels beyond the image border are assigned a value of 0 and when
% doing an erosion the pixels beyond the image border are assigned a
% value of 1 (the default Matlab behavior).
% What is the number of foreground pixels in the resulting image?

flwer = imread('data\flower.png');
flwer_hsv = rgb2hsv(flwer);                 % Convert from rgb 2 hsv
lgcl_flwer = flwer_hsv(:,:,1) < 0.25 & flwer_hsv(:,:,2) > 0.8 & flwer_hsv(:,:,3) > 0.8; % Do thresholding
se1 = strel('disk',5);                      % Define disk structuring element
out = imopen(lgcl_flwer,se1);               % perform morphological opening on image
sum(out(:))

%% Q6
% Five photos have been taken. They are named car1.jpg - car5.jpg and
% they have the dimensions (W=800, H=600). A principal component
% analysis (PCA) is performed on the grey values of the five images. You
% can use the two helper functions pc_evectors.m and sortem.m to
% compute the PCA. How much of the total variation in the images is
% explained by the first principal component?

Mc = zeros(800*600,5);
for i = 1:5
    str = ['data\car',num2str(i),'.jpg'];
    tmp_pic = double(imread(str));
    Mc(:,i) = tmp_pic(:) - mean(tmp_pic(:));
end
[vectors,values,psi] = pc_evectors(Mc,5);
values(1)/sum(values)                           % Find the explained variance of the first principal component
[nv,nd] = sortem(vectors',diag(values));        % Sort the vectors and values
vnorm = nd/sum(nd(:)) * 100                     % Find all variance explained

%% Q7
% The photo called sky_gray.png is transformed using a gamma mapping
% with gamma=1.21. The output image is filtered using a 5x5 median filter.
% What is the resulting pixel value in the pixel at row=40, column=50 (when
% using a 1-based matrix-based coordinate system)?

SG = double(imread('data\sky_gray.png'));
gamma = 1.21;
SG = 255*((SG / 255).^gamma);                    % Gamma mapping transformation
kernel_size = 5
SG = medfilt2(SG,[kernel_size,kernel_size]);     % 5x5 median filtering
round(SG(40, 50))

%% Q8
% The photo called flowerwall.png is filtered using an average filter with a
% filter size of 15. The filtering is performed with border replication. What is
% the resulting pixel value in the pixel at row=5 and column=50 (when
% using a 1-based matrix-based coordinate system)?

FW = double(imread('data\flowerwall.png'));
windowWidth = 15;                                   % Define size of kernel
se = ones(windowWidth) / windowWidth .^ 2;          % Create average filtering
out = imfilter(FW,se);                              % Filter image with kernel
out(5,50)

%% Q9
% A photo has been taken of a set of floorboards (floorboards.png) and the
% goal is to measure the amounts of knots in the wood. First, a threshold of
% 100 is used, so pixels below the threshold are set to foreground and the
% rest is set to background. To remove noise a morphological closing is
% performed with a disk-shaped structuring element with radius=10
% followed by a morphological opening with a disk-shaped structuring
% element with radius=3. When doing a dilation, pixels beyond the image
% border are assigned a value of 0 and when doing an erosion the pixels
% beyond the image border are assigned a value of 1 (the default Matlab
% behavior).
% Finally, all BLOBs that are connected to the image border are removed.
% How many foreground pixels are remaining in the image?

FB = double(imread('data\floorboards.png'));
lgcl_FB = FB < 100;                                         % Threshold values under 100 to be 1
se1 = strel('disk',10);                                     % Disk structuring element of size 10
se2 = strel('disk',3);
fnl_FB = imclearborder(imopen(imclose(lgcl_FB,se1),se2));   % imclearborder removes all blobs from the border. Performs first a closing, then an opening and at the end clears border
L8 = bwlabel(fnl_FB,8);                                     % Find blobs using 8-connectivity
imagesc(L8);                                                % Prints image
colormap(hot);
title('8 connectiviy')
sum(fnl_FB(:))

%% Q10
% A photo has been taken of a set of floorboards (floorboards.png) and the
% goal is to measure the amounts of knots in the wood. First, a threshold of
% 100 is used, so pixels below the threshold are set to foreground and the
% rest is set to background. To remove noise a morphological closing is
% performed with a disk-shaped structuring element with radius=10
% followed by a morphological opening with a disk-shaped structuring
% element with radius=3. When doing a dilation, pixels beyond the image
% border are assigned a value of 0 and when doing an erosion the pixels
% beyond the image border are assigned a value of 1 (the default Matlab
% behavior). A BLOB analysis is performed where all BLOBS are found using
% 8-connectivity. All BLOBs that are connected to the image border are
% removed.
% The area of the found BLOBs are computed and only the BLOBs with an
% area larger than 100 pixels are kept. How many BLOBs are found in the
% final image?

FB = double(imread('data\floorboards.png'));
lgcl_FB = FB < 100;                                         % Threshold values under 100 to be 1
se1 = strel('disk',10);                                     % Disk structuring element of size 10
se2 = strel('disk',3);
fnl_FB = imclearborder(imopen(imclose(lgcl_FB,se1),se2));   % imclearborder removes all blobs from the border. Performs first a closing, then an opening and at the end clears border
L8 = bwlabel(fnl_FB,8);                                     % Uses 8-connectivity to find blobs
imagesc(L8);                                                % Prints image

stats8 = regionprops(L8, 'Area');                           % Find areas of each blob
bw2 = numel(find([stats8.Area] > 100))                      % Find number of blobs with area over 100

%% Q11
% The binary image books_bw.png contains letters. A BLOB analysis is
% performed using 8-connectivity. For each BLOB, the area and the
% perimeter is computed. The BLOBs with area > 100 and perimeter > 500
% are kept. Which letters are visible in the final image?

im = imread('books_bw.png');
im_lab = bwlabel(im,8);                                     % Use 8-connectivity to find all blobs
stats = regionprops(im_lab,'all');                          % Get all stats about all blobs
idx = find([stats.Area] > 100 & [stats.Perimeter] > 500 );  % Find blobs where area is over 100 and perimeter is over 500
im_bw = ismember(im_lab,idx);                               % Find blobs in im_lab with idx                           
imshow(im_bw)                                               % Plot final image

%% Q12
% Seven corresponding landmarks have been placed on two images
% (cat1.png and cat2.png). The landmarks are stored in the files
% catfixedPoints.mat and catmovingPoints.mat. What is the sum of
% squared dierences between the fixed and the moving landmarks?

load('catfixedPoints.mat')
load('catmovingPoints.mat')
sum((fixedpoints-movingpoints).^2,'all')                % Find sum of squared differences between two vectors

%% Q13
% Seven corresponding landmarks have been placed on two images
% (cat1.png and cat2.png). The landmarks are stored in the files
% catfixedPoints.mat and catmovingPoints.mat. A similarity transform
% (translation, rotation, and scaling) has been performed that aligns the
% moving points to the fixed points. The computed transform is applied to
% the cat2.png photo. How does the resulting image look like?

mytform = fitgeotrans(movingpoints, fixedpoints,'NonreflectiveSimilarity');         % Find the best transform. Outputs transformation matrix
forward = transformPointsForward(mytform,movingpoints);                             % Transform points using best transformation

% plot them together with the points from hand1. What do you observe?
cat = im2double(imread('cat2.png'));
cat_moved = imwarp(cat, mytform);                                                   % Move the entire image based on the point translation

%Show the transformed version of hand2 together with hand1. What do you observe?
subplot(1,2,1)
imshow(cat)
title('cat')
subplot(1,2,2)
imshow(cat_moved)
title('cat moved')

%% Q14
% An abdominal scan has been acquired on a CT scanner. One of the slices
% of the scan is stored as a DICOM file called 1-179.dcm. An expert has
% marked a part of the liver as a binary mask (region of interest). The
% binary mask is stored as the file LiverROI.png.
% By using the DICOM image and the mask image, the image values in the
% DICOM image inside the mask (the liver) can be extracted.
% The average value and the standard deviation of the extracted pixel
% values are computed. A low threshold, T1, is defined as the average
% value minus the standard deviation and a high threshold, T2, is defined
% as the average value plus the standard deviation.
% Finally, a segmentation of the DICOM image (1-179.dcm) is made where
% all pixels with values > T1 and < T2 are set to foreground and the rest are
% set to background. How many foreground pixels are there?

dc = double(dicomread('1-179.dcm'));

liver = imread('LiverRoi.png');                             % Load region of interest mask
dc_l = dc(liver);                                           % Indexing into the image with the liver values
T = mean(dc_l,'all')+[-1,1]*std(dc_l,[],'all')              % Find mean +/- standard deviation

dc_t = dc>T(1) & dc<T(2) ;                                  % Where in the image are values lower than T2 or higher than T1
sum(dc_t(:))                                                % Compute total number of elements where the above is true

%% Q15
% An abdominal scan has been acquired on a CT scanner. One of the slices
% of the scan is stored as a DICOM file called 1-179.dcm. A low threshold,
% T1 = 90, and a high threshold, T2 = 140, are defined. The pixel values of
% the DICOM image are segmented by setting all pixel values that are >T1
% and <T2 to foreground and the rest are set to background.
% The binary image is processed by first applying a morphological closing
% using a disk-shaped structuring element with radius=3 followed by a
% morphological openingwith the same structuring element. When doing a
% dilation, pixels beyond the image border are assigned a value of 0 and
% when doing an erosion the pixels beyond the image border are assigned
% a value of 1 (the default Matlab behavior).
% In the final step, a BLOB analysis is done using 8-connectivity. The largest
% BLOB is found. The area (in pixels) of the largest BLOB is:

dc = double(dicomread('1-179.dcm'));
T = [90, 140];
dc_t = dc>T(1) & dc<T(2) ;

se = strel('disk',3);                          % Define structuring element with disk shape and size 3  

dc_close = imclose(dc_t,se);                   % Do a morphological closing on the image
dc_open = imopen(dc_close,se);                 % Do a morphological opening on the image

im_lab = bwlabel(dc_open,8);                   % Do 8-connectivity on the image after morphologizing it
stats = regionprops(im_lab,'area');            % Find all the areas of the blobs
max([stats.Area])                              % What is the area of the largest blob?

%% Q16
% NASA's Mars Perseverance rover has explored Mars since its landing at
% the beginning of 2021. To explore the surface of Mars, the rover uses a
% custom build camera. Now the rover has discovered three spectral peaks
% that might reflect dierent types of cosmic dust. Each dust spectra
% appears to follow a normal distribution. The parametric distributions of
% the three dust classes are N(7,2*2), N(15,5*5), and N(3,5*5).
% NASA asks help to define the thresholds to perform robust classification.
% They wish to perform a minimum distance classification of the three dust
% classes.
% What signal thresholds should NASA use?

T = [(3+7), (7+15)]/2

%% Q17
% NASA's Mars Perseverance rover has explored Mars since its landing at
% the beginning of 2021. To explore the surface of Mars, the rover uses a
% custom build camera. Now the rover has discovered three spectral peaks
% that might reflect dierent types of cosmic dust. Each dust spectra
% appears to follow a normal distribution. The parametric distributions of
% the three dust classes are N(7,2*2), N(15,5*5), and N(3,5*5).
% NASA asks help to define the thresholds to perform robust classification.
% They wish to perform a parametric classification of the three dust
% classes.
% What signal thresholds should NASA use?

xrange = 0:0.01:20;
pdf1 = normpdf(xrange, 3, 5);               % Define the normal distribution using xrange, mean, std
pdf2 = normpdf(xrange, 7, 2);
pdf3 = normpdf(xrange, 15, 5);

plot(xrange,[pdf1;pdf2;pdf3])               % Show the normal distributions in a plot to find intersections
% The Gaussians crosses in 4.24 and 10.26

%% Q18
% The normalised cross correlation (NCC) between the image and the
% template is computed. What is the NCC in the marked pixel in the image?

im = [167,193, 180;
      9, 189, 8;
      217, 100, 71];
tem = [208, 233, 71;
       231, 161, 139;
       32, 25, 244];
   
sum(im.*tem,'all')/sqrt(sum(im(:).^2)*sum(tem(:).^2))       % Compute NCC normalized cross correlation from im and kernel

%% Q19
% A company is making an automated system for fish inspection. They are
% using a camera with a CCD chip that measures 5.4 x 4.2 mm and that has
% a focal length of 10 mm. The camera takes photos that have dimensions
% 6480 x 5040 pixels and the camera is placed 110 cm from the fish, where
% a sharp image can be acquired of the fish.
% How many pixels wide is a fish that has a length of 40 cm?

f = 10;
g = 1100;
G = 400;
b = f;
pixel_mm = 6480/5.4;

B = b*G/g;                  % Solve b/B = g/G with respect to missing variable to find focal length focal point or missing variable
B*pixel_mm

%% Q20
% Two types of mushrooms (A and B) have been grown in Petri dishes. It
% appears that the mushrooms only can grow in specific positions in the
% Petri dish. You are asked to train a linear discriminant analysis (LDA)
% classifier to estimate the probability of a mushroom type growing at a
% given position in the Petri dish. It is a very time-consuming experiment,
% so only five training examples for each type of mushroom were
% collected.
% The training data are:
% Class 0: Mushroom type A and their grow positions (x,y):
% (1.00, 1.00)
% (2.20, -3.00)
% (3.50, -1.40)
% (3.70, -2.70)
% (5.00, 0)
% Class 1: Mushroom type B and their grow positions(x,y):
% ( 0.10, 0.70)
% (0.22, -2.10)
% (0.35, -0.98)
% (0.37, -1.89)
% (0.50, 0)
% Note: To train the LDA classifier to obtain the weight-vector W for
% classification, use the provided Matlab function: LDA.m
% What is the probability that the first training example of Mushroom Type
% A, with position (1.00, 1.00), actually belongs to class 1?

X = [1, 1; 2.2, -3; 3.5, -1.4; 3.7, -2.7; 5, 0;
    0.1, 0.7; 0.22, -2.1; 0.35, -0.98; 0.37, -1.89; 0.5, 0];    % Put all data into one vector
T = [zeros(5,1); ones(5,1)];                                    % Define ones and zeros for each class
W = LDA(X,T);                                                   % Run LDA linear discriminant analysis on the points
ex1 = [1; 1; 1];                                                % Set up training example
L = [ones(10,1) X] * W';                                        % Convert to LDA space
P = exp(L) ./ repmat(sum(exp(L),2),[1 2])                       % Return probability of belonging to each class

%Y = W*ex1;                                                      % Convert to LDA space
%exp(Y)./sum(exp(Y))                                             % Return
%probability of the point ex1 belonging to each class



























%% W1







%% Ex 1
% Read an image into the Matlab workspace and to get information about
% the dimensions of the image.

close all; clear; clc;
%Read image
mc = imread('metacarpals.png');

%Check size
size(mc)

%% Ex 2
% Display an image.
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
% Display an image histogram.
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
% Inspect pixel values in an image using both (x, y) and (row, column) pixel
% coordinates.
%Pixel value at (r,c)=(100,90)
pix_val = mc(100,90); %118

%% Ex 5
% Use the Matlab Image Tool (imtool) to visualize and inspect images.
imtool(mc);

%Background has lots of pixel values equal to 113

%% Ex 6
% Inspect RGB images in a color image.

% Resize an image.
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
% Transform a RGB image into a grey-level image (rgb2gray).

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

horns = imread('horns.jpg');
im_dark_gr = rgb2gray(horns);
figure
imhist(horns);

%% Ex 10
% Read DICOM files into the Matlab workspace and to get information
% about the image from the DICOM header.

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
% Use Matlab colormaps (colormap) to visualize 8-bit images.
close all; clear; clc;
mc = imread('metacarpals.png');
imshow(mc);
colormap(gca, jet);
colorbar;

%% Colour channels
% Visualise individual color channels in an RGB image.
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
% Change and manipulate individual color channels in an RGB image.
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
% Use the image contour tool (imcontour) to visualize greylevel contours in
% an image.

close all; clear; clc;
fing = imread('finger.png');
imshow(fing);

%%
figure;
imcontour(fing, 5); %Shows 5 levels of equally spaced contour lines
%Noisy background gives rise to contours

%%
% Use the image profile tool (improfile) to sample and visualise grey scale
% profiles.

%Look at profile across the bone
imshow(fing);
improfile;

%%
% Use the mesh tool to visualize a 2D image as a height map.

%Image plotted as landscape
mesh(double(fing));

%% W1b













%% Exercise 1b - PCA
%% Ex1v
% Load data from a text file into the Matlab workspace
% PCA Analysis exercise with iris data
clear; close all; clc;
load irisdata.txt;

% Create a data matrix from a text file
% Load data
X = irisdata(1:50,1:4)';

[M,N] = size(X);
% M is number of features and N is number of observations

%% Ex2
% Create vectors of individual measurements:
sep_L = X(1,:);
sep_W = X(2,:);
pet_L = X(3,:);
pet_W = X(4,:);

% Compute the variance of a set of measurements
% Compute variance of each feature:
var_sep_L = var(sep_L)
var_sep_W = var(sep_W)
var_pet_L = var(pet_L)
vvar_pet_W = var(pet_W)

%% Ex3

% Compute the covariance between two sets of measurements
% compute covariance matrix between sepal length and sepal width:
cov_sepL_sepW = cov(sep_L,sep_W)
% compute covariance matrix between sepal length and petal length:
cov_sepL_petL = cov(sep_L,pet_L)

%% Ex4 

% Use the Matlab function plotmatrix to visualise the covariance between
% multiple sets of measurements
% Plot scatterplots of all variables:
[~,ax]=plotmatrix(X');
ax(1,1).YLabel.String='Sepal L'; 
ax(2,1).YLabel.String='Sepal W'; 
ax(3,1).YLabel.String='Petal L'; 
ax(4,1).YLabel.String='Petal W'; 
ax(4,1).XLabel.String='Sepal L'; 
ax(4,2).XLabel.String='Sepal W'; 
ax(4,3).XLabel.String='Petal L'; 
ax(4,4).XLabel.String='Petal W'; 

% This plots shows that sepal W and sepal L is correlated (as seen in
% covariance matrix as well). The other variables are not significantly
% correlated, as indicated by the low covariance of sepal L and petal L

%% Ex5 - PCA computation:
% Compute the covariance matrix from multiple sets of measurements
% Compute the principal components using Eigenvector analysis (Matlab
% function eig).

data = X;

% subtract the mean for each dimension
mn = mean(data,2);
data = data - repmat(mn,1,N);

% calculate the covariance matrix
Cx = 1 / (N-1) * data * data';

% find the eigenvectors and eigenvalues
[PC, V] = eig(Cx);

% extract diagonal of matrix as vector
V = diag(V);

% sort the variances in decreasing order
[junk, rindices] = sort(-1*V);
V = V(rindices);
PC = PC(:,rindices);

% Project original measurements into principal component space
% project the original data set
signals = PC' * data;

%% Ex6
% Visualize how much of the total of variation each principal component
% explain.

% plot explained variance of principal components:
plot(V)
Vnorm = V / sum(V) * 100
plot(Vnorm, '*-')
ylabel('Percent explained variance')
xlabel('Principal component')
% It is clear, that the first component explains most of the variance, and
% the first 2-3 components will explain more that 90% of the variance

%% Ex7
% Use the Matlab function plotmatrix to visualise the covariance structure
% after projecting to principal component space.

[~,ax]=plotmatrix(signals');

% compute covariance of the two first components:
pv1 = signals(1,:);
pv2 = signals(2,:);
cov_pv1_2 = cov(pv1,pv2);
% They are not correlated at all, as the definition of the principal
% components is that they are uncorrelated with each other.















%% W2






%% Exercise 2 - Cameras
%% Ex2.1
% Create a Matlab function in a separate file
clc;clear all;
% Theta can be calculated by theta= arctan(b/a)
a = 10; b = 3;
theta = atand(b/a);  %theta=16.70 degress (atand = arctan in degrees)

%% Ex2.2
% The function can be written as:

% function b = CameraBDistance(f,g)
% %CameraBDistance returns the distance (b) where the CCD should be placed
% %when the object distance (g) and the focal length (f) are given
% b = 1/(1/f -1/g);

% Use your function to find out where the CCD should be placed when the focal
% length is 15 mm and the object distance is 0.1, 1, 5, and 15 meters.

% Create a Matlab function that uses the thin lens equation to compute
% either the focal length (f), where the rays are focused (b) or an object
% distance (g) when two of the other measurements are given

b1 = CameraBDistance(15,100);
disp(['With f=15mm and g=100mm we get b=',num2str(b1),'mm'])
b2 = CameraBDistance(15,1000);
disp(['With f=15mm and g=1000mm we get b=',num2str(b2),'mm'])
b3 = CameraBDistance(15,5000);
disp(['With f=15mm and g=5000mm we get b=',num2str(b3),'mm'])
b4 = CameraBDistance(15,15000);
disp(['With f=15mm and g=15000mm we get b=',num2str(b4),'mm'])

% What happens to the place of the CCD when the object distance is increased?
% -> the place of the CCD converges to the size of the focal length
%% Ex2.3
%info = imfinfo('DTUSigns.jpg')
%info.DigitalCamera

%% Ex2.4
clc;clear all;
% we wish to use mm for our constants in this exercise

G = 1800; % Thomas' height in mm
f = 5; % Cameras focal length in mm
g = 5000; % Thomas' distance to camera in mm
pixels = 640*480; % number of pixels in camera
area = 4.8*6.4; % area of camera in mm^2

%% 1) A focused image of Thomas is formed inside the camera. At which 
% distance from the lens?
% -> We use our CameraBDistance function:
b = CameraBDistance(f,g);
sprintf('Thomas is formed at the distance %f mm inside the camera ',b)

%% 2)How tall (in mm) will Thomas be on the CCD-chip? 
B = RealSizeOnCCD(G,b,g);

% function B = RealSizeOnCCD(G,b,g)
% %Input:
% %  G = Real height of the object in mm
% %  b = the distance (b) where the CCD should be placed in mm
% %  g = the object distance in mm
% %Output:
% %  B = Size of obejct in lens in mm

% B = G*b/g;
sprintf('Thomas will have the hegiht %f mm on the CCD chip ',B)

%% 3) What is the size of a single pixel on the CCD chip? (in mm)?
% -> The size of a single pixel can be found by taking the whole chip area
% divided with the number of pixels in the chip:
pixelsize = area/pixels;
sprintf('The size of a single pixel is %f mm2 on the CCD chip ',pixelsize)

%% 4) How tall (in pixels) will Thomas be on the CCD-chip? 
% Since a pixel is square we can find the side-length of the pixel by
pixelheight = sqrt(pixelsize);

% Now can Thomas' height in pixels be found
Hp = PixelSizeOnCCD(G,b,g,pixelheight);

% function  Hp = PixelSizeOnCCD(G,b,g,pixelheight)
% % Input:
% %   G = Real height of the object in mm
% %   b = the distance (b) where the CCD should be placed in mm
% %   g = the object distance in mm
% %   pixelheight = height of a pixel in mm
% % Output:
% %   Hp = Height in pixels
% 
% B = RealSizeOnCCD(G,b,g);
% Hp = B/pixelheight;

sprintf('Thomas will be %f pixels tall ',Hp)

%% 5)What is the horizontal field-of-view (in degrees)?
% See illustration on page 17. To compute v we divide the width of the chip
% with 2
W = 6.4/2;
% Same concept as in Ex2.1
v1 = atand(W/f)*2;

sprintf('The horizontal field of view will have the angle of %f degrees',v1)

%% 6) What is the vertical field-of-view (in degrees)?
% We have to take half of the CCD height
H = 4.8/2;
v2 = atand(H/f)*2;

sprintf('The vertical field of view will have the angle of %f degrees',v2)

% function [horizontal, vertical] = CameraFOV(f,CCD_height,CCD_width)
% %Input
% %  f: focal length
% %  CCD_height: Height of the CCD chip
% %  CCD_width: Width of the CCD chip
% %Output:
% % horizontal: horizontal field-of-view
% % vertical: vertical field-of-view
%
% horizontal = atand((CCD_width/2) /f) *2;
% vertical = atand((CCD_height/2) /f) *2;

%% Exam question on camera geometry
clc; clear all;

f = 65; % in mm
imagesize = [5120 , 4096]; % Pixelsize
g = 1200; % Distance to camera in mm
CCDsize = [10,8]; % in mm
r = 400; % Radius of melanoma in pixels

% First we want to find the radius of the melanoma in mm
Pixeltomm = CCDsize(1)/imagesize(1);
B = Pixeltomm * r;
% Now we can use b/B = g/G to calculate G:
G = g/(f/B); 
% Now we have the radius of the physical object in mm. We can then find the
% area by:
Area = G^2*pi;

sprintf('The physical area of the melanoma is %f',Area) % Answer is option 1
%% Exam question of field-of-view
clc;clear all;
% We have to take half of v to get our triangle that calculates half of the 
% length of the finger. Then we multiply the result by 2 to get full length

v = 15/2; % Degrees
g = 31.5; % cm

length = tand(v)*g*2;

sprintf('The finger has the length of %.2f cm',length) % The answer is option 2













%% W3





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
% Implement and test a function that can do linear histogram stretching of
% a grey level image.

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
% Implement and test a function that can perform gamma mapping of a grey
% level image.

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
% Use the Matlab function imadjust to modify grey scale images.

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
% Implement and test a function that can threshold a grey scale image.

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
% Perform RGB thresholding in a color image.

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
% Convert a RGB image to HSV using the Matlab function rgb2hsv.
% Visualise individual H, S, V components of a color image.
% Implement and test thresholding in HSV space.

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














%% W3b






%% Read yale data
clear; clc; close all;
P = 'YaleSubset\';
D = dir(fullfile(P, '*.png'));

N = numel(D);
% load first image to get size information
img = imread(fullfile(D(1).folder, D(1).name));

%% Exercise 1
% Create an empty data matrix that can hold N images and M measurement
% per image.

H = size(img, 1);
W = size(img, 2);
M = H * W;

data = zeros(M, N);

%% Exercise 2
% Use the Matlab function reshape to transform an image into a column
% matrix.
% Read a set of image files and put them into one data matrix.

for k=1:N     
      img = imread(fullfile(D(k).folder, D(k).name));
      tt = reshape(img, [], 1);
      data(:, k)=tt;
end
%% Exercise 3
% Compute the average column.
% Visualise eigenvectors as images.

% Average image
meanI = mean(data, 2);
I = reshape(meanI, H, W);
imshow(I,[]);

%% Exercise 4
% Compute the eigenvalues and eigenvector of a data matrix.

%I = uint8(reshape(meanI, H, W));

[Vecs, Vals, Psi] = pc_evectors(data, 30); 

%% Exercise 5
subplot(1,2,1),
plot(Vals);                       % To plot the eigenvalues

pc1_variance = Vals(1)/sum(Vals(:)); % 39.34%
pc2_variance = Vals(2)/sum(Vals(:)); % 8.24% 

% Normalize the eigenvalues to get percent explained variation
Valsnorm = Vals / sum(Vals) * 100;
subplot(1,2,2)
plot(Valsnorm, '*-')
ylabel('Percent explained variance')
xlabel('Principal component')

%% Exercise 6
% Synthesise facial image by a linear combination of the average face and a
% set of eigenvectors.
% Subtract the average face from a new face image.

% First eigenvector
eigvec1 = Vecs(:,1);
v1img = reshape(eigvec1, H, W);

figure;
subplot(1,2,1)
imshow(v1img, []);
subplot(1,2,2)
imshow(-v1img, []);

% Second eigenvector
eigvec2 = Vecs(:,2);
v2img = reshape(eigvec2, H, W);

figure;
subplot(1,2,1)
imshow(v1img, []);
title('1st eigenvector')
subplot(1,2,2)
imshow(v2img, []);
title('2nd eigenvector')

% PC1 explains the variance in nose structure 
% Harder to see for PC2 but mouth and eyebrows variance

%% Exercise 7

newface = 1000 * Vecs(:, 1) - 3000 * Vecs(:, 2) + meanI;
syntheticFace = reshape(newface,H,W);
imshow(syntheticFace,[])


%% Exercise 8+9
% Project a new face image to the face eigenvector space.

realFace = imread('FaceCroppedGray.png');
imshow(realFace, [])
RealFaceMat = double(reshape(realFace, [], 1));
RealFaceMat = RealFaceMat - meanI;
Proj = Vecs(:, 1:2)' * RealFaceMat; 

%% Exercise 10
% Synthesise the closest possible synthetic face to a real facial image by a
% linear combination of the average face and a set of eigenvectors.

figure;
Reconstruct = Proj(1) * Vecs(:, 1) + Proj(2) *Vecs(:, 2) + meanI;
ReconImg = reshape(Reconstruct, H, W);
imshow(ReconImg, []);

Proj = Vecs(:,1:6)' * RealFaceMat; 

figure;
tI2 = meanI + Proj(1) * Vecs(:, 1) + Proj(2) * Vecs(:, 2) + Proj(3) * Vecs(:, 3) + Proj(4) * Vecs(:, 4) + Proj(5) * Vecs(:, 5) + Proj(6) * Vecs(:, 6);
I2 = reshape(tI2, H, W);
imshow(I2, []);

% Slightly more detailed image but still does not look like the
% original face. For instance, the beard becomes more
% emphasized in the reconstructed image with more eigenvectors.
% To obtain a good reconstruction, many more images would be required.














%% W4







clc; clear; close all;

%% Exercise 1
% Use the Matlab imfilter function to filter an image using a given filter
% kernel.

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
% Filter an image using imfilter using zero-padding and border replication.

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
% Compute a mean filtered image using dierent kernel sizes.

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
% Remove salt and pepper noise using a median filter.

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
% Filter an image using the Matlab fspecial function.
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
% Compute and visualise edges in an image using Sobel and Prewitt filters.

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
% Filter an image using the Matlab edge function.
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
% Filter an image using a Gaussian filter using the Matlab fspecial func-
% tion.
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














%% W4b







%% Intro
clear; close all; clc
load Image1.mat
imagesc(Image1);
imagegrid(gca,size(Image1));
colormap(gca,hot);

%% Erosion
% Use the Matlab strel function to create structuring elements.
% Compute an eroded binary image using the Matlab imerode function.

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
% Compute a dilated binary image using the Matlab imdilate function.

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
% Implement and test a compound morphological operation.
% Compute an opened binary image using the Matlab imopen function.

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
% Compute a closed binary image using the Matlab imclose function.

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
% Select appropriate morphological operations and structuring elements for
% noise removal in binary images.

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
% Compute an approximate boundary of a binary object using a combination
% of dilations, erosions and image subtractions.

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














%% W5







%% Ex 1 & 2
% Use the Matlab function bwlabel to create labels from a binary image
% using both 4- and 8-connectivity.

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
% Visualize labels using the Matlab function label2rgb

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
% Compute BLOB features using the Matlab function regionprops includ-
% ing BLOB area and perimeter.
 stats8 = regionprops(L8, 'Area');
 val1= stats8(1).Area
 val2= stats8(2).Area
 val3= stats8(3).Area
%% Ex 6
allArea = [stats8.Area]
%% Ex 7
% Select BLOBs that have certain features using the Matlab function ismember.
idx = find([stats8.Area] > 16);
BW2 = ismember(L8,idx);
 
imagesc(BW2);
imagegrid(gca,size(BW2));
colormap(hot);

%% Ex 8
% Extract BLOB features and plot feature spaces as for example area versus
% perimeter.

stats8 = regionprops(L8, 'All');
 
allPerimeter = [stats8.Perimeter]
perimeter_20 = sum(allPerimeter>20)
plot(allArea, allPerimeter, '*'), xlabel('Area'), ylabel('Perimeter');


%% Chemometec U2OS cell analysis - raw images
% Crop regions from an image using the Matlab function imcrop.

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
% Select an appropriate threshold by inspecting the histogram of an image.

imhist(Im)
% 10 is choosen, zoom in to see the values in the histogram
BW = (Im > 10);
figure, imshow(BW); title('Thresholded image');
%% Ex 14
% Remove BLOBs at the image border using the Matlab function imclearborder.

BWc = imclearborder(BW);
figure, imshow(BWc); title('Thresholded image - border cells removed');
%% Ex 15 
% Choose a set of BLOB features that separates objects from noise.
% Implement and test a simple function that can separate objects from noise
% using BLOB extraction and BLOB features classification.
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














%% W6








% Initial commands and loading
clc; close all; clear all;
ct2 = dicomread('CTangio2.dcm');
I2 = imread('CTAngio2Scaled.png');
imshow(I2);

%% Exercise 1
% Use the Matlab function roipoly to select regions in image that each
% represent a defined class.

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
% Plot histograms and compute the average and standard deviations of the
% pixel values in each of the pre-defined classes.

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
% Fit a Gaussian to a set of pixel values using the Matlab function normpdf.

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
% Visualize and evaluate the class overlap by plotting fitted Gaussian func-
% tions of each pre-defined class.

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
% Do pixel classification of an image using minimum distance classification.
% Determine the class ranges in a parametric classifier by visual inspection
% of fitted Gaussian distributions.

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
% Do colour classification by selecting class ranges in RGB space.

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
% Visually evaluate colour classification by classifying unseen images.

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














%% W6b
% Implement, train and evaluate multi-dimensional segmentation using a
% Linear Discriminate classifier i.e. Fisherman' Linear discriminant analysis
% To visualise the 1D intensity histograms of two difierent image modalities
% that contain difierent intensity information of the same image features.
% To identify the expected intensity thresholds in each of the 1D histograms
% that best segment the same feature in the two image modalities.







% 02502 - Image Analysis DTU 
% Exercise 6b on Advanced segmentation
% Topic: Fisherman's Linear discriminant analysis for segmentation
%
%The exercise is to perform advanced segmentation of two brain tissue types. 
% To improve the segmentation result over the simpler intensity histogram thresholding methods we 
% will use two input features compared to a single. 
% The two input features are two different image modalities acquired on the same brain. 
%
% The two tissue types to be segmented are: 
%   I) The White Matter (WM) is the tissue type that contain the brain network - like the cables 
%      in the internet. 
%.     The  WM ensure the communication flow between functional brain regions. 
%   II) The Grey Matter (GM) is the tissue type that contain the cell bodies at the end of the 
%       brain network and are the functional units in the brain. 
%.      The functional units are like CPUs in the computer. 
%.      They are processing our sensorial input and are determining a reacting 
%       to these. It could be to start running.
% 
% Provided data:
% ex6_ImgData2Load.mat contain all image and ROI data which are loaded into the 
% variables:
%   ImgT1 - One axial slice of brain using T1W MRI acquisition
%   ImgT2 - One axial slice of brain using T2W MRI acquisition
%   ROI_WM - Binary training data for class 1: Expert identification of voxels belonging to 
% tissue type: White Matter
%   ROI_GM - Binary training data for class 2: Expert identification of voxels belonging to 
% tissue type: Grey Matter
%
% LDA-m  Is matlab function that realise the Fisher's linear discriminant analyse as described in Note for lecture 6
%
% Exercise -  You simply go step-by-step and fill the command
% lines and answer/discuss the questions.
%
% October 2020, Tim Dyrby, tbdy@dtu.dk

%Clean up variables and figures
clear all
close all

% 0) Set path to working directory using cd('path') where image data and
% the LDA.m function are placed. 

%cd('/mnt/projects/timd/02502/Exercise_week7_LDA')
load('ex6_ImagData2Load.mat')


%% 2) Display both the T1 and T2 images, their histograms and scatter plots 
% To visually the 2D histogram of two image modalities that map the same
% object but with difierent intensity information.
% To interpret the 2D histogram information by identifying clusters of 2D
% intensity distributions and relate these to features in the images.

% Tips: Use the 'imagesc()', 'histogram()' and 'scatter()' functions
% Add relevant title and label for each axis. One can use 'subplot()' to show more
% subfigures in the same figure 


%Solution:
figure(1), colormap('gray')

subplot(2,2,1)
imagesc(ImgT1), title('ImgT1')
subplot(2,2,2)
imagesc(ImgT2), title('ImgT2')
subplot(2,2,3)
histogram(ImgT1), title('Histogram ImgT1')
subplot(2,2,4)
histogram(ImgT2), title('Histogram ImgT2')

figure(2), hold on
scatter(ImgT1(:),ImgT2(:),'xb'), title('Scatter T1 vs T2')
xlabel('ImgT1')
ylabel('ImgT2')


%% 3) Place trainings examples i.e. ROI_WM and ROI_GM into variables C1 and C2 representing  class 1 and class 2 respectively. 
% To draw an expected linear hyper plan in the 2D histogram that best
% segment and feature in the two image modalities

% Show in a figure the manually expert drawings of the C1 and C2 training
% examples. Tips: use 'imagesc()'

C1=ROI_WM;
C2=ROI_GM;

figure(3), hold on
subplot(1,2,1)
imagesc(C1), title('Training data WM')
subplot(1,2,2)
imagesc(C2), title('Training data GM')

%% 4) For each binary training ROI find the corresponding training examples in ImgT1 and ImgT2. 
% Later these will be extracted for LDA training. 
% Tips: Use the 'find()' function which return the index to voxels in the image full filling e.g. intensity values >0 hence belong to a given class. 
% Name the index variables qC1 and qC2, respectively.

%Solution:
qC1=find(C1(:)>0);
qC2=find(C2(:)>0);

%% 5) Make a training data vector (X) and target class vector (T) as input for the 'LDA()' function. 
% T and X should have the same length of data points.
% - X: training data vector should first include all data points for class 1 and
% then the data points for class 2. Data points are the two input features ImgT1, ImgT2
% - T: Target class identifier for X where '0' are Class 1 and a '1' is Class 0.

%Solution:
X=[ImgT1(qC1), ImgT2(qC1)];
X=[X;[ImgT1(qC2), ImgT2(qC2)]];

T=[zeros(length(qC1),1); ones(length(qC2),1)];

%% 6) Make a scatter plot of the training points of the two input features for class 1 and class 2. 
% Show Class 1 and 2 as green and black circles, respectively. 
% Add relevant title and labels to axis

%Solution:
figure(4), hold on
scatter(ImgT1(qC1),ImgT2(qC1),'og')
scatter(ImgT1(qC2),ImgT2(qC2),'ok')
title('Scatter plot of Class1 and Class2 training points')
xlabel('ImgT1')
ylabel('ImgT2')


%% 7) Estimate the Fisher discriminant function coefficients (w0 and w) i.e. y(x)~ w+w0 given X and T by using the
% To relate the Bayesian theory to a linear discriminate analysis classifier
% for estimating class probabilities of segmented features.

%'W=LDA()' function.
% Tip: Read the Bishop note on Chapter 4. 
%The LDA function outputs W=[[w01 w1]; [w02 w2]] for class 1 and 2 respectively. 

W = LDA(X,T);


%% 8) Classification using W: Calculate linear scores for *all* image data points within the brain slice to be classified i.e. y(x)~ w+w0

Xall=[ImgT1(:),ImgT2(:)];
Y = [ones(length(ImgT1(:)),1) Xall] * W';
 
%% 9) Classification: Calculate the posterior probability i.e. P(X|C1) of a data point belonging to class 1
% Note: Using Bayes: Since y(x) is the exponential of the posterior probability 
% hence we take exp(y(x)) to get P(X|C1)=(P(X|mu,var)P(C1))) and divide with the conditional probability (P(X)) as normalisation factor.  
 
 PosteriorProb = exp(Y) ./ repmat(sum(exp(Y),2),[1 2]);

 
%% 10) Segmentation: Find all voxles in the T1w and T2w image with P(X|C1)>0.5 as belonging to Class 1 using the 'find()' function. 
% Similarly, find all voxels belonging to class 2.

%Solution:
qSegC1=find(PosteriorProb(:,1)>0.5);
qSegC2=find(PosteriorProb(:,1)<=0.5);



%% 11) Show scatter plot of segmentation results as in 6). 
% To judge if the estimated linear or a non-linear hyper plan is optimal
% placed for robust segmentation of two classes.

% Q1: Can you identify where the hyper plan is placed i.e. y(x)=0? 
% -->A1: Yes, all the segmented points show a clear line at their
% interface=Hyperplan
%
% Q2: Is the hyperplan positioned as you expected?
% -->A2: Yes and no - I would expect all black and green training data points to be on each side of the hyperplan. 
%
% Q3: Would segmentation be as good as using a single image modality using
% thresholding?
% -->A3: If that should be the case the hyperplan would be orthogonal to one
% of the features. It is not. So no.
%
% Q4: From the scatter plot does the segmentation results make sense? Are the two tissue types segmented correctly .
% -->A4: Looks good, but hard to say from a scatter plot

%Solution:
figure(5), hold on, colormap('gray')
scatter(ImgT1(:),ImgT2(:),'xb')
scatter(ImgT1(qSegC1),ImgT2(qSegC1),'or')

scatter(ImgT1(qC1),ImgT2(qC1),'og')
scatter(ImgT1(qC2),ImgT2(qC2),'ok')
xlabel('ImgT1')
ylabel('ImgT2')
title(' Hyperplan?: C1 and C2 training samples onto all segmented voxels' )


%% 12) Show segmentation results and training examples of class 1 and 2 in sub figures.  
% Q5: Are the training examples representative for the
% segmentation results? Are you surprised that so few training examples perform
% so well? Do you need to be an anatomical expert to draw these?
%
% -->A5: Yes they are representative.  Yes and no; one do not need to be an
% anatomical expert, but you need to understand what the method needs as input to perform well.
%
% Q6: Compare the segmentation results with the original image. Is the segmentation results satisfactory? Why not?
% -->A6: Yes, the WM and GM classes looks correct segmented when comparing to the anatomical structures in ImgT1 and ImgT2.
%
% Q7: Is one class completely wrong segmented? What is the problem?
% --> A7: Yes -The problem is that all background and scull voxels also are classified as
%     GM. Solution is to identify the background voxels and exclude these
%     from the segmentation or to add an extra class for background voxels.

%Solution:
figure(6),title('LDA Segmentation results: Class 1, WM')
colormap('gray')
hold on

subplot(3,2,1)
imagesc(ImgT1),title('Feature 1: T1 Image')

subplot(3,2,2)
imagesc(ImgT1),title('Feature2: T2 Image')

subplot(3,2,3)
SegT1=ImgT1;
SegT1(qC1)=50000; % Set C1 pixels to value of 5000 
imagesc(SegT1), title('Training samples C1 (WM)')

subplot(3,2,4)
SegT1=ImgT1;
SegT1(qSegC1)=50000; % Set C1 pixels to value of 5000 
imagesc(SegT1),title('Segmentation result: C1 (WM)')


subplot(3,2,5)
SegT1=ImgT1;
SegT1(qC2)=50000; % Set C1 pixels to value of 5000 
imagesc(SegT1), title('Training samples C2 (GM)')

subplot(3,2,6)
SegT1=ImgT1;
SegT1(qSegC2)=50000; % Set C2 pixels to value of 5000 
imagesc(SegT1),title('Segmentation result: C2 (GM)')














%% W7








%% Geometric Transformations
%% Exercise 1
% The first step is to create the 2D points of the grid. Here the
% function meshgrid is useful. Look at the documentation in Matlab:
clear, clc, close all;
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
% Implement and apply simple point transformations (including rotation and
% scaling) to 2D points.

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
% Implement and apply simple transformations to images using both forward
% and backward mapping.

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
% Annotate images with corresponding landmarks. For example by using
% the Matlab tool cpselect.

figure;
plot(fixedpoints(:,1), fixedpoints(:,2), 'b*-', ...
movingpoints(:,1), movingpoints(:,2), 'r*-');
legend('Hand 1 - The fixed image', 'Hand 2 - The moving image');
axis ij; % This reverses the direction of the axis

%% Exercise 11
% Compute the sum of squared dierence objective function for two sets of
% corresponding landmarks.

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
% Compute the center of mass for a set of landmarks.
% Align two sets of landmarks by aligning their center of masses.

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
% Use the Matlab function fitgeotrans to compute the transformation that
% transforms one set of landmarks into another set of landmarks.

% Compute the transform that makes hand2 fit hand1:
mytform = fitgeotrans(movingpoints, fixedpoints,'NonreflectiveSimilarity');

%% Exercise 15
% Use the Matlab function transformPointsForward to apply a transfor-
% mation to a set of points.

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
% Use the Matlab function imwarp to apply a transformation to an image.

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














%% W9











% Start by clearing the workspace
clc; clear; close all;

% Test that the image is loaded correctly
ct = dicomread('1-105.dcm');
imshow(ct, [-100, 200]);

%% Exercise 1
% Use the Matlab function roipoly to mark representative regions in a
% DICOM image.

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
% Select suitable Hounsfield unit ranges based on histogram inspection.
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
% Use morphological opening and closing to repair holes in objects and sep-
% arate objects in binary images.

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
% Compute BLOBs in a binary image using the Matlab function bwlabel.

L8 = bwlabel(IMorph, 8);
RGB8 = label2rgb(L8);
figure
% Showing the labeled image, the spleen
% has a seperate label from the other
% organs/tissue in the image which means 
% the labelling works
imagesc(RGB8);
%% Exercise 9
% Compute BLOB features using the Matlab function regionprops.

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
% Compute the DICE score between a computed segmentation and a ground
% truth segmentation.

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