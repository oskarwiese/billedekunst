clc, clear, close all
path_folder = '/home/oc/Documents/billedekunst/' ;
cd(path_folder)
addpath data
%% 
% Load in data from txt file for PCA analysis
M = load('data/irisdata.txt');
M = M(:,1:4); % OPEN DATA TO CHECK THAT IT HAS THE COLUMNS YOU ACTUALLY WANT BOI
M = M - mean(M,1); %subtracting mean
%% Q1  % TO FIND THE EXPLAINED VARIANCE
% The irisdata.txt file contains measurements from 150 iris flowers. The
% measurements are the sepal length, sepal width, petal length and petal
% width. So you have M=4 features, N=150 observations. A principal
% component analysis (PCA) should be done on these data. How many percent
% of the total variation do the two first principal components explain?
Cx = cov(M); % Find COVARIANCE MATRIX
[PC, V] = eig(Cx); %
V = diag(V);
[junk, rindices] = sort(-1*V);
V = V(rindices);
PC = PC(:,rindices);
Vnorm = V / sum(V) * 100;
plot(Vnorm, '*-')
ylabel('Percent explained variance')
xlabel('Principal component')
Vnorm(1)+Vnorm(2) 
%% Q2
% The irisdata.txt file contains measurements from 150 iris flowers. The
% measurements are the sepal length, sepal width, petal length, and petal
% width. So you have M=4 features, N=150 observations. A principal
% component analysis (PCA) should be done on these data. A er the PCA, the
% flower data (sepal lengths and widths, petal lengths and widths) are
% projected into PCA space. What are the projected values of the first
% flower?
signals = PC' * M'; % M IS DATA SO YOU GET INTO EIGENSPACE 
signals(:,1)' %%-2.6841    0.3194    0.0279   -0.0023
%%
[vectors,values,psi] = pc_evectors(M',4);

sigs = vectors'*M';
sigs(:,1)
%% Q3
% The photo called sky_gray.png is loaded and a linear histogram stretching
% is performed so the new image has a maximum pixel value of 200 and a
% minimum pixel value of 10. What is the average pixel value of the new
% image?
SG = double(imread('data/sky_gray.png'));
%imshow(SG)
min_d = 10 ;
max_d = 200 ;
st_SG = (max_d-min_d) /(max(SG(:)) - min(SG(:))) * (SG - min(SG(:))) + min_d;
mean(st_SG(:)) %%87
%% Q4
% The photo called sky.png is loaded and an RGB threshold is performed with
% the limits R <  100, G > 85, G < 200, and B > 150. Pixels with values
% within these limits are set to foreground and the rest of the pixels are
% set to background. The resulting 2D binary image is morphologically
% eroded using a disk- shaped structuring element with radius=5. When doing
% an erosion the pixels beyond the image border are assigned a value of 1
% (the default Matlab behavior). How many foreground pixels are there in
% the final image?
SG_RGB = double(imread('data/sky.png'));
lgcl_SG = SG_RGB(:,:,1) < 100 & SG_RGB(:,:,2) > 85 & SG_RGB(:,:,2) < 200 & SG_RGB(:,:,3) > 150;
se1 = strel('disk',5); % structuring element with radius=5
out = imerode(lgcl_SG,se1); % EROSION
sum(out(:)) %19977
%% Q5
% The photo called flower.png is loaded and it is converted from the RGB
% color space to the HSV color space. Secondly, a threshold is performed on
% the HSV values with the limits H < 0.25, S > 0.8 and V > 0.8. Pixels with
% values within these limits are set to foreground and the rest of the
% pixels are set to background. Finally, a morphological opening is
% performed on the binary image using a disk-shaped structuring element
% with radius=5. When doing a dilation, pixels beyond the image border are
% assigned a value of 0 and when doing an erosion the pixels beyond the
% image border are assigned a value of 1 (the default Matlab behavior).
% What is the number of foreground pixels in the resulting image?
flwer = imread('data/flower.png');
flwer_hsv = rgb2hsv(flwer);
lgcl_flwer = flwer_hsv(:,:,1) < 0.25 & flwer_hsv(:,:,2) > 0.8 & flwer_hsv(:,:,3) > 0.8;
se1 = strel('disk',5);
out = imopen(lgcl_flwer,se1);
sum(out(:)) %5665
%% Q6
% Five photos have been taken. They are named car1.jpg - car5.jpg and they
% have the dimensions (W=800, H=600). A principal component analysis (PCA)
% is performed on the grey values of the five images. You can use the two
% helper functions pc_evectors.m and sortem.m to compute the PCA. How much
% of the total variation in the images is explained by the first principal
% component?
Mc = zeros(800*600,5);
for i = 1:5
    str = ['data/car',num2str(i),'.jpg'];
    tmp_pic = double(imread(str));
    Mc(:,i) = tmp_pic(:) - mean(tmp_pic(:));
end
[vectors,values,psi] = pc_evectors(Mc,5);
values(1)/sum(values)
[nv,nd] = sortem(vectors',diag(values));
vnorm = nd/sum(nd(:)) * 100
%% Q7
% The photo called sky_gray.png is transformed using a gamma mapping with
% gamma=1.21. The output image is filtered using a 5x5 median filter. What
% is the resulting pixel value in the pixel at row=40, column=50 (when
% using a 1-based matrix-based coordinate system)?
SG = double(imread('data/sky_gray.png'));
gamma = 1.21 ;
SG = 255*((SG / 255).^gamma); % GAMS MIG LIGE BROOO ??HLLLL
kernel_size = [5,5] ;
SG = medfilt2(SG,kernel_size);
round(SG(40,50))
%% Q8
% The photo called flowerwall.png is filtered using an average filter with
% a filter size of 15. The filtering is performed with border replication.
% What is the resulting pixel value in the pixel at row=5 and column=50
% (when using a 1-based matrix-based coordinate system)?
% DEFINE SIZE OF KERNEL 
FW = double(imread('data/flowerwall.png'));
kernel_size = 15;
se = ones(kernel_size) / kernel_size .^ 2;
out = imfilter(FW,se);
out(5,50) %167
%% Q9
% A photo has been taken of a set of floorboards (floorboards.png) and the
% goal is to measure the amounts of knots in the wood. First, a threshold
% of 100 is used, so pixels below the threshold are set to foreground and
% the rest is set to background. To remove noise a morphological closing is
% performed with a disk-shaped structuring element with radius=10 followed
% by a morphological opening with a disk-shaped structuring element with
% radius=3. When doing a dilation, pixels beyond the image border are
% assigned a value of 0 and when doing an erosion the pixels beyond the
% image border are assigned a value of 1 (the default Matlab behavior).
% Finally, all BLOBs that are connected to the image border are removed.
% How many foreground pixels are remaining in the image?
FB = double(imread('data/floorboards.png'));
lgcl_FB = FB < 100; % The threshold of 100 from question
se1 = strel('disk',10); % Radius of disk
se2 = strel('disk',3); % Radius of disk
fnl_FB = imclearborder(imopen(imclose(lgcl_FB,se1),se2));
L8 = bwlabel(fnl_FB,8);
imagesc(L8);
colormap(hot);
title('8 connectiviy')
sum(fnl_FB(:)) %6735
%% Q10
% A photo has been taken of a set of floorboards (floorboards.png) and the
% goal is to measure the amounts of knots in the wood. First, a threshold
% of 100 is used, so pixels below the threshold are set to foreground and
% the rest is set to background. To remove noise a morphological closing is
% performed with a disk-shaped structuring element with radius=10 followed
% by a morphological opening with a disk-shaped structuring element with
% radius=3. When doing a dilation, pixels beyond the image border are
% assigned a value of 0 and when doing an erosion the pixels beyond the
% image border are assigned a value of 1 (the default Matlab behavior). A
% BLOB analysis is performed where all BLOBS are found using
% 8-connectivity. All BLOBs that are connected to the image border are
% removed. The area of the found BLOBs are computed and only the BLOBs with
% an area larger than 100 pixels are kept. How many BLOBs are found in the
% final image?
FB = double(imread('data/floorboards.png'));
lgcl_FB = FB < 100;
se1 = strel('disk',10); % Radius of disk
se2 = strel('disk',3); % Radius of disk
fnl_FB = imclearborder(imopen(imclose(lgcl_FB,se1),se2));
L8 = bwlabel(fnl_FB,8); % Make unique label for each blob
stats8 = regionprops(L8, 'Area'); % regionprop has all properties for blobs bounding box etc etc
bw2 = numel(find([stats8.Area] > 100)) %16
stats8 = regionprops(L8, 'BoundingBox'); % Regionpro can be used for BoudningBox ratio as well
BoundingBois = numel(find([stats8.BoundingBox] > 50)) 
%% Q11
% The binary image books_bw.png contains letters. A BLOB analysis is
% performed using 8-connectivity. For each BLOB, the area and the perimeter
% is computed. The BLOBs with area > 100 and perimeter > 500 are kept.
% Which letters are visible in the final image?
im = imread('data/books_bw.png');
im_lab = bwlabel(im,8); % Find BLOBS USING 8 CONNECTIVITY
stats = regionprops(im_lab,'all');
idx = find([stats.Area] > 100 & [stats.Perimeter] > 500 );
im_bw = ismember(im_lab,idx);
imshow(im_bw)
%% Q12
% Seven corresponding landmarks have been placed on two images (cat1.png
% and cat2.png). The landmarks are stored in the files catfixedPoints.mat
% and catmovingPoints.mat. What is the sum of squared di erences between
% the fixed and the moving landmarks?
load('data/catfixedPoints.mat')
load('data/catmovingPoints.mat')
sum((fixedpoints-movingpoints).^2,'all')
%% Q13
% Seven corresponding landmarks have been placed on two images (cat1.png
% and cat2.png). The landmarks are stored in the files catfixedPoints.mat
% and catmovingPoints.mat. A similarity transform (translation, rotation,
% and scaling) has been performed that aligns the moving points to the
% fixed points.  The computed transform is applied to the cat2.png photo.
% How does the resulting image look like?
load('data/catfixedPoints.mat')
load('data/catmovingPoints.mat')
transform = fitgeotrans(movingpoints, fixedpoints,'NonreflectiveSimilarity'); % OUTPUT IS A TRANSFORMATION MATRIX USED TO TRANSFORM IMAGE E.G RANSLATION, ROTATION, SCALING
forward = transformPointsForward(transform, movingpoints);
disp(forward)
disp(imwarp(movingpoints, transform)) % THIS ILLUSTRATES THE STUPIDITY OF MATLAB
% plot them together with the points from hand1. What do you observe? %VERY
% GOOD COMMENT IDIOT
cat = im2double(imread('data/cat2.png'));
cat_moved = imwarp(cat, transform);

%Show the transformed version of hand2 together with hand1. What do you observe?
subplot(1,2,1)
imshow(cat)
title('cat')
subplot(1,2,2)
imshow(cat_moved)
title('cat moved')
%% Q14
% An abdominal scan has been acquired on a CT scanner. One of the slices of
% the scan is stored as a DICOM file called 1-179.dcm. An expert has marked
% a part of the liver as a binary mask (region of interest). The binary
% mask is stored as the file LiverROI.png. By using the DICOM image and the
% mask image, the image values in the DICOM image inside the mask (the
% liver) can be extracted.
%  
% The average value and the standard deviation of the extracted pixel
% values are computed. A low threshold, T1, is defined as the average value
% minus the standard deviation and a high threshold, T2, is defined as the
% average value plus the standard deviation.
%  
% Finally, a segmentation of the DICOM image (1-179.dcm) is made where all
% pixels with values > T1 and < T2 are set to foreground and 
% the rest are
% set to background. How many foreground pixels are there?
dc = double(dicomread('1-179.dcm'));

liver = imread('data/LiverROI.png');

dc_l = dc(liver);
T = mean(dc_l,'all')+[-1,1]*std(dc_l,[],'all') % M??SKE KOGT NOTATION KAN V??RE BRUGBART TIL EKSAMEN

dc_t = dc>T(1) & dc<T(2) ;
sum(dc_t(:))
%% Q15
% An abdominal scan has been acquired on a CT scanner. One of the slices of
% the scan is stored as a DICOM file called 1-179.dcm. A low threshold, T1
% = 90, and a high threshold, T2 = 140, are defined. The pixel values of
% the DICOM image are segmented by setting all pixel values that are >T1
% and <T2 to foreground and the rest are set to background. The binary
% image is processed by first applying a morphological closing using a
% disk-shaped structuring element with radius=3 followed by a morphological
% openingwith the same structuring element. When doing a dilation, pixels
% beyond the image border are assigned a value of 0 and when doing an
% erosion the pixels beyond the image border are assigned a value of 1 (the
% default Matlab behavior). In the final step, a BLOB analysis is done
% using 8-connectivity. The largest BLOB is found. The area (in pixels) of
% the largest BLOB is:
dc = double(dicomread('data/1-179.dcm'));
T = [90, 140];
dc_t = dc>T(1) & dc<T(2) ;

se = strel('disk',3);

dc_close = imclose(dc_t,se);
dc_open = imopen(dc_close,se);

im_lab = bwlabel(dc_open,8);
stats = regionprops(im_lab,'area');
max([stats.Area])


%% Q17
% NASA's Mars Perseverance rover has explored Mars since its landing at the
% beginning of 2021. To explore the surface of Mars, the rover uses a
% custom build camera. Now the rover has discovered three spectral peaks
% that might reflect di erent types of cosmic dust.  Each dust spectra
% appears to follow a normal distribution.  The parametric distributions of
% the three dust classes are N(7,2*2), N(15,5*5),  and N(3,5*5). NASA asks
% help to define the thresholds to perform robust classification. They wish
% to perform a parametric classification of the three dust classes.
%  
% What signal thresholds should NASA use?

xrange = 0:0.01:20;  % MAY NEED TO CHANGE THRESHOLD NIGGA
pdf1 = normpdf(xrange, 3, 5); % NORMPDF INDTAGER SD OG IKKE VAR, svin
pdf2 = normpdf(xrange, 7, 2);
pdf3 = normpdf(xrange, 15, 5);

plot(xrange,[pdf1;pdf2;pdf3])
% The Gaussians crosses in 4.24 and 10.26

%% Q18
% The normalised cross correlation (NCC) between the image and the template
% is computed. What is the NCC in the marked pixel in the image?

im = [167,193, 180;
      9, 189, 8;
      217, 100, 71];
tem = [208, 233, 71;
       231, 161, 139;
       32, 25, 244];
   
sum(im.*tem,'all')/sqrt(sum(im(:).^2)*sum(tem(:).^2))

%% Q19
% A company is making an automated system for fish inspection. They are
% using a camera with a CCD chip that measures 5.4 x 4.2 mm and that has a
% focal length of 10 mm. The camera takes photos that have dimensions 6480
% x 5040 pixels and the camera is placed 110 cm from the fish, where a
% sharp image can be acquired of the fish.
%  
% How many pixels wide is a fish that has a length of 40 cm?

f = 10;
g = 1100;
fish = 400;
pixel_mm = 6480/5.4;

b = 1/(1/f-1/g);
% assume b = f ?
b = f;
B = b*fish/g;
B*pixel_mm

%% Q20
% Two types of mushrooms (A and B) have been grown in Petri dishes. It
% appears that the mushrooms only can grow in specific positions in the
% Petri dish.  You are asked to train a linear discriminant analysis (LDA)
% classifier to estimate the probability of a mushroom type growing at a
% given position in the Petri dish.  It is a very time-consuming
% experiment, so only five training examples for each type of mushroom were
% collected.
%  
% The training data are:
%  
% Class 0: Mushroom type A and their grow positions (x,y):
%      (1.00, 1.00) (2.20, -3.00) (3.50, -1.40) (3.70, -2.70) (5.00, 0)
%  
% Class 1: Mushroom type B and their grow positions(x,y):
%     ( 0.10, 0.70)
%      (0.22, -2.10) (0.35, -0.98) (0.37, -1.89) (0.50, 0)
%  
% Note: To train the LDA classifier to obtain the weight-vector W for
% classification, use the provided Matlab function: LDA.m What is the
% probability that the first training example of Mushroom Type A, with
% position (1.00, 1.00),  actually belongs to class 1?
X = [1, 1; 2.2, -3; 3.5, -1.4; 3.7, -2.7; 5, 0;
    0.1, 0.7; 0.22, -2.1; 0.35, -0.98; 0.37, -1.89; 0.5, 0];
T = [zeros(5,1); ones(5,1)];
W = LDA(X,T);
L = [ones(10,1) X] * W';
P = exp(L) ./ repmat(sum(exp(L),2),[1 2]); % Sandsynlighed for hvert punkt er i respektive klasser
round(P,2)

%%
% Luder staver bagl??ns 1.1 Read an image into the Matlab workspace and to get information about
%the dimensions of the image.
so = imread('data/flower.png') ; 
%%
% Lx 1.2. Display an image.
disp(so)
%% 
% Lx 1.3. Display an image histogram.
so = imread('data/flower.png'); 
imhist(so) ; 
%%
%Lx 1.4. Inspect pixel values in an image using both (x, y) and (row, column) pixel
%coordinates.
img = imread('data/flower.png'); 
img(1,2)
%%
%Lx 1.5. Use the Matlab Image Tool (imtool) to visualize and inspect images
img = imread('data/flower.png'); 
imshow(img)
%%
%Lx 1.7. Resize an image.
img = imread('data/flower.png'); 
img_1 = imresize(img, 0.5) ; 
%imshow(img)
%%
% Lx 1.9. Transform a RGB image into a grey-level image (rgb2gray)
img = imread('data/flower.png'); 
img = rgb2gray(img) ;
%%
% Read DICOM files into the Matlab workspace and to get information
% about the image from the DICOM header
img = dicomread('data/1-179.dcm') ; 
info = dicominfo('1-179.dcm') ; 
disp(info) ;
%%
%Lx 1.11. Use the Matlab Image Tool (imtool) to adjust contrast and brightness in
%a DICOM image
imtool('data/1-179.dcm')
%%
% Lx 1.13. Visualise individual color channels in an RGB image.
img = imread('data/flower.png'); 
imshow(img(:,:,3)) % THIS IS B CHANNEL
%%
% Lx 1.14. Change and manipulate individual color channels in an RGB image
img = imread('data/flower.png'); 
img = img(:,:,3) < 10 ; % All B values in RGB are less than 115
%imshow(img)
%%
% Lx 1.15. Use the image contour tool (imcontour) to visualize greylevel contours in
% an image
img = imread('data/flower.png'); 
imcontour(img(:,:,3))
%% 
% Lx 1.16. Use the image profile tool (improfile) to sample and visualise grey scale
% profiles.
img = imread('data/flower.png'); 
imshow(img(:,:,1)) ;
improfile()
%%
% Lx 1.17. Use the mesh tool to visualize a 2D image as a height map
img = double(imread('data/flower.png')) ;
mesh(img(:,:,1)) ;
%%
% Lx 1b.1. Load data from a text file into the Matlab workspace
M = load('data/irisdata.txt');
%%
% Lx 1b.2. Create a data matrix from a text file
clear
M = load('data/irisdata.txt');
M = M(:,1:4); M2  = M(:,1:4)' ; % M2 would be the correct one for data analysis i.e. PCA
%%
% Lx 1b.7. Compute the covariance matrix from multiple sets of measurements
X = load('data/irisdata.txt');
X = X(:,1:4)'; 
[M,N] = size(X);
data = X ; 
mn = mean(data,2);
data = data - repmat(mn,1,N);
% calculate the covariance matrix
Cx = 1 / (N-1) * data * data';
%%
% Lx 1b.8. Compute the principal components using Eigenvector analysis Matlab function eig). 
X = load('data/irisdata.txt');
X = X(:,1:4)'; 
[M,N] = size(X);
data = X ; 
mn = mean(data,2);
data = data - repmat(mn,1,N);
Cx = 1 / (N-1) * data * data';

% find the eigenvectors and eigenvalues
[PC, V] = eig(Cx);
% extract diagonal of matrix as vector
V = diag(V);

% sort the variances in decreasing order
[junk, rindices] = sort(-1*V);
V = V(rindices);
PC = PC(:,rindices); % The principal components :)

%%
% Lx 1b.9. Visualize how much of the total of variation each principal component explain.
X = load('data/irisdata.txt');
X = X(:,1:4)'; 
[M,N] = size(X);
data = X ; 
mn = mean(data,2);
data = data - repmat(mn,1,N);
Cx = 1 / (N-1) * data * data';
[PC, V] = eig(Cx);
V = diag(V);
[junk, rindices] = sort(-1*V);
V = V(rindices);
PC = PC(:,rindices);
figure
subplot(2,1,1);
Vnorm = V / sum(V) * 100
plot(cumsum(Vnorm), '*-')
title('CumSum Nigga')
ylabel('Percent explained variance')
xlabel('Principal component')
subplot(2,1,2)
plot(Vnorm, '*-')
title('Non-cum whitz')
ylabel('Percent explained variance')
xlabel('Principal component')
%%
% Lx 1b.10. Project original measurements into principal component space
X = load('data/irisdata.txt');
X = X(:,1:4)'; 
[M,N] = size(X);
data = X ; 
mn = mean(data,2);
data = data - repmat(mn,1,N);
Cx = 1 / (N-1) * data * data';
[PC, V] = eig(Cx);
V = diag(V);
[junk, rindices] = sort(-1*V);
V = V(rindices);
PC = PC(:,rindices);
% Project original measurements into principal component space
data_in_PC_space = PC' * data ;

%%
% Lx 1b.11. Use the Matlab function plotmatrix to visualise the covariance
% structure after projecting to principal component space.
X = load('data/irisdata.txt');
X = X(:,1:4)'; 
[M,N] = size(X);
data = X ; 
mn = mean(data,2);
data = data - repmat(mn,1,N);
Cx = 1 / (N-1) * data * data';
[PC, V] = eig(Cx);
V = diag(V);
[junk, rindices] = sort(-1*V);
V = V(rindices);
PC = PC(:,rindices);
data_in_PC_space = PC' * data ;
%plotmatrix to visualise the covariance structure after projecting to principal component space.
[~,ax]=plotmatrix(data_in_PC_space');

%%
% Lx 3.1. Implement and test a function that can do linear histogram stretching of
% a grey level image

% function Io = HistStretch(I)
% % Function that stretches the histogram of an image
% % Input:
% %   I = Image you want to be stretched
% % Output:
% %   Io = The histogram-stretched image
% 
% Itemp = double(I);  %First we have to convert to a format that can contain decimals
% 
% vmind = 0;  % General minimum pixel value defines (256 bit image)
% vmaxd = 255;    % General maximum pixel value defined (256bit image)
% vmin = min(min(Itemp));   % Min pixel value for the image defined
% vmax = max(max(Itemp));   % Max pixel value for the image defined
% 
% Im = (vmaxd-vmind)/(vmax - vmin) *(I-vmin)+vmind; %Formula for hist-stretch
% Io = uint8(Im);

%%
clear ; 
% Lx 3.2. Convert image pixels between doubles and 8-bit unsigned integers (UINT8)
I = imread('data/flower.png'); 
% Convert image to double
I_tmp1 = double(I);
% Convert to uint8
I_tmp2 = uint8(I);

%% 
% Lx 3.3. Implement and test a function that can perform gamma mapping of a grey
% level image
% 
% function gIm = GammaMap(Im,gamma)
% % Function that takes an image an creates its gamma map
% % Input:
% %   Im = image you want to gamme map
% %   gamma = the gamme value
% % Output:
% %   gIm = a graph of the gamma map
% 
% Itemp = double(Im); %First we convert the image to double
% vmax = 255; % Max pixel value
% 
% NyIm = (Itemp/vmax);    % Convertes down to pixelvalues between [0,1]
% 
% NyIm2 = NyIm.^gamma;   %The graph is computed
% 
% ko = 255*(NyIm2);   %We convert the gamma map back
% 
% gIm = uint8(ko); % Now it has to be converted back to integer data type
% imshow(gIm);

%% I should but do like that gamma u know?
% Lx 3.4. Use the Matlab function imadjust to modify grey scale images.
clear ;
i = imread('cat1.png') ;
figure
tmp = imadjust(i, [0.5, 1], [0, 1]) ;
subplot(2,1,1)
imshow(i)
subplot(2,1,2)
imshow(tmp);
clear;

%% 
% Lx 3.5. Implement and test a function that can threshold a grey scale image

% function Thres = ImageThreshold(Im,T)
% % Function creates a treshold image
% % Input:
% %   Im = image you want to treshold
% %   T = The treshold value
% % Output:
% %   Tres = treshold image
% 
% [m,n] = size(Im);   %First we define the size of the image
% 
% % Now we loop over number of rows and columns and change each pixel to
% % either 1 or 0 in correlation to the T-value:
% for i = 1:m 
%     for j = 1:n
%         if Im(i,j) <= T
%             Im(i,j) = 0;
%         else 
%             Im(i,j) = 1;
%         end
%     end
% end
% Thres = Im*255;  % Important to multiply back up so the range is [0,255]
clear ;
i = imread('cat1.png') ;
figure
tmp = i > 70 ;
subplot(2,1,1)
imshow(i)
subplot(2,1,2)
imshow(tmp);
clear;

%%
% Lx 3.6. Perform RGB thresholding in a color image
clear ; 
I = imread('flower.png') ; 
I_thresholded = I(:,:,1) < 100 & I(:,:,2) > 85 & I(:,:,2) < 200 & I(:,:,3) > 150;
% Here R < 100 and 85 < G < 200 and B > 150
imshow(I_thresholded)           

%%
% Lx 3.7. Convert a RGB image to HSV using the Matlab function rgb2hsv
clear 
I = imread('flower.png') ; 
I_hsv = rgb2hsv(I) ;

%%
% Lx 3.8. Visualise individual H, S, V components of a color image.
clear 
I = imread('flower.png') ;
[H,S,V] = rgb2hsv(I) ;
% Now good sir, you will be able to see all dis values okay?
% Hue, Saturation, Values?!

%%
% Lx 3.9. Implement and test thresholding in HSV space.
I = imread('flower.png') ;
I_hsv = rgb2hsv(I) ;
I_thresholded = I_hsv(:,:,1) < 100 & I_hsv(:,:,2) > 85 & I_hsv(:,:,2) < 200 & I_hsv(:,:,3) > 150;

%%
% Lx 3b.1 Create an empty data matrix that can hold N images and M measurement
% per image
clear
img = imread('flower.png') ;
H = size(img, 1);
W = size(img, 2);
M = H * W; % for a 512 x 512 image this would be 512 * 512 xD
data = zeros(M, N); % The empty data matrix

%%
% Lx 3b.2. Use the Matlab function reshape to transform an image into a column
% matrix aka flatten image.
clear
img = imread('flower.png') ;
flatten_img = reshape(img,[], 1) ;

%% 
% Lx 3b.3. Read a set of image files and put them into one data matrix
clear
tmp_img = imread('car1.jpg') ; 
M = length(reshape(tmp_img, [], 1)); % Flattens image and finds the length (Height x Width )
N = 5 ; % Number of images in directory
data = zeros(M, N); 
for i=1:N
      img_path = sprintf('car%d.jpg', i) ;
      img = imread(img_path);
      tt = reshape(img, [], 1);
      data(:, i)=tt;
end

%%
% Lx 3b.4. Compute the average column of a data matrix
clear
tmp_img = imread('car1.jpg') ; 
M = length(reshape(tmp_img, [], 1));
N = 5 ; 
data = zeros(M, N); 
for i=1:N
      img_path = sprintf('car%d.jpg', i) ;
      img = imread(img_path);
      tt = reshape(img, [], 1);
      data(:, i)=tt;
end
meanI = mean(data, 2); % THIS IS THE AVErAGE COLUMN

%%
% Lx 3b.5. Use the Matlab function reshape to transform an column from a matrix
% into an image.
clear
tmp_img = imread('car1.jpg') ; 
M = length(reshape(tmp_img, [], 1));
N = 5 ; 
data = zeros(M, N); 
for i=1:N
      img_path = sprintf('car%d.jpg', i) ;
      img = imread(img_path);
      tt = reshape(img, [], 1);
      data(:, i)=tt;
end
meanI = mean(data, 2); % THIS IS THE AVErAGE COLUMN
[H, W] = size(img) ; % Height and Width of image
I = reshape(meanI, H, W);
imshow(uint8(I)); % Alternatively you can use imshow(I, [])

%%
% Lx 3b.6. Compute the eigenvalues and eigenvector of a data matrix
clear
tmp_img = imread('car1.jpg') ; 
M = length(reshape(tmp_img, [], 1));
N = 5 ; 
data = zeros(M, N); 
for i=1:N
      img_path = sprintf('car%d.jpg', i) ;
      img = imread(img_path);
      tt = reshape(img, [], 1);
      data(:, i)=tt;
end
num_vecs = 5 ;
[eig_vectors,eig_values,psi] = pc_evectors(data,num_vecs); % contains the eigen vectors and eigen values
[nv,nd] = sortem(eig_vectors',diag(eig_values)); % Sorted eigen values and vecotrs
% BELOW IS CODE FOR PLOttTNING
% Vnorm = nd/sum(nd(:)) * 100 ; 
% Vnorm = diag(Vnorm) ;
% figure
% subplot(2,1,1)
% plot(cumsum(Vnorm), '*-')
% title('CumSum Nigga')
% ylabel('Percent explained variance')
% xlabel('Principal component')
% subplot(2,1,2)
% plot(Vnorm, '*-')
% title('Non-cum whitz')
% ylabel('Percent explained variance')
% xlabel('Principal component')

%%
% Lx 3b.7. Visualise eigenvectors as images
clear
tmp_img = imread('car1.jpg') ; 
M = length(reshape(tmp_img, [], 1));
N = 5 ; 
data = zeros(M, N); 
for i=1:N
      img_path = sprintf('car%d.jpg', i) ;
      img = imread(img_path);
      tt = reshape(img, [], 1);
      data(:, i)=tt;
end
num_vecs = 5 ;
[eig_vectors,eig_values,psi] = pc_evectors(data,num_vecs); % contains the eigen vectors and eigen values
[nv,nd] = sortem(eig_vectors',diag(eig_values)); % Sorted eigen values and vecotrs
% First eigenvector
eigvec1 = eig_vectors(:,1);
[H, W] = size(tmp_img) ; % Height and Width of image
v1img = reshape(eigvec1, H, W);
% Visualise eigenvectors as images
figure;
subplot(1,2,1)
imshow(v1img, []);
subplot(1,2,2)
imshow(-v1img, []);

%%
% Lx 3b.8. Synthesise facial image by a linear combination of the average face and a
% set of eigenvectors ||| Creating the mean image from eigen vectors
clear
tmp_img = imread('car1.jpg') ; 
M = length(reshape(tmp_img, [], 1));
N = 5 ; 
data = zeros(M, N); 
for i=1:N
      img_path = sprintf('car%d.jpg', i) ;
      img = imread(img_path);
      tt = reshape(img, [], 1);
      data(:, i)=tt;
end
num_vecs = 5 ;
[eig_vectors,eig_values,psi] = pc_evectors(data,num_vecs); 
[nv,nd] = sortem(eig_vectors',diag(eig_values));
meanI = mean(data, 2); 
% Chose scaling facots i.e. how much of each eigenvector component in the
% linear combination you need / want or feel for bitch ass 
s1 = 10*492 ; s2 = 20 ; s3 = 30;
disp('Nu k??rer jeg so') ; 
synthesized_img = s1 * eig_vectors(:, 1) - s2 * eig_vectors(:, 2) + s3 * eig_vectors(:, 3) + meanI;
disp('Okay, det var da ez') ; 
[H, W] = size(tmp_img) ; % Height and Width of image
synthesized_img = reshape(synthesized_img, H, W);
imshow(synthesized_img, [])

%%
% Lx 4.1. Use the Matlab imfilter function to filter an image using a given filter kernel.
% FILTER IS KERNEL, KERNEL IS FILTER, DEFINE KERNEL, DEFINE FILTER
clear 
im = imread('flower.png') ; 
kernel_size = 15; % SIZE OF KERNEL
se = ones(kernel_size) / kernel_size .^ 2; % AVerage filter, HERE YOU USE ONES SINCE THIS FINDS MEAN of 15x15 img patch
out = imfilter(im,se);
out2 = imfilter(im, fspecial('average', 15));
sum(out - out2, 'all')
%imshow(out)  % DOESNT EVEN NEED THAT DOUBLE IM BOI?

%%
% Lx 4.2. Filter an image using imfilter using zero-padding and border replication.
clear 
im = imread('flower.png') ; 
kernel_size = 15; % SIZE OF KERNEL
se = ones(kernel_size) / kernel_size .^ 2;
out = imfilter(im,se); % Filter with zero padding - it is default argument for imfilter
out = imfilter(im,se, 'replicate'); % Filter with replication

%%
% Lx 4.3. Remove salt and pepper noise using a median filter.
im = imread('flower.png') ;
kernel_size = [5,5] ;
out = medfilt2(im(:,:,1),kernel_size); % Applies median filter with specified kernel_size here 5x5
imshow(out) % SE DET PEPPER OG SALT ER V??K

%%
% Lx 4.5. Filter an image using the Matlab fspecial function
% Can also be used for computing and visualizing edges in an image using Sobel and Prewitt filters
close all
kernel = fspecial('disk',3);
% Below are types of kernels
% h = fspecial(type)
% h = fspecial('average',hsize)
% h = fspecial('disk',radius)
% h = fspecial('gaussian',hsize,sigma)
% h = fspecial('laplacian',alpha)
% h = fspecial('log',hsize,sigma)
% h = fspecial('motion',len,theta)
% h = fspecial('prewitt')
% h = fspecial('sobel')
im = imread('flower.png') ;
disked = imfilter(im(:,:,1), kernel, 'replicate');
imshow(disked)

%%
% Lx 4.7. Filter an image using the Matlab edge function
clear ; close all ; clc
I = imread('flower.png'); I = rgb2gray(I);
BW2 = edge(I,'Prewitt');
imshow(BW2);

%%
% Lx 4b.1. Use the Matlab strel function to create structuring elements.
se = strel('disk',5);

%%
% Lx 4b.2. Compute an eroded binary image using the Matlab imerode function.
im = imread('flower.png') ; im = rgb2gray(im) ; im = im < 100;
se = strel('disk', 5) ;
out = imerode(im,se) ;
imshow(out);

%%
% Lx 4b.3. Compute a dilated binary image using the Matlab imdilate function
im = imread('flower.png') ; im = rgb2gray(im) ; im = im < 100;
se = strel('disk', 5) ;
out = imdilate(im,se) ;
imshow(out);

%%
% Lx 4b.5. Compute an opened binary image using the Matlab imopen function.
im = imread('flower.png') ; im = rgb2gray(im) ; im = im < 100;
se = strel('disk', 5) ;
out = imopen(im,se) ;
imshow(out);

%%
% Lx 4b.6. Compute a closed binary image using the Matlab imclose function.
im = imread('flower.png') ; im = rgb2gray(im) ; im = im < 100;
se = strel('disk', 5) ;
out = imclose(im,se) ;
imshow(out);

%%
% Lx 5.1. Use the Matlab function bwlabel to create labels from a binary image
% using both 4- and 8-connectivity.
clc ; close all ; clear ; 
im = imread('data/floorboards.png') ; im = im < 100 ;
blob_4_connec = bwlabel(im,4);
blob_8_connec = bwlabel(im,8);

%% 
% Lx 5.2. Visualize labels using the Matlab function label2rgb
clc ; close all ; clear ; 
im = imread('data/floorboards.png') ; im = im < 100 ;
blob_4_connec = bwlabel(im,4);
blob_label = label2rgb(blob_4_connec) ; 

%%
% Lx 5.3. Compute BLOB features using the Matlab function regionprops including BLOB area and perimeter

clc ; close all ; clear ; 
im = imread('data/floorboards.png') ; im = im < 100 ;
blobs = bwlabel(im,4) ;
blob_features = regionprops(blobs, 'All') ; 
disp(blob_features) ; 

%%
% Lx 5.4. Select BLOBs that have certain features using the Matlab function ismember
clc ; close all ; clear ; 
im = imread('data/floorboards.png') ; im = im < 100 ;
blobs = bwlabel(im,4) ;
blob_features = regionprops(blobs, 'All') ;
idx = find([blob_features.Area] > 100 & [blob_features.Perimeter] > 500 );
im_bw = ismember(blobs,idx); % Blobs that have certain features

%%
% Lx 5.6. Crop regions from an image using the Matlab function imcrop
clc ; close all ; clear ; 
im = imread('data/flower.png') ; %im = im < 100 ;
cropped_im = imcrop(im, [75 68 130 112]) ;
imshow(cropped_im)
    
%%
% Lx 5.7. Select an appropriate threshold by inspecting the histogram of an image
close all ; clear ; 
im = imread('data/flower.png') ;
imhist(im) ;


%%
% Lx 5.8. Remove BLOBs at the image border using the Matlab function imclearborder.
close all ; clear ; 
im = imread('data/floorboards.png') ; im = im < 100 ;
blobs_cleared_of_border = imclearborder(im, 8) ; % Here 8-connectivity is used

%%
clear
% Lx 7.1. Implement and apply simple point transformations (including rotation and
% scaling) to 2D points.
im = imread('cat2.png') ; 
scale = 1 ;
sx = 0 ;
sy = 0 ;
rotten = 3 ;
tform = randomAffine2d('Scale',[scale,scale],'XShear',[sx sx],'YShear', [sy,sy],'Rotation',[rotten,rotten]);
transformed_img = imwarp(im,tform);

%%
% Lx 8.3. Compute the DICE score between two binary images.
% dice(img1, img2)

%%
% Lx 7.4. Compute the sum of squared difference objective function for two sets of
% corresponding landmarks
clear
load('data/catfixedPoints.mat')
load('data/catmovingPoints.mat')
sum((fixedpoints-movingpoints).^2, 'all')

%%
% Lx 7.5. Compute the center of mass for a set of landmarks.
% For BLOBS you can use regionprops use 'Centroid' else:
clear
load('data/catfixedPoints.mat')
load('data/catmovingPoints.mat')
fixed_COM = 1/length(fixedpoints(:,1)) * sum(fixedpoints);
moving_COM = 1/length(movingpoints(:,1)) * sum(movingpoints);

%%
% Lx 7.6. Align two sets of landmarks by aligning their center of masses.
% Align two sets of landmarks by aligning their center of masses.

% Create two translated landmark sets by subtracting the computed
% centre-of-masses:
clear
load('data/catfixedPoints.mat')
load('data/catmovingPoints.mat')
fixed_COM = 1/length(fixedpoints(:,1)) * sum(fixedpoints);
moving_COM = 1/length(movingpoints(:,1)) * sum(movingpoints);
fixed_trans = [fixedpoints(:,1) - fixed_COM(1) ...
fixedpoints(:,2) - fixed_COM(2)];
moving_trans = [movingpoints(:,1) - moving_COM(1) ...
movingpoints(:,2) - moving_COM(2)];

%%
% Lx 7.7. Use the Matlab function fitgeotrans to compute the transformation that transforms one set of landmarks into another set of landmarks.
clear
load('data/catfixedPoints.mat')
load('data/catmovingPoints.mat')
transform = fitgeotrans(movingpoints, fixedpoints,'NonreflectiveSimilarity'); 

%%
% Lx 7.8. Use the Matlab function transformPointsForward to apply a transformation to a set of points.
clear
load('data/catfixedPoints.mat')
load('data/catmovingPoints.mat')
transform_matrix = fitgeotrans(movingpoints, fixedpoints,'NonreflectiveSimilarity'); % OUTPUT IS A TRANSFORMATION MATRIX USED TO TRANSFORM IMAGE E.G RANSLATION, ROTATION, SCALING
forward = transformPointsForward(transform_matrix, movingpoints);

%%
% Lx 7.9. Use the Matlab function imwarp to apply a transformation to an image.
clear
load('data/catfixedPoints.mat')
load('data/catmovingPoints.mat')
transform_matrix = fitgeotrans(movingpoints, fixedpoints,'NonreflectiveSimilarity'); % OUTPUT IS A TRANSFORMATION MATRIX USED TO TRANSFORM IMAGE E.G RANSLATION, ROTATION, SCALING
cat_moved = imwarp(cat, transform_matrix);

