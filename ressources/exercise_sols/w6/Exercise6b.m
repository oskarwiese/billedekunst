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