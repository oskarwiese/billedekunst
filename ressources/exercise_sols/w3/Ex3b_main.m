%% Read yale data
clear; clc; close all;
P = 'YaleSubset\';
D = dir(fullfile(P, '*.png'));

N = numel(D);
% load first image to get size information
img = imread(fullfile(D(1).folder, D(1).name));

%% Exercise 1
H = size(img, 1);
W = size(img, 2);
M = H * W;

data = zeros(M, N);

%% Exercise 2
for k=1:N     
      img = imread(fullfile(D(k).folder, D(k).name));
      tt = reshape(img, [], 1);
      data(:, k)=tt;
end
%% Exercise 3

% Average image
meanI = mean(data, 2);
I = reshape(meanI, H, W);
imshow(I,[]);

%% Exercise 4
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
realFace = imread('FaceCroppedGray.png');
imshow(realFace, [])
RealFaceMat = double(reshape(realFace, [], 1));
RealFaceMat = RealFaceMat - meanI;
Proj = Vecs(:, 1:2)' * RealFaceMat; 

%% Exercise 10
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
