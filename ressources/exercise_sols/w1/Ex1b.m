%% Exercise 1b - PCA
%% Ex1
% PCA Analysis exercise with iris data
clear; close all; clc;
load irisdata.txt;

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

% Compute variance of each feature:
var_sep_L = var(sep_L)
var_sep_W = var(sep_W)
var_pet_L = var(pet_L)
vvar_pet_W = var(pet_W)

%% Ex3

% compute covariance matrix between sepal length and sepal width:
cov_sepL_sepW = cov(sep_L,sep_W)
% compute covariance matrix between sepal length and petal length:
cov_sepL_petL = cov(sep_L,pet_L)

%% Ex4 

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

% project the original data set
signals = PC' * data;

%% Ex6
% plot explained variance of principal components:
plot(V)
Vnorm = V / sum(V) * 100
plot(Vnorm, '*-')
ylabel('Percent explained variance')
xlabel('Principal component')
% It is clear, that the first component explains most of the variance, and
% the first 2-3 components will explain more that 90% of the variance

%% Ex7
[~,ax]=plotmatrix(signals');

% compute covariance of the two first components:
pv1 = signals(1,:);
pv2 = signals(2,:);
cov_pv1_2 = cov(pv1,pv2);
% They are not correlated at all, as the definition of the principal
% components is that they are uncorrelated with each other.