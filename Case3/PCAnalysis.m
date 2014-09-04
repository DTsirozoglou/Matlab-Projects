function [ L, signals,U] = PCAnalysis( train )

% PCAnalysis - MATLAB function to Perform a principal component analysis of the training data (train NxM).
% We perform PCA using covariance and the function returns a matrix with covariance's eigenvalues
% and a MxN matrix (signals) of the projected data.
% by Dimitris Tsirozoglou
%
% Use:
% [ L, signals,U ] = PCAnalysis( train )
%
% train       = The Training Data with their last column as the target value.
% L           = A matrix with covariance's eigenvalues.
% signals     = A MxN matrix of the projected data.
% U           = Columns of matrix U are the eigenvectors of covariance, sorted in conjuction with the eigenvalues.

TrainPCA    = train(:,1:(end-1)); 
[N,M]       = size(TrainPCA); 
% subtract off the mean for each dimension  
Mean        = sum(TrainPCA,1) / N;
TrainPCA    = TrainPCA - repmat(Mean,N,1); 
% calculate the covariance matrix 
covariance  = ((TrainPCA') * (TrainPCA)) / N ; 
% find the eigenvectors and eigenvalues 
% [U, L] = eig(covariance) produces matrices of eigenvalues (L) and eigenvectors (U) of matrix covariance,so that covariance*U = covariance*L.
% Matrix L is the canonical form of covariance — a diagonal matrix with covariance's eigenvalues on the main diagonal. 
% Matrix U is the modal matrix — its columns are the eigenvectors of covariance.
[U, L] = eig(covariance); 
% extract diagonal of matrix as vector 
L = diag(L); 
% sort the variances in decreasing order 
[~, order] = sort(-1*L); 
L = L(order); 
U = U(:,order); 
% Project the original data set with their target value at the last column 
signals = [train(:,(1:end-1)) * U' ,train(:,end)];

end
