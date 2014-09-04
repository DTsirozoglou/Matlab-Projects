function [ Wl, sunspotsR, MSErrTrain, MSErrTest ] = linearRegr( train, test )

% linearRegr - MATLAB function to build a linear model of the data using linear regression
% using the maximum likelihood approach to the training data.
% by Dimitris Tsirozoglou
%
% Use:
% [ Wl, sunspotsR, MSErrTrain, MSErrTest ] = linearRegr( train, test )
%
% train       = The Training Data with their last column as the target value.
% test        = The Test Data with their last column as the target value.   
% Wl          = The parameters of the model.
% sunspotsR   = The prediction of the target values of the Test set(average Sunspots in a year).
% MSErrTrain  = The mean-squared-error of the linear model over the complete Training data set.
% MSErrTest   = The mean-squared-error of the linear model over the complete Test data set.


    x = train(:,1:(end-1));                           % A matrix with the observations of the Training set.  
    T = train(:,end);                                 % A matrix with the Target values of the Training set. 
    F = [ones(size(x,1),1),x];                        % The design matrix of the observations of the Training set.

    % Find the maximum likelihood (ML) estimate using the training set (Compute
    % the pseudo inverse of the design matrix with the function pinv).
    Wl = pinv(F)*T;                                        


    xTe= test(:,1:(end-1));                           % A matrix with the observations of the Test set.
    tTe= test(:,end);                                 % A matrix with the Target values of the Test set.
    FTe=[ones(size(xTe,1),1),xTe];                    % The design matrix of the observations of the Test set.


    % Apply the model to the training set using linear regression with the ML parameter estimate 
    for l = 1:size(F,1)

        ytrain(l) = Wl'*F(l,:)';
        % Compute the square error between the "train" prediction and the Target values of the Training set. 
        Atrain(l) = (T(l,1)-ytrain(l))^2;
    end
    
    % Compute the mean-squared-error of the linear model over the complete Training data set.
    MSErrTrain  = mean(Atrain);

    % Apply the model to the test set using linear regression with the ML parameter estimate 
    for i = 1:size(FTe,1)

        yltest(i) = Wl'*FTe(i,:)';
        % Compute the square error between the "test" prediction and the Target values of the Test set.
        Atest(i) = (tTe(i,1)-yltest(i))^2;

    end
    
    % Compute the mean-squared-error of the linear model over the complete Test data set.
    MSErrTest =  mean(Atest);

    % The prediction of the target values of the Test set 
    sunspotsR = yltest;
end

