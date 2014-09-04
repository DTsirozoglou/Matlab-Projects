%% Case 1: Sunspot Prediction

% Question 1

clc;
clear all;
close all;

load sunspotsTestStatML.dt
load sunspotsTrainStatML.dt

train   = sunspotsTrainStatML;
test    = sunspotsTestStatML;

% Call the function linearRegr to build a linear model of the data using linear regression
% with the maximum likelihood approach to the training data.
% Use:
% [ Wl, sunspotsR, MSErrTrain, MSErrTest ] = linearRegr( train, test )
%
% train       = The Training Data with their last column as the target value.
% test        = The Test Data with their last column as the target value.   
% Wl          = The parameters of the model.
% sunspotsR   = The prediction of the target values of the Test set(average Sunspots in a year).
% MSErrTrain  = The mean-squared-error of the linear model over the complete Training data set.
% MSErrTest   = The mean-squared-error of the linear model over the complete Test data set.

[ Wl, sunspotsR, MSErrTrain, MSErrTest ] = linearRegr( train, test );
disp('QUESTION 1');
disp('The six parameters of the linear model are:');
Wl

disp(['The mean-squared-error of the linear model over the complete training data set is:' num2str(MSErrTrain) '']);

disp(['The mean-squared-error of the linear model on the test data set is:' num2str(MSErrTest) '']);


% Question 2

% Call the function Neural to apply gradient-based (batch) training to a neural network model. 
% For the exercise, we consider standard steepest descent. We train neural networks with [NofHiddenNeurons] hidden neurons
% and with [learningRates] different learning rates using all the data in the train data set. We use batch learning (Iterative = 1:10000)
% until the error on the training data stops decreasing significantly (differenceFlag=0.1*10^-7). The function finds the learning rate
% and the number of the neurons of the hidden layer that gives the best mean-squared-error(MSerrorTrain) results for the train data set,
% and it use the weights of the trained neural network with [NofHiddenNeurons(y)] hidden neurons
% and with [learningRates(flag)] learning rate, to find our mean-squared error on test set (MSerrorTest).
% Use:
% [ MSerrorTrain,TrainHNeu,TrainLR, MSerrorTest, sunspotsN] = Neural( train, test, NofHiddenNeurons, learningRates )
%
% train                  = The Training Data with their last column as the target value.
% test                   = The Test Data with their last column as the target value.   
% NofHiddenNeurons       = An array of the number of the hidden neurons of the hidden layer,
%                           for example [5,10] we will train the network first with 5 neurons
%                           at the hidden layer and then with 10.
% learningRates          = An array with the learning rates with witch we want to test our Neural Network.
% MSerrorTrain           = The best mean-squared-error over the complete Training data set.
% MSerrorTest            = The best mean-squared-error over the complete Test data set.
% TrainHNeu              = The number of the neurons of the hidden layer that gives the best mean-squared-error results for the train set
% TrainLR                = The learning rate that gives the best mean-squared-error results for the train set
% sunspotsR              = Rescaled (Best)prediction of the target values of the Test set(average Sunspots in a year).

% Note1: The hidden units have logistic sigmoid activation functions given by h(a)= tanh(a) 
%       (That's why we assign random values from -0.01 to 0.01 to our weights).
% Note2: We use a logistic activation function (h(a) =1/(1 + exp(-a)))in the output layer in order restrict the output to 
%        the range 0-1 (to be positive) as we require.
% Note3: All neurons have bias (offset) parameters.

[ MSerrorTrain,TrainHNeu,TrainLR, MSerrorTest, sunspotsN] = Neural( train, test, 15,0.7 );
disp('QUESTION 2');
disp(['The mean-squared-error of the non-linear model over the complete training data set is:' num2str(MSerrorTrain) ' using ' num2str(TrainHNeu) ' Hidden Neurons with the learning rate set to: ' num2str(TrainLR) ' ']);

disp(['The mean-squared-error of the non-linear model over the complete test data set is:' num2str(MSerrorTest) ' using ' num2str(TrainHNeu) ' Hidden Neurons with the learning rate set to: ' num2str(TrainLR) ' ']);

% Question 3

% We Plot the sunspots time series from year 1916 to 2011 (years on the x-axis, average number of sunspots on the y-axis). 
% Then we add the predictions of our linear and non-linear model to the plot to visualize the quality of our model.

figure ('name','Question 3')
hold on
plot(1916:2011,test(:,end),'r')
plot(1916:2011,sunspotsR,'gr')
plot(1916:2011,sunspotsN,'b')
hold off

title('Plot to visualize the quality of our models.');
xlabel('Sunspots time series from year 1916 to 2011');
ylabel('Average number of sunspots ');
legend('Visualization of the test data','The corresponding outputs of the linear model', 'The corresponding outputs of the non-linear model');

%% Case 2: Surveying the Sky

% Question 4

clear all;

load quasarsStarsStatMLTest.dt
load quasarsStarsStatMLTrain.dt

train = quasarsStarsStatMLTrain;
test = quasarsStarsStatMLTest;

% We will first normalize the data sets
for i = 1:(size(train,2)-1)
    
    feature                 = train(:,i);
    testfeature             = test(:,i);
    
% We Compute the mean and the variance of every input feature 
    mean_val(i)             = mean(feature(:));
    varian_val              = std(feature(:));

% We transform the training data such that the mean and the variance of every feature in the normalized data are 0 and 1, respectively    
    normalizedtrain(:,i)              = (feature-mean_val(i))/varian_val;    
    meanofnormalizedtrain(i)          = mean(normalizedtrain(:,i)); % We verify by computing these values are 0
    varianceofnormalizedtrain(i)      = var(normalizedtrain(:,i));  % We verify by computing these values are 1

% We use the function:((feature-mean_val)/varian_val) to also encode the test data.
    normalizedtest(:,i)               = (testfeature - mean_val(i))/varian_val;
end

ClassLabel  =unique(train(:,6));                            % we store the different classes in the matrix  "ClassLabel"
Gstars = (train(:,6) == ClassLabel(1));
Gquasars = (train(:,6) == ClassLabel(2));

stars = normalizedtrain(Gstars,:);
quasars = normalizedtrain(Gquasars,:);
% We consider all pairs consisting of a training input vector from the positive class(stars) 
% and a training input vector from the negative class(quasars).
% Compute the difference in input space between all pairs. 
l=1;
for i =1:size(stars,1)
    for j =1:size(quasars,1)
        G(l,:) = norm(stars(i,:)-quasars(j,:));
        l = l+1;
    end
end

% The median of these distances can be used as a measure of scale and therefore as a guess for Sjaakkola.
Sjaakkola = median(G);
% Compute the bandwidth parameter GJaakkola from SJaakkola using the identity given
Gjaakkola = 1 / (2*(Sjaakkola^2));

n = -3:3;
z = -1:3;
Bestacc = 0;
% We will use 5-fold cross validation to randomly generated indices for a 5-fold cross-validation of N  observations.
% Indices will contain equal (or approximately equal) proportions of the integers (j) 1 through K (5) that define a partition of 
% the N observations into K disjoint subsets.
k=5;
cvFolds  = crossvalind('Kfold', train(:,6), k);

for i=1:numel(z);       % z = {-1,...,3}
% Use grid-search to determine appropriate SVM hyperparameters c and g.
    c                   = 2^n(i); 
        
    for p=1:numel(n);       % n = {-3,...,3}
        g                   = Gjaakkola*(2^(-n(p)));
        accuracy = nan(k);
        for j = 1:k

% We will use K-1 folds for training and the last fold is used for validation.
% This process is repeated K times, leaving one different fold for evaluation each time.    
            foldgroup               = (cvFolds == j);                       % We find the validation group
            temp                    = train(:,6);

            valid                   = normalizedtrain(foldgroup,:);         % We take the validation normalizedtrain features
            validtarget             = temp(foldgroup,:);                    % We take the validation train targets

            trainingpartition       = (cvFolds ~= j);                       % K-1 folds

            modeltrain              = normalizedtrain(trainingpartition,:); % We take the training normalizedtrain features of K-1 folds
            modeltarget             = temp(trainingpartition,:);            % We take the training targets of K-1 folds

% We create the model,with the selected combination of the hyperparameters,usinng the "modeltrain" features        
            model               = svmtrain2(modeltarget, modeltrain,['-q -g ' num2str(g) ' -c ' num2str(c)]);

% We calculate the accuracy for the normalized validation training data        
            evalc('[~, acc, ~]     = svmpredict2(validtarget, valid, model )');

            accuracy(k)         = acc(1);

            if accuracy(k) > Bestacc
                Bestacc  = accuracy(k);  % We create a table with the best accuracies achieved for every trainingpartition
                BestC    = c;            % We create a table with the hyperparameters c of the best accuracy achieved for every trainingpartition
                Bestg    = g;            % We create a table with the hyperparameters g of the best accuracy achieved for every trainingpartition
            end
        end
    end
end

% We pick the hyperparameter pair with the lowest average 0-1 loss (with the highest accuracy)

c = BestC;                % this is the optimal of bestc
g = Bestg;                % ths is the optimal of bestg

% We create model by training it using the complete normalized training dataset.
% Only the training data are used in the model selection process.
model                   = svmtrain2(train(:,6), normalizedtrain(:,1:5),['-q -g ' num2str(g) ' -c ' num2str(c)]); 

% test on the normalized testset
evalc('[~, Testacc, ~]          = svmpredict2(test(:,6), normalizedtest(:,1:5), model, [])');
% test on the normalized trainset
evalc('[~, Trainacc, ~]          = svmpredict2(train(:,6), normalizedtrain(:,1:5), model, [])');

% show accuracy for both tests
disp('QUESTION 4');
disp(['Test Accuracy with optimized C (' num2str(c) ...          
      ') and optimal Gama (' num2str(g) ') on Testset: ' num2str(Testacc(1)) '%']);
  
disp(['Train Accuracy with optimized C (' num2str(c) ...          
      ') and optimal Gama (' num2str(g) ') on Trainset: ' num2str(Trainacc(1)) '%']);
  
disp(['The initial Gamma value suggested by Jaakkola heuristic is :' num2str(Gjaakkola) '']); 

%% Case 3: Wheat Seeds

% Question 6

clear all;

load seedsTest.dt
load seedsTrain.dt

train   = seedsTrain;
test    = seedsTest;

% Call the function PCAnalysis to Perform a principal component analysis of the training data (train NxM).
% We perform PCA using covariance and the function returns a matrix with covariance's eigenvalues
% and a MxN matrix (signals) of the projected data.
%
% Use:
% [ L, signals,U] = PCAnalysis( train )
%
% train       = The Training Data with their last column as the target value.
% L           = A matrix with covariance's eigenvalues.
% signals     = A matrix of the projected data.
% U           = Columns of matrix U are the eigenvectors of covariance, sorted in conjuction with the eigenvalues.

[ L, signals,U] = PCAnalysis( train );

Kama        = signals(signals(:,8)==0,:);
Rosa        = signals(signals(:,8)==1,:);
Canadian    = signals(signals(:,8)==2,:);

% Plot the eigenspectrum
figure ('name','Question 6.1')
hold on
plot(1:size(L,1),sort(L,'descend'))
hold off

title('A plot of the complete spectrum of eigenvalues, sorted into decreasing order');
ylabel('Eigenvalues sorted into decreasing order');

%plot of the data projected on the first two principal components with different colors indicating the 3 different classes
figure ('name','Question 6.2 and Question 7')
hold on
plot(Kama(:,1),Kama(:,2),'ro',...
     'MarkerSize',6)
plot(Rosa(:,1),Rosa(:,2),'gx',...
     'MarkerSize',6)
plot(Canadian(:,1),Canadian(:,2),'b*',...
     'MarkerSize',6)

title('Visualize the data projected on the first two principal components.');
xlabel('First principal component');
ylabel('Second principal component');
legend('Kama Class','Rosa Class', 'Canadian Class','location','NorthWest');

% Question 7

% Call the function KMeansCluster to Perform 3-means clustering of the training data and report 
% the three cluster centers and the elements they own.
%
% Use:
% [KamaCentr,RosaCentr,CanadianCentr,BKama,BRosa,BCanadian] = KMeansCluster( train )
%
% train         = The Training Data with their last column as the target value.
% KamaCentr     = The cluster center of the first class (Kama).
% RosaCentr     = The cluster center of the second class (Rosa).
% CanadianCentr = The cluster center of the third class (Canadian).
% BKama         = The elements of the first cluster.
% BRosa         = The elements of the second cluster.
% BCanadian     = The elements of the third cluster.

[KamaCentr,RosaCentr,CanadianCentr,BKama,BRosa,BCanadian] = KMeansCluster( train );
disp('QUESTION 7');
disp('The cluster centers are :');
KamaCentr
RosaCentr
CanadianCentr

% We store the cluster centers in a matrix and we project them to the first two principal components of the training data.
A = [KamaCentr;RosaCentr;CanadianCentr];
Centroids = A*U';

% We visualize the clusters by adding the cluster centers to the plot from the Question 6 (to the figure ('name','Question 6.2')).
plot(Centroids(1,1),Centroids(1,2),'ko',...
     'MarkerSize',14,'LineWidth',2)
plot(Centroids(2,1),Centroids(2,2),'kx',...
     'MarkerSize',14,'LineWidth',2)
plot(Centroids(3,1),Centroids(3,2),'k*',...
     'MarkerSize',14,'LineWidth',2)
hold off

% Question 8.1

% Call the function LDAnalysis to Apply linear discriminant analysis (LDA) to the training data set
% and report the accuracies of the classifier on the training set as well as on the test set.
% Use:
% [ ldaErrTrain, ldaErrTest ] = LDAnalysis(train , test);
%
% train        = The Training Data with their last column as the target value.
% test         = The Test Data with their last column as the target value.   
% ldaErrTrain  = The classification error on training set.
% ldaErrTest   = The classification error on test set.

[ ldaErrTrain, ldaErrTest ] = LDAnalysis(train , test);
disp('QUESTION 8');
disp(['Classification error on Training set using LDA is:' num2str(ldaErrTrain) '%']);

disp(['Classification error on Test set using LDA is:' num2str(ldaErrTest) '%']);

% Question 8.2

% Call the function KNN to apply a non-linear, non-parametric method to the train data, namely nearest
% neighbor classification. We Implement a k(Neighbours)-nearest neighbor classifier
% (k-NN) with Euclidean metric and we report its  best accuracy on the training
% set depending on the number of the Neighbours and then we use the test data to evaluate it.
% Use:
% [ bestTrainknnErr,BestKtrain, TestknnErr ] = KNN( train , test, Neighbours )
%
% train             = The Training Data with their last column as the target value.
% test              = The Test Data with their last column as the target value. 
% Neighbours        = Different number of Neighbours that our model will be trained with. 
% bestTrainknnErr   = The best classification error achieved on the training set.
% TestknnErr        = The classification error on the test set.
% BestKtrain        = The number of the Neighbours used to achieve The best classification error on the training set.

[ bestTrainknnErr,BestKtrain, TestknnErr ] = KNN( train , test, [21,35] );

disp(['Best classification error on Training set is:' num2str(bestTrainknnErr) '% using K(' num2str(BestKtrain) ')NN ']);

disp(['Classification error on Test set using K(' num2str(BestKtrain) ')NN is:' num2str(TestknnErr) '%']);

