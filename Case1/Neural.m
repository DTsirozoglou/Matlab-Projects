function [ MSerrorTrain,TrainHNeu,TrainLR, MSerrorTest, sunspotsN] = Neural( train, test, NofHiddenNeurons, learningRates )

% Neural - MATLAB function to Apply gradient-based (batch) training to a neural network model. 
% For the exercise, we consider standard steepest descent. We train neural networks with [NofHiddenNeurons] hidden neurons
% and with [learningRates] different learning rates using all the data in the train data set. We use batch learning (Iterative = 1:10000)
% until the error on the training data stops decreasing significantly (differenceFlag=0.1*10^-7). The function finds the learning rate
% and the number of the neurons of the hidden layer that gives the best mean-squared-error(MSerrorTrain) results for the train data set,
% and it use the weights of the trained neural network with [NofHiddenNeurons(y)] hidden neurons and 
% with [learningRates(flag)] learning rate, to find our mean-squared error on test set (MSerrorTest).
% by Dimitris Tsirozoglou
%
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
% MSerrorTest            = The mean-squared-error over the complete Test data set.
% TrainHNeu              = The number of the neurons of the hidden layer that gives the best mean-squared-error results for the train set
% TrainLR                = The learning rate that gives the best mean-squared-error results for the train set
% sunspotsR              = Rescaled prediction of the target values of the Test set(average Sunspots in a year).

% Note1: The hidden units have logistic sigmoid activation functions given by h(a)= tanh(a) 
%       (That's why we assign random values from -0.01 to 0.01 to our weights).
% Note2: We use a logistic activation function (h(a) =1/(1 + exp(?a)))in the output layer in order restrict the output to 
%        the range 0-1 (it must be positive) as we require.
% Note3: All neurons have bias (offset) parameters.

    % Perform a scaling to put the train data in range [0-1]
    Trainmin    = min(train(:));
    Trainmax    = max(train(:));
    TrainScaled = (train-Trainmin)./(Trainmax-Trainmin);    % This is the scaled train set

    % Perform a scaling to put the test data in range [0-1]
    Testmin    = min(test(:));
    Testmax    = max(test(:));
    TestScaled = (test-Testmin)./(Testmax-Testmin);         % This is the scaled test set
 
    xTe = TestScaled(:,1:(end-1));                          % A matrix with the observations of the scaled Test set
    tTe = TestScaled(:,end);                                % our desired output of the scaled Test set
    Dt  = size(test,1);                                     % The number of all the inputs of test data
    Kt  = size(tTe,1);

    D   = size(train,1);                                    % The number of all the inputs of train data
    X   = TrainScaled(:,1:(end-1));                         % A matrix with the observations of the scaled Training set
    T   = TrainScaled(:,end);                               % our desired output of the scaled Training set
    K   = size(T,1);
    
    neurons         = NofHiddenNeurons;                     % declare how many neurons there are on the hidden layer
    h               = learningRates;                        % learning rates
    differenceFlag  = 0.1*10^-7;    
    flag            = 1;
    bestErrorTrain  = 10^10;
    
    while flag < size(h,2)+1                               % for all the learning rates
        
        for  y  = 1:size(neurons,2)                        % for all the differrent numbers of hidden neurons
       
            M = neurons(y);
            W1  = unifrnd(-0.01,0.01,M,(size(X,2)+1));     % assign random values to the weights W1 of the first layer
            W2  = unifrnd(-0.01,0.01,1,M+1);               % assign random values to the weights W2 of the second layer
            [rows1, cols1] = size(W1);
            [rows2, cols2] = size(W2);
            wr = (rows1 * cols1) + (rows2 * cols2);

            for Iterative = 1:10000

                gradient= zeros(wr,D);
                a2      = zeros(1,K);
                y2      = zeros(1,K);
                delta2  = zeros(K,1);   

            % we run our network for all the inputs of train data
                for i = 1:D

                    delta1  = zeros(1,M+1);
                    a       = zeros(M+1,1);
                    z       = zeros(1,M+1);
                    z(1)    = 1;  

            % we find the a for every hidden neuron with the weight vector W1
                    for j = 2:M+1
                        for b = 1:size(X,2)            
                            a(j) = W1(j-1,b)*X(i,b) + a(j); 
                        end
                        a(j)   = a(j) + W1(j-1,(size(X,2)+1));
                        % We use tanh as activation function
                        z(j)   = (exp(a(j)) - exp(-a(j))) / (exp(a(j)) + exp(-a(j))); %a(j) / (1 + abs(a(j))); %1 / (1 + exp(-a(j))); 
                    end

            % we find the a2 for every hidden neuron with the weight vector W2 and then
            % we use it to find the error signal y2(i)
                    for j      = 1:M+1
                        a2(i)   = a2(i) + W2(1,j)*z(j);
                    end
                    % We use a logistic activation function (h(a) =1/(1 + exp(?a)))in the output layer
                    y2(i) = 1 / (1 + exp(-a2(i)));
                    % we find the error delta2(i)
                    delta2(i)  = y2(i) - T(i);

            % We calculate the error signal delta1(j) using  W2
                    for j = 1:M+1
                        if j == 1
                            temp        = 1;
                        else
                            % We use the derivative of tanh function
                            temp        = 1 - ((exp(a(j)) - exp(-a(j)))^2 / (exp(a(j)) + exp(-a(j)))^2) ;%exp(a(j)) / ((1 + exp(a(j)))^2); %1 / (1 + abs(a(j))).^2;   
                        end
                            delta1(j)   = temp*(W2(1,j)*delta2(i));
                    end
            % calculate gradients for W2 weights
                    l=1;
                    for j = 1:M+1
                        gradient(l,i)   = delta2(i)*z(j);
                        l = l+1;
                    end

            % calculate gradients for W1 weights 
                    for r = 1:size(X,2)
                        for j = 2:M+1
                            gradient(l,i) = delta1(j)*X(i,r);
                            l = l+1;
                        end          
                    end

                    for j = 2:M+1
                        gradient(l,i) = delta1(j);
                        l = l+1;
                    end
                end

            % we calculate the mean-squared error as loss/error function E
            % and we create the matrix avggradient with the values of the avg of the gradients.
                avggradient     = mean(gradient,2);
                E(Iterative)    = mean(delta2.^2);

                % Check if the error on the training data stops decreasing significantly (differenceFlag=0.1*10^-60;),if yes then break
                if Iterative>1
                    if E(Iterative-1)-E(Iterative)<differenceFlag
                        break
                    end
                end

            % we calculate the new weights of W2    
                m=1;
                for j = 1:M+1
                    W2(1,j) = W2(1,j) - (h(flag)*avggradient(m));
                    m = m+1;
                end

            % we calculate the new weights of W1
                for r = 1:(size(X,2)+1)
                    for j = 1:M
                        W1(j,r) = W1(j,r) - (h(flag)*avggradient(m));
                        m = m+1;
                    end
                end
                
            end
            
            % We find the best error achieved on train set and we store the number of neurons at the hidden layer,the learning rate 
            % and the weights with whom we achieved the best error on train set
            if E(Iterative)<bestErrorTrain
                bestErrorTrain  = E(Iterative);
                MSerrorTrain    = mean((((y2*(Trainmax - Trainmin)) + repmat(Trainmin,1,K)) - train(:,end)').^2);
                TrainHNeu       = neurons(y);
                TrainLR         = h(flag);
                BestW1          = W1;
                BestW2          = W2;
            end
            
        end
        flag = flag + 1; % number of tested learning rates
    end
            
% We have the weights (BestW1,BestW2) of the trained neural network with (TrainHNeu) hidden neurons
% and with (TrainLR) learning rate and we use them to find our mean-squared error on test set.


    a2t       = zeros(1,Kt);
    delta2te  = zeros(Kt,1);
    M         = TrainHNeu;
    W1        = BestW1;
    W2        = BestW2;
    
    % we run our network for all the inputs of test set
    for i = 1:Dt

        at       = zeros(M+1,1);
        zt       = zeros(1,M+1);
        zt(1)    = 1;

    % we find the at for every hidden neuron with the weight vector W1
        for j = 2:M+1
            for b = 1:size(xTe,2)            
                at(j) = W1(j-1,b)*xTe(i,b) + at(j); 
            end
            at(j)   = at(j) + W1(j-1,(size(xTe,2)+1));
            % We use tanh as activation function
            zt(j)     = (exp(at(j)) - exp(-at(j))) / (exp(at(j)) + exp(-at(j)));%a(j) / (1 + abs(a(j)));
        end

  % we find the a2t for every hidden neuron with the weight vector W2 and then
  % we use it to find the error signal y2te(i)
        for j      = 1:M+1
            a2t(i)   = a2t(i) + W2(1,j)*zt(j);
        end
        % We use a logistic activation function (h(a) =1/(1 + exp(?a)))in the output layer
        y2te(i) = 1 / (1 + exp(-a2t(i)));
        % we find the rescaled Test error delta2te(i)
        delta2te(i)  = ((y2te(i)*(Testmax - Testmin)) + Testmin) - test(i,end);
    end
    
    % We find the rescaled prediction of the target values of the Test set(average Sunspots in a year) 
    % and the The mean-squared-error over the complete Test data set.
    sunspotsN   = (y2te*(Testmax - Testmin)) + repmat(Testmin,1,Dt);
    MSerrorTest = mean(delta2te.^2);

end

