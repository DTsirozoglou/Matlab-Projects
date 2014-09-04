function [ bestTrainknnErr,BestKtrain, TestknnErr ] = KNN( train , test, Neighbours )

% KNN - MATLAB function to apply a non-linear, non-parametric method to the train data, namely nearest
% neighbor classification. We Implement a k(Neighbours)-nearest neighbor classifier
% (k-NN) with Euclidean metric and we report its  best accuracy on the training
% set depending on the number of the Neighbours and then we use the test data to evaluate it.
% by Dimitris Tsirozoglou
% Use:
% [ bestTrainknnErr,BestKtrain, TestknnErr ] = KNN( train , test, Neighbours );
%
% train             = The Training Data with their last column as the target value.
% test              = The Test Data with their last column as the target value. 
% Neighbours        = Different number of Neighbours that our model will be trained with. 
% bestTrainknnErr   = The best classification error achieved on the training set.
% TestknnErr        = The classification error on the test set.
% BestKtrain        = The number of the Neighbours used to achieve The best classification error on the training set.

    % Perform a scaling to put the train data in range [0-1]
    Trainmin    = min(train(:,1:(end-1)));
    Trainmax    = max(train(:,1:(end-1)));
    TrainScaled = (train(:,1:(end-1))-repmat(Trainmin,size(train,1),1))./repmat((Trainmax-Trainmin),size(train,1),1); % This is the scaled train set
 
    % Perform a scaling to put the test data in range [0-1]
    Testmin    = min(test(:,1:(end-1)));
    Testmax    = max(test(:,1:(end-1)));
    TestScaled = (test(:,1:(end-1))-repmat(Testmin,size(test,1),1))./repmat((Testmax-Testmin),size(test,1),1);        % This is the scaled test set


    X                   = TrainScaled;                               % A matrix with the observations of the scaled Train set.
    t                   = train(:,end);                              % A matrix with the target values of the Train set.
    Xtest               = [TestScaled,test(:,end)];                  % A matrix with the observations of the scaled Test set and the target values of the Test set in the last column.
    testlabel           = zeros(size(Xtest,1),1);
    trainlabel          = zeros(size(X,1),1);
    numoftestdata       = size(Xtest,1);
    numoftrainingdata   = size(X,1);
    k                   = Neighbours;
    A                   = [0 0 0];
    bestTrainknnErr     = 100;

    %training data estimation
    for z=1:size(k,2) %for each neighbor
        for i=1:numoftrainingdata

            % we calculate the euclidian distance of each point from training 
            % set from each data point of the training set
            y                   = repmat(X(i,:),numoftrainingdata,1);
            euclideandistance   = sum((y-X).^2,2);
            [ ~ , position]     = sort(euclideandistance,'ascend');
            %we sort the distances so as to get the closests neighbor

            %gets the label from the neighbors and adding points to a matrix
            %for each
            for j=1:k(z)
                if t(position(j))==0 %thats how we get the label from the closest neighbor
                    A(1)=A(1)+1;
                elseif t(position(j))==1
                    A(2)=A(2)+1;
                else A(3)=A(3)+1;
                
                end
            end
            %we compare the different values on the created matrix and asign the label 
            %that has the bigger value to the new data point

            if A(1)>A(2) && A(1)>A(3)
                trainlabel(i)=0;
            elseif A(2)>A(3)
                 trainlabel(i)=1;
            elseif A(3)>A(2)
                trainlabel(i)=2;
            %we check the equality of the values so as to pick randomly the new label
            elseif A(1)==A(2) && A(3)<A(1)
                nList=[0 1];
                trainlabel(i)=nList(floor(rand*length(nList))+1);
            elseif A(2)==A(3) && A(1)<A(2)
                nList=[1 2];
                trainlabel(i)=nList(floor(rand*length(nList))+1);
            elseif A(1)==A(3) && A(2)<A(1)
                nList=[0 2];
                trainlabel(i)=nList(floor(rand*length(nList))+1);
            end
             A=[0 0 0];
        end
% we calculate the error of the training data set
        bad             = (trainlabel==t);
        TrainknnErr(z)  = (size(X,1) - sum(bad)) * (size(X,1)/100);
        bad             = zeros(size(trainlabel,1),1);

% we compare each error (of different Neighbours ) of the training data set
% and we store the best error and the number of the Neighbours we used to achieve it.
        if TrainknnErr(z)<bestTrainknnErr
            bestTrainknnErr = TrainknnErr(z);
            BestKtrain           = k(z);
        end
    end
    
% test data estimation
% we perform the same procedure for the test set using the number of
% Neighbours we used to achieve the best error on training set.
   
    k = BestKtrain;
    
    for i=1:numoftestdata

        y=repmat(Xtest(i,1:7),numoftrainingdata,1);
        euclideandistance = sum((y-X).^2,2);
        [ ~ , position] = sort(euclideandistance,'ascend');

        for j=1:k
            if t(position(j))==0
                A(1)=A(1)+1;
            elseif t(position(j))==1
                A(2)=A(2)+1;
            else A(3)=A(3)+1;
            end
        end

        if A(1)>A(2) && A(1)>A(3)
            testlabel(i)=0;
        elseif A(2)>A(3)
             testlabel(i)=1;
        elseif A(3)>A(2)
            testlabel(i)=2;
        elseif A(1)==A(2) && A(3)<A(1)
            nList=[0 1];
            testlabel(i)=nList(floor(rand*length(nList))+1);
        elseif A(2)==A(3) && A(1)<A(2)
            nList=[1 2];
            testlabel(i)=nList(floor(rand*length(nList))+1);
        elseif A(1)==A(3) && A(2)<A(1)
            nList=[0 2];
            testlabel(i)=nList(floor(rand*length(nList))+1);
        end
         A=[0 0 0];
    end

    bad             = (testlabel==Xtest(:,8));
    TestknnErr      = (size(Xtest,1) - sum(bad)) * (size(Xtest,1)/100);

end

