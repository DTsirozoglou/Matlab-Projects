function [ ldaErrTrain, ldaErrTest ] = LDAnalysis(train , test)

% LDAnalysis - MATLAB function to Apply linear discriminant analysis (LDA) to the training data set
% and report the accuracies of the classifier on the training set as well as on the test set.
% by Dimitris Tsirozoglou
%
% Use:
% [ ldaErrTrain, ldaErrTest ] = LDAnalysis(train , test)
%
% train        = The Training Data with their last column as the target value.
% test         = The Test Data with their last column as the target value.   
% ldaErrTrain  = The classification error on training set.
% ldaErrTest   = The classification error on test set.

    InputTrain  =train(:,1:(end-1));                        % InputTrain is an array with the values of the attributes of the Training set
    target      =train(:,end);                              % target is an array pointing the value of the class of the Training set
    InputTest   =test(:,1:(end-1));                         % InputTest is an array with the values of the attributes of the Test set
    targettest  =test(:,end);                               % targettest is an array pointing the value of the class of Test set
    ntest       =size(test,1);
    [n,m]       =size(InputTrain);                          % n is the number of the training data and m is the number of input variables
    ClassLabel  =unique(target);                            % we store the different classes in the matrix  "ClassLabel"
    k           =length(ClassLabel);                        % number of classes
    Sum         =zeros;
    GroupMean   =NaN(k,m);                                  % Group sample means


    for i = 1:k
        % we create the binary table Group, whose binary ones points the records of trainng data that belong to the class "i"
        Group            =(target == ClassLabel(i));
        % we make the values of the matrix "Group" from binary to logic, so we can find the sum of each row (class) "i" in the table "nGroup(i)"
        trial            = double(Group);
        nGroup(i)        = sum(trial);    
        % We take only the elements that belong in Group (class) "i"
        ClassElem        =InputTrain(Group,:);
        % Calculate group mean vectors
        GroupMean(i,:)   =(sum(ClassElem))/nGroup(i);
        % Calculate covariance matrix for all classes
        % First we create the matrix "fixSizeOfmean". 
        % It is a table with the value of the current GroupMean repeted as many times as the number of the elements belonging to this group (class)
        fixSizeOfmean    =repmat(GroupMean(i,:),nGroup(i),1);
        % We did that in order to be able to find later in the matrix"covariance" the covariance between the class elements and their GroupMean
        covariance       =((ClassElem - fixSizeOfmean))'*(ClassElem - fixSizeOfmean);
        % Assuming we have identical covariance matrix for all class-conditional we calculate the sum of the covariance of the classes
        Sum              = Sum + covariance;  
    end
    Scov        = Sum/(n-k);
    % Use the train data probabilities for each class
    PriorProb   = nGroup / n;
    % Loop over classes to calculate linear discriminant coefficients for every data
    for i = 1:k
        type1   = GroupMean(i,:) / Scov;
        type2   = -0.5 * type1 * GroupMean(i,:)';
        type3   = InputTrain / Scov;
        type4   = type3 * GroupMean(i,:)';
        type5   = type2 + log(PriorProb(i));
        W(:,i)  = type4 + repmat(type5,n,1);

        type3test   = InputTest / Scov;
        type4test   = type3test * GroupMean(i,:)';
        Wtest(:,i)  = type4test + repmat(type5,ntest,1);
    end
    
% For the Train set:
    %we compare each linear score for each data for each class to put label to
    %every new point
    for i=1:n
        if W(i,1)>W(i,2) && W(i,1)>W(i,3)
            A(i)=0;
        else if W(i,2)>W(i,3)
            A(i)=1;
            else A(i)=2;
            end
        end
    end

% We find how many of our predictions where right and we calculate the error on the training set 
    bad = (A'==target);
    ldaErrTrain = (n - sum(bad)) * (n/100);

% For the Test set:
    %we compare each linear score for each data for each class to put label to
    %every new point
    for i=1:ntest
        if Wtest(i,1)>Wtest(i,2) && Wtest(i,1)>Wtest(i,3)
            Atest(i)=0;
        else if Wtest(i,2)>Wtest(i,3)
            Atest(i)=1;
            else Atest(i)=2;
            end
        end
    end

% We find how many of our predictions where right and we calculate the error on the test set
    badtest = (Atest'==targettest);
    ldaErrTest = (ntest - sum(badtest)) * (ntest/100);


end

