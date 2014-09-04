function [KamaCentr,RosaCentr,CanadianCentr,BKama,BRosa,BCanadian] = KMeansCluster( train )

% KMeansCluster - MATLAB function to Perform 3-means clustering of the training data and report 
% the three cluster centers and the elements they own.
% by Dimitris Tsirozoglou
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

Train   = train(:,1:end);
BestAcc = 0;

    for b = 1:100                   % we repeat 100 times and its time we take different initialized centroids
 
        % We initialize our three centroids and we ensure that there will
        % be three different ones by using "rng Shuffle"
        RandCenK =randi([1 size(train,1)],1);
        rng Shuffle
        RandCenR =randi([1 size(train,1)],1);
        rng Shuffle
        RandCenC =randi([1 size(train,1)],1);

        Ckama = train(RandCenK,(1:end-1));          % Initialize the cluster center of the first class (Kama).
        Crosa = train(RandCenR,(1:end-1));          % Initialize the cluster center of the second class (Rosa).
        Canad = train(RandCenC,(1:end-1));          % Initialize the cluster center of the third class (Canadian).
    
        Kama        = zeros(size(Train,1),size(Train,2));       % Initialize the elements of the first cluster.
        Rosa        = zeros(size(Train,1),size(Train,2));       % Initialize the elements of the second cluster.
        Canadian    = zeros(size(Train,1),size(Train,2));       % Initialize the elements of the third cluster.

        for i = 1:10000

            l=0;                % counter of elements of the first class (Kama)
            r=0;                % counter of elements of the second class (Rosa)
            c=0;                % counter of elements of the third class (Canadian)

            NKama       = zeros(size(Train,1),size(Train,2));
            NRosa       = zeros(size(Train,1),size(Train,2));
            NCanadian   = zeros(size(Train,1),size(Train,2));

            for j =1:size(Train,1)      % for every record of Train 
                
                % We calculate the euclideandistance of every record from our three class centers
                euclideandistanceK = sum((Ckama-Train(j,(1:end-1))).^2,2);
                euclideandistanceR = sum((Crosa-Train(j,(1:end-1))).^2,2);
                euclideandistanceC = sum((Canad-Train(j,(1:end-1))).^2,2);

                % Similarity is measured by the Euclidean distance. We assign the record to the Cluster with the shortest distance. 
                if euclideandistanceK < euclideandistanceR && euclideandistanceK < euclideandistanceC
                     l=l+1;
                     NKama(l,:) = Train(j,:);
                elseif euclideandistanceC < euclideandistanceR && euclideandistanceC < euclideandistanceK
                    c=c+1;
                    NCanadian(c,:) = Train(j,:);
                elseif euclideandistanceR < euclideandistanceK && euclideandistanceR < euclideandistanceC
                    r=r+1;
                    NRosa(r,:) = Train(j,:);
                elseif euclideandistanceK == euclideandistanceR && euclideandistanceK == euclideandistanceC
                    nList=[0 1 2];
                    RandClass =floor(rand*length(nList));
                    if RandClass == 0
                        l=l+1;
                        NKama(l,:) = Train(j,:);
                    elseif RandClass == 1
                        r=r+1;
                        NRosa(r,:) = Train(j,:);
                    else
                        c=c+1;
                        NCanadian(c,:) = Train(j,:);
                    end   
                elseif euclideandistanceK == euclideandistanceR
                        nList=[0 1];
                        RandClass =floor(rand*length(nList));                        
                        if RandClass == 1
                            r=r+1;
                            NRosa(r,:) = Train(j,:);
                        else
                            l=l+1;
                            NKama(l,:) = Train(j,:);
                        end  
                elseif euclideandistanceK == euclideandistanceC
                            nList=[0 2];
                            RandClass =nList(floor(rand*length(nList))+1);                  
                            if RandClass == 0
                                l=l+1;
                                NKama(l,:) = Train(j,:);
                            else
                                c=c+1;
                                NCanadian(c,:) = Train(j,:);
                            end           
                else                    
                        nList=[1 2];
                        RandClass =nList(floor(rand*length(nList))+1);
                        if RandClass == 1
                            r=r+1;
                            NRosa(r,:) = Train(j,:);
                        else
                            c=c+1;
                            NCanadian(c,:) = Train(j,:);
                        end
                end
            end
            
            % We have assigned each data point to cluster represented by the most similar prototype. 
            % This leads to a new partitioning of the data.
            % Thus we Recompute cluster centroids as mean of data points assigned to respective cluster.
            Ckama = sum(NKama(:,1:(end-1)),1) / l;
            Crosa = sum(NRosa(:,1:(end-1)),1) / r;
            Canad = sum(NCanadian(:,1:(end-1)),1) / c;

            % We check if the elements of the groups of the recomputed
            % centoid are the same as the previous one's.
            if (Rosa == NRosa) 
                if (Kama == NKama)
                    if (Canadian == NCanadian)
                        i;
                        break     % if they are the same break(we don't need to recompute the cluster centers)
                    end
                end
            end

            Rosa        = NRosa;
            Kama        = NKama;
            Canadian    = NCanadian;
        end

        % We find the Accurancy of our clustering using differntly initialized class centroids
        Acc(b) = sum((NRosa(1:r,8)==1)) + sum((NKama(1:l,8)==0)) + sum((NCanadian(1:c,8)==2));    
        
        % We find the best accurancy achieved by our algorithm and we store
        % the three centroids and the elements of the classes.
        if Acc(b)>BestAcc
            BestAcc = Acc(b);
            BKama = NKama(1:l,:);
            BRosa = NRosa(1:r,:);
            BCanadian = NCanadian(1:c,:);
            KamaCentr = Ckama;
            RosaCentr = Crosa;
            CanadianCentr = Canad;
        end
    end
end
