close all; clear all;
filepath = 'yourfilepath';

%%
Data = load(filepath);
Results = Data.results;
Annotations = categorical(Data.annotations);
Features = Data.features;
Patient_IDS = Data.patient_ids;
selected_features = [1:length(Features)];
selected_idxs = Annotations~='Stenose';

y = Annotations(selected_idxs);
X = Results(selected_idxs, selected_features);
Features = Features(selected_features);

cvp = cvpartition(y,'holdout',0.3);
Xtrain = X(cvp.training, :);
ytrain = y(cvp.training, :);
Xtest = X(cvp.test, :);
ytest = y(cvp.test, :);
idxs = 1:size(Xtrain, 1);

nbootstraps = 100; 
n = (length(y)*0.8);
lambdavals = linspace(0,8,40)/n;
Feature_Weights_Min = zeros(nbootstraps, size(X, 2));
Feature_Weights_Min1SE = zeros(nbootstraps, size(X, 2));
Mean_Losses = zeros(nbootstraps, length(lambdavals));


%%
for x = 1:nbootstraps
    disp(x);
    btstrp = datasample(idxs, size(Xtrain, 1));
    Xb = Xtrain(btstrp, :);
    yb = ytrain(btstrp, :);

    cvp = cvpartition(yb,'kfold',5);
    numvalidsets = cvp.NumTestSets;

    n = length(yb);
    lossvals = zeros(length(lambdavals),numvalidsets);

    for i = 1:length(lambdavals)
        for k = 1:numvalidsets
            Xt = Xb(cvp.training(k),:);
            yt = yb(cvp.training(k),:);
            Xvalid = Xb(cvp.test(k),:);
            yvalid = yb(cvp.test(k),:);

            nca = fscnca(Xt,yt,'FitMethod','exact', ...
                 'Solver','sgd','Lambda',lambdavals(i), ...
                 'IterationLimit',30,'GradientTolerance',1e-4, ...
                 'Standardize',true);

            lossvals(i,k) = loss(nca,Xvalid,yvalid,'LossFunction','classiferror');
        end
    end

    meanloss = mean(lossvals,2);
    Mean_Losses(x, :) = meanloss;

    [minimum,idx] = min(meanloss); % Find the index
    stdev = std(meanloss);

    [dmin, didx] = min(abs(meanloss(idx:end)-(minimum+stdev)));
    lambdamin = lambdavals(idx);
    lambdamin1SE = lambdavals(idx+didx-1);


    nca = fscnca(Xb,yb,'FitMethod','exact','Solver','sgd',...
        'Lambda',lambdamin,'Standardize',true);
    nca2 = fscnca(Xb,yb,'FitMethod','exact','Solver','sgd',...
        'Lambda',lambdamin1SE,'Standardize',true);
    
    Feature_Weights_Min(x, :) = nca.FeatureWeights;
    Feature_Weights_Min1SE(x,:) = nca2.FeatureWeights;
end


%% plots
close all

figure('Name', 'Feature Weights Min1SE');
[~, i] = sort(mean(Feature_Weights_Min1SE), 'descend');
boxplot(normalize(Feature_Weights_Min1SE(:,i),2), 'labels', Features(1,i));

figure('Name', 'Feature Weights Min');
[~, j] = sort(mean(Feature_Weights_Min), 'descend');
boxplot(normalize(Feature_Weights_Min(:,j), 2), 'labels', Features(1,j));


figure('Name', 'Mean_Losses');
errorbar(lambdavals(1:12), mean(Mean_Losses(:,1:12), 1), std(Mean_Losses(:,1:12), 0, 1));
hold on
errorbar(lambdavals(12:end), mean(Mean_Losses(:,12:end), 1), std(Mean_Losses(:,12:end), 0, 1), 'color', [0.9100 0.4100 0.1700]);
xlabel('Lambda value')
ylabel('Loss')
ax = gca;
ax.XAxis.FontSize = 12;
ax.YAxis.FontSize = 12;

%%
figure();
[values, i] = sort(sum(Feature_Weights_Min>mean(Feature_Weights_Min,2))/nbootstraps*100, 'descend');
idxs = values>50;
Features_copy = Features(i);
for j = 1:length(Features_copy); Features_copy(j) = strrep(Features_copy(j), '_', ' '); end
labels = categorical(Features_copy(idxs));
labels = reordercats(labels, Features_copy(idxs));

bar(labels(:),values(idxs));
ylabel('frequency of selection [%]', 'FontSize', 12)
ax = gca;
ax.XAxis.FontSize = 12;

features = Xtrain(:,idxs);
t = templateSVM('Standardize','on', 'KernelFunction', 'polynomial', 'PolynomialOrder', 2);
          
svmMdl1 = fitcecoc(Xtrain, ytrain, 'Coding', 'Onevsone', 'Learners',t);
svmMdl2 = fitcecoc(features,ytrain, 'Coding', 'Onevsone', 'Learners',t);

p1 = predict(svmMdl1, Xtest(:,:));
p2 = predict(svmMdl2, Xtest(:,idxs));

L1 = loss(svmMdl1,Xtest,ytest);
L2 = loss(svmMdl2,Xtest(:,idxs),ytest);

%%    
Cms = confusionmat(ytest, p1);
%Cms = confusionmat(ytest, p2)

TP = Cms(1, 1); FP = Cms(1, 2); FN = Cms(2, 1); TN = Cms(2, 2);

Accuracy = (TP+TN)./(TP+TN+FN+FP)
Sensitivity = (TP)./(TP+FN)
Specificity = (TN)./(TN+FP)
precision = TP./(TP+FP)
