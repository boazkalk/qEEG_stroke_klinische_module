close all; clear all;
Data = load('results/results_new_ED_10_EO_0.5.mat');
Results = Data.results;
Results_NN = Data.results_norm;
Annotations = Data.annotations;
Labels = Data.labels;
Usable_Leads = [1:16 20 21];

results_selected = Results_NN;
X = results_selected(:,Usable_Leads);
y = categorical(Annotations);

cvp = cvpartition(y,'holdout',0.2);
Xtrain = X(cvp.training, :);
ytrain = y(cvp.training, :);
Xtest = X(cvp.test, :);
ytest = y(cvp.test, :);
idxs = 1:size(Xtrain, 1);

nbootstraps = 10; 
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


%%
figure();
boxplot(Feature_Weights_Min1SE, 'labels', {Labels{1,Usable_Leads}});

figure();
boxplot(Feature_Weights_Min, 'labels', {Labels{1,Usable_Leads}});
%%

figure();
errorbar(lambdavals, mean(Mean_Losses, 1), std(Mean_Losses, 0, 1));

figure();
plot(lambdavals, Mean_Losses);

%%
figure();
values = sum(Feature_Weights_Min1SE>0.7)/nbootstraps;
labels = categorical(Labels(Usable_Leads));
idxs = values>0.5;
bar(labels(idxs), values(idxs), 'g');
hold on
bar(labels(~idxs), values(~idxs), 'k');
hold off

%%
features = Xtrain(:,idxs);

%%
svmMdl1 = fitcecoc(Xtrain, ytrain);
svmMdl2 = fitcecoc(features,ytrain);

p1 = predict(svmMdl2, Xtest(:,idxs))

%%
L1 = loss(svmMdl1,Xtest,ytest)
L2 = loss(svmMdl2,Xtest(:,idxs),ytest)
