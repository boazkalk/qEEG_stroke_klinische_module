function [cm, cm_labels, Accuracy, Sensitivity, Specificity, F1] = ML_func_revised(data, annotaties, UI)

% UI (userinput) is a struct containing:
% 
% UI.algo = 'knn'/'svm' - declare ML algorithm
% UI.coding = 'onevsone'/'onevsall'/'binarycomplete'/'ordinal'/'ternarycomplete' -  declare ML coding
% UI.standardize = 'on'/'off'- standardize data
% UI.kernel = 'linear'/'rbf'/'gaussian'/'polynomial' - declare SVM kernel if algo=svm
% UI.order = 2/3/4 - declare polynomial order if kernel=polynomial and algo=svm
% UI.distance = 'cityblock'/'chebychev'/'euclidean'/'minkowski' - declare distance metric if algo=knn
% UI.neigh = >1 - declare amount of neighbours if algo=knn
% 
% UI.datarandom_patient = 0/1 - 0=random training/testing data division, 1=patient-to-patient
% UI.trainpercrand = 1-99 - percentage training data if datarandom_patient=0
% UI.patient = array of patient numbers that links rows of data and annotations to patients
% UI.pos_ann = the annotation that is to be regarded as positive in the metrics

    if ~((size(data, 1)==size(annotaties, 1))&&(size(data, 1)==size(UI.patient, 1)))
        disp('ERROR: length of data, annotaties and patient numbers are not equal')
        return
    elseif ~ismember(UI.coding, {'onevsone', 'onevsall', 'binarycomplete' ...
                                        'ordinal', 'ternarycomplete'})
        disp('ERROR: wrong input coding') 
        return
    elseif ~ismember(UI.standardize, {'on', 'off'})
        disp('ERROR: wrong input standardize')
        return
    end
    
    if UI.algo == "svm"
        if ~ismember(UI.kernel, {'linear', 'gaussian', 'rbf'})
            if UI.kernel ~= "polynomial"
                disp('ERROR: wrong input kernel')
                return
            else
                if floor(UI.order) ~= UI.order || ~(1<UI.order<5)
                    disp('ERROR: wrong polynomial order')
                    return
                end
                t = templateSVM('Standardize',UI.standardize, 'KernelFunction', UI.kernel, 'PolynomialOrder', UI.order);
            end
        else 
             t = templateSVM('Standardize',UI.standardize, 'KernelFunction', UI.kernel);   
        end
        
    elseif UI.algo == "knn"
        if ~ismember(UI.distance, {'cityblock', 'chebychev', ...
                                          'euclidean', 'minkowski'})
            disp('ERROR: wrong input distance')
            return
        elseif floor(UI.neigh)~=UI.neigh || UI.neigh <= 0
            disp('ERROR: wrong amount of neighbours')
            return
        end
        t = templateKNN('Standardize', UI.standardize, 'NSMethod', 'kdtree', 'Distance', UI.distance, 'NumNeighbors', UI.neigh);
    else 
        disp('ERROR: wrong input algorithm declaration')
        return
    end
    
    if UI.datarandom_patient == 0 
        if floor(UI.trainpercrand)~=UI.trainpercrand || ~((0 <= UI.trainpercrand) && (UI.trainpercrand <= 100))
            disp('ERROR: wrong training data percentage, use value between 0-100')
            return
        end
        
        n_folds = UI.trainpercrand / (100-UI.trainpercrand) + 1;
        cv = cvpartition(size(data, 1), 'KFold', n_folds);
        test_idxs = zeros(cv.NumTestSets, cv.NumObservations);
        for i = 1:cv.NumTestSets; test_idxs(i,:) = cv.test(i); end
        
    elseif UI.datarandom_patient == 1
        patient_ids = unique(UI.patient);
        test_idxs = zeros(length(patient_ids), size(data, 1));
        for i = 1:length(patient_ids); test_idxs(i,:) = (UI.patient==patient_ids(i)); end
        
    else
        disp('ERROR: wrong training data division declaration, use 0 or 1')
        return
    end
    
    test_idxs = logical(test_idxs);
    cm_labels = sort(unique(annotaties));
    cm = zeros(length(cm_labels), length(cm_labels), size(test_idxs,1));
    
    for i = 1:size(test_idxs, 1)
        [Training_data, Training_class, Testing_data, Testing_class] = traintestdivision(data, annotaties, test_idxs(i,:));        
        model = fitcecoc(Training_data, Training_class, 'Coding', UI.coding, 'Learners',t);
        
        [Predictions, ~] = predict(model, Testing_data);
        cm(:,:,i) = confusionmat(Testing_class, Predictions)./length(Testing_class);
    end
    
    if size(cm, 1) == 2
        if cm_labels(1)~= "Clamp"; cm = flip(flip(cm, 1),2); end
        [Accuracy, Sensitivity, Specificity, F1] = results(cm);
    else
        Accuracy = sum(bsxfun(@times, eye(size(cm(:,:,1))), cm), [1, 2]);
        Sensitivity=[]; Specificity=[]; F1=[];      
        fprintf('Accuracy = %.3f (+/- %.3f)\n', mean(Accuracy), std(Accuracy))
    end
end

function [Training_data, Training_class, Testing_data, Testing_class] = traintestdivision(data, annotaties, idxs)
    Training_data = data(~idxs, :);
    Testing_data = data(idxs, :);
    Training_class = annotaties(~idxs); 
    Testing_class = annotaties(idxs); 
end

function [Accuracy, Sensitivity, Specificity, F1] = results(Cms)
    TP = Cms(1, 1, :); FP = Cms(1, 2, :); FN = Cms(2, 1, :); TN = Cms(2, 2, :);
    
    Accuracy = (TP+TN)./(TP+TN+FN+FP);
    Sensitivity = (TP)./(TP+FN);
    Specificity = (TN)./(TN+FP);
    precision = TP./(TP+FP);
    recall = TP./(TP+FN);
    F1 = 2.*(precision.*recall)./(precision+recall);
    
    fprintf('Accuracy = %.3f (+/- %.3f)\n', mean(Accuracy), std(Accuracy))
    fprintf('Sensitivity = %.3f (+/- %.3f)\n', mean(Sensitivity), std(Sensitivity))
    fprintf('Specificity = %.3f (+/- %.3f)\n', mean(Specificity), std(Specificity))
    fprintf('F1 = %.3f (+/- %.3f)\n', mean(F1), std(F1))
end
