clear all
close all

data = load('C:\Users\20164798\OneDrive - TU Eindhoven\UNI\ME 2\KM1\Project 1\code\results\results_elab_ED_10_EO_0.0.mat');
selected_annotations = {'Clamp', 'Shunt'};
selected_features = {'pdBSI','lf_pdBSI','hf_pdBSI','mean_DAR','mean_DAR_L','mean_DAR_R','mean_DTABR','mean_DTABR_L','mean_DTABR_R','mean_alpha','mean_alpha_L','mean_alpha_R','mean_beta','mean_beta_L','mean_beta_R','mean_delta','mean_delta_L','mean_delta_R','mean_theta','mean_theta_L','mean_theta_R'};

%%
userinput = struct;

userinput.algo = 'knn';
userinput.coding = 'onevsone';
userinput.standardize = 'on';
userinput.kernel = 'polynomial';
userinput.order = 4;
userinput.distance = 'euclidean';
userinput.neigh = 5;

userinput.datarandom_patient = 1;
userinput.trainpercrand = 80;

%%
idxs = ismember(data.annotations, selected_annotations);
cols = ismember(data.features, selected_features);

features = data.results(idxs, cols);
annotations = data.annotations(idxs);
userinput.patient = data.patient_ids(idxs);

%%
[Cms, cm_labels, Accuracy, Sensitivity, Specificity, F1] = ML_func_revised(features, annotations, userinput);
