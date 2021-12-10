clear all; close all;
%%
DATA_DIR                = "C:\Users\20164798\OneDrive - TU Eindhoven\UNI\ME 2\KM1\Project 1\data";

EPOCH_DURATIONS          = [10 20 30 40 50 60];   % [s]
EPOCH_OVERLAPS           = 0;  % fraction

SELECTED_LEADS = (2:22);
LEAD_PAIRS = [1:2:16 20; 2:2:16 21];

for i = 1:length(EPOCH_DURATIONS)
    for j = 1:length(EPOCH_OVERLAPS)
        %%
        EPOCH_DURATION = EPOCH_DURATIONS(i);
        EPOCH_OVERLAP = EPOCH_OVERLAPS(j);

        results = [];
        annotations = [];
        patient_ids = [];

        for x = 1:4
            Filename_Edf            = "shunt_pat"+x+".edf";
            Filename_Annotations    = "shunt_pat"+x+"_annotations.txt";
            fprintf('\ncalculate for patient %i\nwith ED: %.0f and EO: %.1f\n', ...
                    x, EPOCH_DURATION, EPOCH_OVERLAP)

            %load the EDF data
            [info, data] = edfreadUntilDone_x(DATA_DIR+"/"+Filename_Edf);
            [stenose_time, clamp_time, shunt_time] = read_annotations(DATA_DIR+"/"+Filename_Annotations);

            %create DAR_analyse class
            [results_sub, annotations_sub, features] = ...
                stroke_analysis(data(SELECTED_LEADS,:), EPOCH_DURATION, ...
                                EPOCH_OVERLAP, stenose_time, clamp_time,...
                                shunt_time, info.frequency(2), LEAD_PAIRS, ...
                                info.label(SELECTED_LEADS));

            results = [results; results_sub];
            annotations = [annotations; annotations_sub];
            patient_ids = [patient_ids; repelem(x, size(results_sub, 1), 1)];
        end
        
        %%
        results_table = array2table(results, 'VariableNames', features);
        results_table.Patient_ID = patient_ids;
        results_table.Annotation = annotations;
        results_table = [results_table(:,end-1:end) results_table(:,1:end-2)];

        %%
        filename = sprintf('results_elab_ED_%.0f_EO_%.1f.mat',EPOCH_DURATION, EPOCH_OVERLAP);
        save("results/"+filename, 'results', 'annotations', 'patient_ids', 'features', 'results_table')
        
    end
end
