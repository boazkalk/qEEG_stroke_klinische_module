%% without class
function [results, annotations, labels] = ...
    stroke_analysis(Data, Epoch_Duration, Epoch_Overlap, Stenose_Time, ...
                    Clamp_Time, Shunt_Time, Sample_Frequency, Lead_Pairs, ...
                    Leads)
                                               
    Max_Frequency = Sample_Frequency/2;
    Nr_Leads = size(Data, 1);
    
    times = {Stenose_Time, Clamp_Time, Shunt_Time};
    names = {'Stenose', 'Clamp', 'Shunt'};
    
    alpha_power = zeros(1, Nr_Leads);
    beta_power = zeros(1, Nr_Leads);
    delta_power = zeros(1, Nr_Leads);
    theta_power = zeros(1, Nr_Leads);
    pdBSI = zeros(1, 1);
    lf_pdBSI = zeros(1, 1);
    hf_pdBSI = zeros(1, 1);
    annotations = cell(1, 1);
    
    k = 1;        
    for i = 1:numel(times)
        for j = 1:size(times{i}, 1)
            current_time = times{i}(j, 1);
            while current_time + Epoch_Duration <= times{i}(j, 2)
                [start_sample, end_sample] = time_to_samples(current_time, ...
                                            Epoch_Duration, Sample_Frequency);
                Pxx_save = [];
                for lead = 1:Nr_Leads
                    subset = Data(lead, start_sample:end_sample);
                    [Pxx,~] = perio_hamming(subset, Epoch_Duration, ...
                                        Max_Frequency, Sample_Frequency);
                    Pxx_save = [Pxx_save; Pxx];      
                    alpha_power(k, lead) = freq_power(Pxx, Epoch_Duration, 8, 12, 0.5);
                    beta_power(k, lead) = freq_power(Pxx, Epoch_Duration, 12, 30, 0.5);
                    delta_power(k, lead) = freq_power(Pxx, Epoch_Duration, 1, 4, 0.5);
                    theta_power(k, lead) = freq_power(Pxx, Epoch_Duration, 4, 8, 0.5);
                end 
                annotations(k, 1) = {names{i}};
                pdBSI(k, 1) = calculate_pdBSI(Pxx_save, Lead_Pairs, 0.5, 31, Epoch_Duration, 0.1, 0.5);
                lf_pdBSI(k, 1) = calculate_pdBSI(Pxx_save, Lead_Pairs, 1, 8, Epoch_Duration, 0.1, 0.5);
                hf_pdBSI(k, 1) = calculate_pdBSI(Pxx_save, Lead_Pairs, 8, 30, Epoch_Duration, 0.1,0.5);
                
                current_time = current_time + Epoch_Duration*(1-Epoch_Overlap);
                k = k+1;
            end
        end
    end
    
    DAR = delta_power./alpha_power;
    DTABR = (delta_power+theta_power)./(alpha_power+beta_power);
    [mean_DAR, mean_DAR_L, mean_DAR_R] = meanLR(DAR, Lead_Pairs);
    [mean_DTABR, mean_DTABR_L, mean_DTABR_R] = meanLR(DTABR, Lead_Pairs);
    [mean_alpha, mean_alpha_L, mean_alpha_R] = meanLR(alpha_power, Lead_Pairs);
    [mean_beta, mean_beta_L, mean_beta_R] = meanLR(beta_power, Lead_Pairs);
    [mean_delta, mean_delta_L, mean_delta_R] = meanLR(delta_power, Lead_Pairs);
    [mean_theta, mean_theta_L, mean_theta_R] = meanLR(theta_power, Lead_Pairs);
    
    variables = {'alpha_power', 'beta_power', 'delta_power', ...
                 'theta_power', 'DAR', 'DTABR'};
    labels = create_labels(Leads, variables);
    labels = ['pdBSI', 'lf_pdBSI', 'hf_pdBSI', 'mean_DAR', ...
              'mean_DAR_L', 'mean_DAR_R', 'mean_DTABR', 'mean_DTABR_L', ...
              'mean_DTABR_R', 'mean_alpha', 'mean_alpha_L', ...
              'mean_alpha_R', 'mean_beta', 'mean_beta_L', ...
              'mean_beta_R', 'mean_delta', 'mean_delta_L', ...
              'mean_delta_R', 'mean_theta', 'mean_theta_L', ...
              'mean_theta_R', labels];
          
    results = [pdBSI, lf_pdBSI, hf_pdBSI, mean_DAR, mean_DAR_L, ...
               mean_DAR_R, mean_DTABR, mean_DTABR_L, mean_DTABR_R, mean_alpha, ...
               mean_alpha_L, mean_alpha_R, mean_beta, mean_beta_L, ...
               mean_beta_R, mean_delta, mean_delta_L, mean_delta_R, ...
               mean_theta, mean_theta_L, mean_theta_R, alpha_power, ...
               beta_power, delta_power, theta_power, DAR, DTABR,];
    
end
                            
function [start_sample, end_sample] = time_to_samples(start_time, duration, sample_frequency)
    start_sample =  ceil(start_time*sample_frequency);
    end_sample   =  ceil(duration*sample_frequency) + start_sample;
end

function [Pxx,F] = perio_hamming(data, segmentlength_sec, fmax, Fs)
    [Pxx,F] = periodogram(data, hamming(length(data)),...
                          [0:1/segmentlength_sec:fmax],Fs);
end

function power = freq_power(Pxx, segment_duration, lower_f, upper_f, base_lower_f)
    freq_band = Pxx(round((lower_f*segment_duration+1)):(round(upper_f*segment_duration+1)));
    base    = Pxx(round(base_lower_f*segment_duration+1):end);
    power   = trapz(freq_band)/trapz(base);
end

function pdBSI = calculate_pdBSI(Pxx, lead_pairs, lower_f, upper_f, segment_duration, df, base_lower_f)
    pdBSI = 0;
    for i = lower_f:df:upper_f
        for j = 1:size(lead_pairs, 2)
            Left = freq_power(Pxx(lead_pairs(1, j),:), segment_duration, i, i+df, base_lower_f);
            Right = freq_power(Pxx(lead_pairs(2, j),:), segment_duration, i, i+df, base_lower_f);
            x = (Right-Left)/(Right+Left);
            pdBSI = pdBSI + abs(x);
        end
    end
    pdBSI = pdBSI / ((upper_f-lower_f)/df*j);
end
            
function selected_data = select_data(data, stenose_time, clamp_time, shunt_time, sample_frequency)
    selected_data = [];
    times = [stenose_time; clamp_time; shunt_time];
    samples = ceil(times*sample_frequency);
    for i = 1:size(samples, 1)
        selected_data = [selected_data, data(:,samples(i, 1):samples(i,2))];
    end
end

function labels = create_labels(leads, variables)
    nr_leads = length(leads);
    nr_variables = length(variables);
    labels = cell(1, nr_leads*nr_variables);
    for i = 1:nr_variables
        for j = 1:nr_leads
            label = {append(variables{i}, '_', leads{j})};
            idx = (i-1)*nr_leads+j;
            labels(1, idx) = label;
        end
    end
end

function [average, average_L, average_R] = meanLR(Data, lead_pairs)
    average = mean(Data, 2);
    average_L = mean(Data(:, lead_pairs(1,:)), 2);
    average_R = mean(Data(:, lead_pairs(2,:)), 2);
end
