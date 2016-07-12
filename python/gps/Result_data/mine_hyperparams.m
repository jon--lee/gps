% Script to mine the data generated after sweeping through differeent
% hyperparam data
clear; clc;

addpath(pwd);        % Adds any function path to Matlab in the main directory
experiment_run  = 4;            % Experiment run number 2
path_to_script  = strcat(pwd, '/Run_', int2str(experiment_run));    % Generates the run directory path
cd(path_to_script);

agent_file      = 'agent_set.csv';
algorithm_file  = 'algorithm_set.csv';

files = {algorithm_file, agent_file};
j = 1;

for i = 1:length(files)
    try
        file_handle(i)      = fopen(files{i}, 'r');  % Opens file as a read only object
        switcher            = true;                  % switch between read param and parsing data
        current_name        = '';
        read_line       = fgets(file_handle(i));
        param_val           = [];   param_result = [];
        
        while read_line ~= -1
            % Reads the entire file till the end
            
            if(switcher)
                % Parses parameter
                hash_indx           = strfind(read_line, '#');
                param_name          = read_line(hash_indx(1) +1 : hash_indx(2) -1);
                
                if isempty(current_name)
                    current_name;   % Do nothing in loop
                
                elseif ~strcmp(strtrim(current_name), strtrim(param_name))
                    save_data(j).name           = current_name;     % Saves all data into a structured array
                    save_data(j).param_val      = param_val;
                    save_data(j).param_result   = param_result;
                    j = j + 1;

                    param_val                   = [];
                    param_result                = [];
                end
                
                current_name        = param_name;
                
                if isempty(str2num(read_line(hash_indx(2) +1 : end)))
                    param_val       = [param_val, {read_line(hash_indx(2) +1 : end)}];
                else
                    param_val       = [param_val , str2num(read_line(hash_indx(2) +1 : end))];
                end
                
            else
                % Parses array value 
                split_val           = strsplit(read_line, ',');     % Splits based on commas
                list_val            = str2double(split_val);
                param_result        = [param_result , process_result(list_val)];     % Declare a function process_result that will handle the clawed values of each run    
            end
            
            read_line       = fgets(file_handle(i));
            switcher            = ~switcher;
        end
        
        save_data(j).name           = current_name;     % Final save for data
        save_data(j).param_val      = param_val;
        save_data(j).param_result   = param_result;
        j = j + 1;
        
    catch excpt
        disp('error opening file');
        disp(excpt.message);
    end
    
    fclose(file_handle(i));
end

H1 = figure(1);
H2 = figure(2);
j = 0;

for i = 1:length(save_data)
    if j >= 4
        set(0,'CurrentFigure',H2);
    else
        set(0,'CurrentFigure',H1);
    end
    
    if strcmp(strtrim(save_data(i).name), 'weights')
       % Plot weights column separately
       continue 
    end
    
    subplot(2 ,2 ,mod(j, 4) +1);
    grid on 
    
    val     = save_data(i).param_val; 
    result  = save_data(i).param_result;
    plot(val, result);
    title(sprintf('Hyperparameter: %s', save_data(i).name));
    
    j = j + 1;
end
