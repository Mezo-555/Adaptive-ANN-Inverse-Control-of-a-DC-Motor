% --- Step 1: Data Preparation for ANN Training ---
% This script runs after the Simulink model produces 'u_data' and 'y_data'.

% We are training the network: u(k-1) = f(y(k), y(k-1), y(k-2))
%
% So, Inputs = [y(k), y(k-1), y(k-2)]
%    Target = u(k-1)

% Ensure 'y_data' and 'u_data' are column vectors
if size(y_data, 2) > size(y_data, 1)
    y_data = y_data';
end
if size(u_data, 2) > size(u_data, 1)
    u_data = u_data';
end

% Get the total number of samples
N = length(y_data);

% We need to create vectors that align.
% The first valid row we can create is for k=3, which gives:
% y(3), y(2), y(1)  and  u(2)
%
% The last valid row is for k=N, which gives:
% y(N), y(N-1), y(N-2) and u(N-1)

% Create the time-shifted vectors
y_k   = y_data(3:N);     % y(k)
y_k_1 = y_data(2:N-1);   % y(k-1)
y_k_2 = y_data(1:N-2);   % y(k-2)

% Create the target vector
u_k_1 = u_data(2:N-1);   % u(k-1)

% Check for length consistency (this should match)
% We have N-2 samples for training
if length(y_k) ~= length(u_k_1)
    error('Vector lengths do not match. Check data simulation.');
end

% Separate into final inputs and targets
ann_inputs = [y_k, y_k_1, y_k_2];   % Columns [y(k), y(k-1), y(k-2)]
ann_targets = u_k_1;                % Column  [u(k-1)]

disp('ANN training data created successfully.');
disp('Input matrix size (ann_inputs):');
disp(size(ann_inputs));
disp('Target vector size (ann_targets):');
disp(size(ann_targets));


% --- Step 2: Export Data to Excel ---
% Combine inputs and targets for easy export
output_table = array2table([ann_inputs, ann_targets], ...
    'VariableNames', {'y_k', 'y_k_1', 'y_k_2', 'u_k_1'});

filename = 'ann_training_data.xlsx';
writetable(output_table, filename);

fprintf('Training data successfully exported to %s\n', filename);