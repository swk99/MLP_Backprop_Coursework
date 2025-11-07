%% Generalized Incremental Backpropagation MLP with randn initialization
clc; clear;

%% ===== 1. Network parameters =====
N_input = 2;       % number of input nodes
N_hidden = 3;      % number of hidden nodes
N_output = 1;      % number of output nodes
eta = 0.2;         % learning rate
n_epochs = 1;      % number of epochs (for demonstration)

%% ===== 2. Training data =====
X = [0 1; 1 0]';   % input: N_input x N_samples
T = [1 1];         % target: 1 x N_samples
N_samples = size(X,2);

%% ===== 3. Initialize weights with randn =====
rng(0); % for reproducibility
sigma = 0.1;  % standard deviation for random initialization
Weights_input_hidden = sigma * randn(N_hidden, N_input);  % N(0, 0.1^2)
Weights_hidden_output = sigma * randn(N_output, N_hidden); % N(0, 0.1^2)

%% ===== 4. Sigmoid function =====
sig = @(x) 1./(1+exp(-x));
sig_deriv = @(y) y .* (1-y);

%% ===== 5. Training loop (incremental) =====
for epoch = 1:n_epochs
    fprintf('Epoch %d\n', epoch);
    
    for p = 1:N_samples
        x = X(:,p);    % current input vector
        t = T(p);      % current target
        
        %% --- Forward Pass ---
        h_in = Weights_input_hidden * x;          % hidden net input
        h_out = sig(h_in);                        % hidden output
        y_in = Weights_hidden_output * h_out;    % output net input
        y_out = y_in;                             % linear output
        
        fprintf('\nSample %d:\n', p);
        fprintf('Forward pass outputs: ');
        fprintf('h_out = '); fprintf('%.4f ', h_out); 
        fprintf(', y_out = %.4f\n', y_out);
        
        %% --- Backward Pass ---
        beta_out = t - y_out;                     % output error
        beta_hidden = (h_out .* (1 - h_out)) .* (Weights_hidden_output' * beta_out); % hidden errors
        
        fprintf('Backward pass betas: ');
        fprintf('beta_hidden = '); fprintf('%.4f ', beta_hidden); 
        fprintf(', beta_out = %.4f\n', beta_out);
        
        %% --- Weight updates ---
        dW_hidden_output = eta * beta_out * h_out';  % N_output x N_hidden
        dW_input_hidden = eta * beta_hidden * x';    % N_hidden x N_input
        
        % Update weights immediately (incremental)
        Weights_hidden_output = Weights_hidden_output + dW_hidden_output;
        Weights_input_hidden = Weights_input_hidden + dW_input_hidden;
        
        fprintf('Weight updates:\n');
        fprintf('dW_input_hidden =\n'); fprintf('%.4f ', dW_input_hidden); fprintf('\n');
        fprintf('dW_hidden_output =\n'); fprintf('%.4f ', dW_hidden_output); fprintf('\n');
        fprintf('Updated weights:\n');
        fprintf('Weights_input_hidden =\n'); fprintf('%.4f ', Weights_input_hidden); fprintf('\n');
        fprintf('Weights_hidden_output =\n'); fprintf('%.4f ', Weights_hidden_output); fprintf('\n');
    end
end
