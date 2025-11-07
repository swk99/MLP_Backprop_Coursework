%% MLP Incremental Backpropagation (2-3-1 network)
clc; clear;

%% Training data
X = [0 1; 1 0];   % x1,x2
T = [1; 1];       % target output

%% Initial weights (W1..W9)
% Input-to-hidden weights: W1-W6
W = [-0.1 0.1 0.2;  % hidden1 weights from x1,x2
     0.2 -0.1 -0.1; % hidden2 weights from x1,x2
     -0.2 0.2 0.2]; % hidden3 weights from x1,x2
% For clarity, we will flatten to 9 weights:
W_vec = [-0.1; 0.1; 0.2; 0.2; -0.1; -0.1; -0.2; 0.2; 0.2];

% Map flattened vector to connections:
% W1,W2 -> hidden1 (x1,x2)
% W3,W4 -> hidden2 (x1,x2)
% W5,W6 -> hidden3 (x1,x2)
% W7,W8,W9 -> output (from hidden1, hidden2, hidden3)

%% Learning parameters
eta = 0.2;       % learning rate
n_epochs = 1;    % only one pass for demonstration
n_samples = size(X,1);

%% Sigmoid function and derivative
sig = @(x) 1./(1+exp(-x));
sig_deriv = @(y) y .* (1-y);  % input y is output of sigmoid

%% Training loop (incremental)
for epoch = 1:n_epochs
    fprintf('Epoch %d\n', epoch);
    for p = 1:n_samples
        x1 = X(p,1);
        x2 = X(p,2);
        target = T(p);
        
        %% ===== Forward Pass =====
        % Hidden layer outputs
        h1_in = W_vec(1)*x1 + W_vec(2)*x2;
        h2_in = W_vec(3)*x1 + W_vec(4)*x2;
        h3_in = W_vec(5)*x1 + W_vec(6)*x2;
        
        h1 = sig(h1_in);
        h2 = sig(h2_in);
        h3 = sig(h3_in);
        
        % Output layer (summation of hidden units, no bias)
        y_in = W_vec(7)*h1 + W_vec(8)*h2 + W_vec(9)*h3;
        y = y_in; % linear output
        
        fprintf('\nTraining example %d:\n', p);
        fprintf('Forward pass outputs: h1=%.4f, h2=%.4f, h3=%.4f, y=%.4f\n', h1,h2,h3,y);
        
        %% ===== Backward Pass =====
        % Output error
        beta_out = target - y;
        
        % Hidden errors
        beta_h1 = sig_deriv(h1) * (beta_out * W_vec(7));
        beta_h2 = sig_deriv(h2) * (beta_out * W_vec(8));
        beta_h3 = sig_deriv(h3) * (beta_out * W_vec(9));
        
        fprintf('Backward pass betas: beta_h1=%.4f, beta_h2=%.4f, beta_h3=%.4f, beta_out=%.4f\n', ...
            beta_h1, beta_h2, beta_h3, beta_out);
        
        %% ===== Weight updates =====
        % Hidden-to-output weights
        dW7 = eta * beta_out * h1;
        dW8 = eta * beta_out * h2;
        dW9 = eta * beta_out * h3;
        
        % Input-to-hidden weights
        dW1 = eta * beta_h1 * x1;
        dW2 = eta * beta_h1 * x2;
        dW3 = eta * beta_h2 * x1;
        dW4 = eta * beta_h2 * x2;
        dW5 = eta * beta_h3 * x1;
        dW6 = eta * beta_h3 * x2;
        
        fprintf('Weight changes:\n');
        fprintf('dW1=%.4f, dW2=%.4f, dW3=%.4f, dW4=%.4f, dW5=%.4f, dW6=%.4f, dW7=%.4f, dW8=%.4f, dW9=%.4f\n', ...
            dW1,dW2,dW3,dW4,dW5,dW6,dW7,dW8,dW9);
        
        % Update weights
        W_vec = W_vec + [dW1; dW2; dW3; dW4; dW5; dW6; dW7; dW8; dW9];
        
        fprintf('Updated weights:\n');
        fprintf('W1=%.4f, W2=%.4f, W3=%.4f, W4=%.4f, W5=%.4f, W6=%.4f, W7=%.4f, W8=%.4f, W9=%.4f\n', ...
            W_vec(1),W_vec(2),W_vec(3),W_vec(4),W_vec(5),W_vec(6),W_vec(7),W_vec(8),W_vec(9));
    end
end
