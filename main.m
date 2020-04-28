%% Initialization
clear ; close all; clc

% import training data

X = loadMNISTImages("train-images.idx3-ubyte")';
y = loadMNISTLabels("train-labels.idx1-ubyte");

% change label for 0 to 10
y(y == 0) = 10;

% disolay 100 random images
sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel, :));


% network architecture
input_layer_size  = size(X,2);  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

fprintf('\nInput layer size = %f\n', input_layer_size);
fprintf('\nHidden layer size = %f\n',hidden_layer_size);
fprintf('\nTraining data loaded from: train-images.idx3-ubyte and train-labels.idx1-ubyte \n');


%% ================ Part 6: Initializing Pameters ================

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];




%% =================== Part 8: Training NN ===================
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 300);

%  You should also try different values of lambda
lambda = 0.1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

%==================================================== 

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

save own_params.mat
