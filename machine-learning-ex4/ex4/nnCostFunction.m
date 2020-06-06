function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%% PART ONE
% adding ones to the first column of X (for bias), input layer
a_1 = [ones(m, 1), X];

% hidden layer
z_2 = a_1 * Theta1';

% inputs for the output layer
% adding ones to the first column of the hidden layer for bias
a_2 = sigmoid(z_2);
a_2 = [ones(size(a_2, 1), 1), a_2];

% output layer
z_3 = a_2 * Theta2';
h = sigmoid(z_3);

% vectorizing y based on the number of labels
y_vector = (1:num_labels) == y;

% calculating the cost function
J = (1/m) * sum(sum((-y_vector .* log(h)) - ((1 - y_vector) .* log(1 - h))));

%% PART TWO

% adding ones to the first column of X for bias
b_a1 = [ones(m, 1), X];

% hidden layer 
b_z2 = b_a1 * Theta1';
b_a2 = sigmoid(b_z2);
b_a2 = [ones(size(b_a2, 1), 1), b_a2];

b_z3 = b_a2 * Theta2';
b_a3 = sigmoid(b_z3);

b_Yvector = (1:num_labels) == y;

% calculate 
% output layer
delta_vector_3 = b_a3 - b_Yvector;

% hidden layer l=2
delta_vector_2 = (delta_vector_3 * Theta2) .* [ones(size(b_z2, 1), 1) sigmoidGradient(b_z2)];
delta_vector_2 = delta_vector_2(:,2:end);

Theta1_grad = (1/m) * (delta_vector_2' * b_a1);
Theta2_grad = (1/m) * (delta_vector_3' * b_a2);
 
%% PART THREE

reg = (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
J = J + reg;

Theta1_reg = (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_reg = (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

Theta1_grad = Theta1_grad + Theta1_reg;
Theta2_grad = Theta2_grad + Theta2_reg;











% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
