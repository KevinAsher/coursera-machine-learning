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


X_biased = [ones(m,1), X]; % 5000x401
accum = 0;


a1 = X_biased'; % 401x5000
% Theta1 - 25x401
z2 = Theta1*a1; % 25x5000
a2 = sigmoid(z2); % 25x5000
a2 = [ones(1, size(a2, 2)); a2]; % 26x5000
% Theta2 - 10x26
h_theta = sigmoid(Theta2*a2); % 10x5000 

% y_vec = zeros(num_labels,1); % it was 10x1
% y_vec(y(i)) = 1;

Y = zeros(num_labels,m); % 10x5000
for i=1:m
  Y(y(i),i) = 1;
end

accum = sum((-Y.*log(h_theta) - (1-Y).*log(1-h_theta))(:));
  

Theta1_nobias = Theta1(:, 2:end); % 25x400
Theta2_nobias = Theta2(:, 2:end); % 10x25
J = 1/m * accum + lambda/(2*m) * (  ...
  sum(Theta1_nobias(:).^2) + sum(Theta2_nobias(:).^2));
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


  % Step 2: Calculate delta 3
  delta_3 = h_theta - Y; % 10x5000
  
  % Step 3: Calculate delta 2
  delta_2 = Theta2_nobias' * delta_3 .* sigmoidGradient(z2);
  
  % Step 4: Accumulate gradient
  Theta1_grad = Theta1_grad + delta_2*a1';
  Theta2_grad = Theta2_grad + delta_3*a2';
  

% Step 5: Obtain unregulized gradient
Theta1_grad = 1/m * Theta1_grad;
Theta2_grad = 1/m * Theta2_grad;
  


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Theta1_grad(:,2:end) = (Theta1_grad + lambda/m * Theta1)(:,2:end);
Theta2_grad(:,2:end) = (Theta2_grad + lambda/m * Theta2)(:,2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
