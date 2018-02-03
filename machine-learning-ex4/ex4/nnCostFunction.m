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

X = [ones(m, 1) X];
a_2 = sigmoid(X * Theta1');
a_2 = [ones(m, 1) a_2];
a_3 = sigmoid(a_2 * Theta2');

% Y_eye = eye(num_labels);
% Y = Y_eye(y,:);

I = eye(num_labels);
Y = zeros(m, num_labels);
for i = 1 : m
  Y(i, :) = I(y(i), :);
end

cost = sum((-Y .* log(a_3)) - ((1 - Y) .* log(1 - a_3)), 2);
J = (1 / m) * sum(cost);
% y_temp = zeros(num_labels,1);
% one_col = ones(num_labels,1);
% cost = zeros(m,1);
% for i=1:m,
%     for k=1:num_labels,
%         y_temp = zeros(num_labels,1);
%         y_temp(k)=1;
%         
% %         J = J + 1/m * (-log(a_3(i,k)) - sum((ones(num_labels,1)-y_temp).*log(1-a_3(i,k))));
%         cost(i) = cost(i) + sum(-y_temp.*log(a_3(i,k)) - (one_col-y_temp).*log(one_col-a_3(i,:)'));
%     end
% end
% J = (1 / m) * sum(cost);

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
[~,n1] = size(Theta1);
[~,n2] = size(Theta2);
J = J + lambda/(2*m) * (sumsqr(Theta1(:,2:n1)) + sumsqr(Theta2(:,2:n2)));

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
Delta1=0;
Delta2=0;

for t = 1:m,
    % part 1
    aa_1 = X(t,:); % 1x401
    
    zz_2 = aa_1 * Theta1'; % 1x25
    aa_2 = [1, sigmoid(zz_2)]; % 1x26
    
    zz_3 = aa_2 * Theta2'; % 1x10
    aa_3 = sigmoid(zz_3); % 1x10
    
    % part 2
    del_3 = aa_3 - Y(t,:); % 1x10
    
    % part 3
    del_2 = del_3*Theta2(:,2:n2).*sigmoidGradient(zz_2); % 1x25
    
    % part 4
    Delta2 = Delta2 + (del_3' * aa_2);
	Delta1 = Delta1 + (del_2' * aa_1);
end

Theta1_grad = 1/m * Delta1 + lambda/m*Theta1;
Theta1_grad(:,1) = Theta1_grad(:,1) - lambda/m*Theta1(:,1);

Theta2_grad = 1/m * Delta2 + lambda/m*Theta2;
Theta2_grad(:,1) = Theta2_grad(:,1) - lambda/m*Theta2(:,1);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
