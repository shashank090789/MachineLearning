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


% Add ones to the X data matrix
X = [ones(m, 1) X];

sum_h=0;
for i=1:m

l1_pre=Theta1*(X(i,:))';
add_one=[ones(1,1); l1_pre];
l1=sigmoid(l1_pre);
z1=[ones(1,1); l1];
z2=Theta2*z1;
l2=sigmoid(z2);
y_vec=zeros(num_labels,1);
y_vec(y(i))=1;
B=y_vec.*log(l2);
C=(1-y_vec).*log(1-l2);
D=B.+C;
sum_h=sum_h+ones(1,num_labels)*D;

%code for computing gradient
delta_output=l2-y_vec;
mul=(Theta2'*delta_output).*sigmoidGradient(add_one);
delta_hidden=mul(2:end);
theta_2_grad_single=delta_output*z1';
Theta2_grad=(Theta2_grad+theta_2_grad_single);
theta1_grad_single=delta_hidden*X(i,:);
Theta1_grad=Theta1_grad+theta1_grad_single;
end
theta_1_removed_1st_column=Theta1(:,2:end);
Theta1_grad_reg=[zeros(size(Theta1,1),1) theta_1_removed_1st_column];

Theta2_grad_reg=[zeros(size(Theta2,1),1) Theta2(:,2:end)];
Theta1_grad=((1/m).*Theta1_grad)+((lambda/m).*Theta1_grad_reg);
Theta2_grad=((1/m).*Theta2_grad)+((lambda/m).*Theta2_grad_reg);
theta_1_sum=sum(sum(Theta1(:,2:end).^2));
theta_2_sum=sum(sum(Theta2(:,2:end).^2));
reg=(lambda/(2*m))*(theta_1_sum+theta_2_sum);
J=((-1/m)*sum_h)+reg;














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
