function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z=X*theta;
A=1+(e.^(-z));
h=1./A;
B=-y'*log(h);
C=(1-y)'*log(1-h);
D=B-C;
ones_m=ones(1,(length(theta)-1));
theta_square=theta(2:end,:).^2;
sum=ones_m*theta_square
J=((1/m)*D)+(lambda/(2*m))*sum;
sub=h-y;
grad_without_reg=(1/m)*(sub'*X);
reg=(lambda/m)*[zeros(1);theta(2:end,:)];

grad=grad_without_reg'+reg;

% =============================================================

end