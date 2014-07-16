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

n = length(theta);
tempSum = 0;
for j = 2 : n
    tempSum = tempSum + theta(j)^2;
end;
tempSum2 = 0;
for i = 1 : m
    tempSum2 = tempSum2 - y(i) * log(sigmoid(X(i, :) * theta)) - (1 - y(i)) * log(1 - sigmoid(X(i, :)  * theta)) + lambda / (2 * m) * tempSum;
end;
J = 1 / m * tempSum2;
temp = 0;
for j = 1 : n
    temp = 1 / m * X(:, j)' * (sigmoid(X * theta) - y);
    if (j == 1)
        grad(j) = temp;
    else
        grad(j) = temp + lambda / m * theta(j);
end;

% =============================================================

end
