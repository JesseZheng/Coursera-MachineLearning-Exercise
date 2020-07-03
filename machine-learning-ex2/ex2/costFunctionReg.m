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

prediction = sigmoid(X*theta);

cost_1 = -log(prediction);
cost_0 = -log(1-prediction);

thetaZore = theta;
thetaZore(1) = 0;

regularized_cost = sum(thetaZore .^ 2);
regularized_grad = thetaZore;


J = (1/m) * sum(y'*cost_1 + (1-y)'*cost_0) + lambda/(2*m) * regularized_cost;

grad = (1/m) * X' * (prediction-y) + lambda/m * regularized_grad;


% =============================================================

end
