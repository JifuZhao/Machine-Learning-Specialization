% This function solves the LP problem for a given weight vector
% to find the threshold theta.
% YOU NEED TO FINISH IMPLEMENTATION OF THIS FUNCTION.

function [theta,delta] = findLinearThreshold(data,w)
%% setup linear program
[m, np1] = size(data);
n = np1 - 1;

% write your code here

A = zeros(m+1 ,n);
A(1:m, 1) = data(1:m, np1);
A(m+1,1) = 0;
A(:, 2) = 1;

b = zeros(m+1, 1);
b(1:m, 1) = data(1:m, 1:n) * w;
for i = 1:m
    b(i) = 1 - data(i, np1) * b(i);
end
b(m+1, 1) = 0;

c = [0; 1];

%% solve the linear program
%adjust for matlab input: A*x <= b

[t, z] = linprog(c, -A, -b);

%% obtain w,theta,delta from t vector

theta = t(1);
delta = t(2);

end
