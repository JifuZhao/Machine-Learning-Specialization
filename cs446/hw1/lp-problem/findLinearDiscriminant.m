% This function finds a linear discriminant using LP
% The linear discriminant is represented by 
% the weight vector w and the threshold theta.
% YOU NEED TO FINISH IMPLEMENTATION OF THIS FUNCTION.

function [w,theta,delta] = findLinearDiscriminant(data)
%% setup linear program
[m, np1] = size(data);
n = np1-1;

% write your code here

A = zeros(m+1, n+2);
for j = 1:m
    A(j, 1:n) = data(j, n+1).*data(j, 1:n);
    A(j, n+1) = data(j, n+1);
    A(j, n+2) = 1;
end
A(m+1, n+2) = 1;

b = zeros(m+1, 1);
b(1:m, 1) = 1;

c = zeros(n+2, 1);
c(n+2, 1) = 1;


%% solve the linear program
%adjust for matlab input: A*x <= b
[t, z] = linprog(c, -A, -b);

%% obtain w,theta,delta from t vector
w = t(1:n);
theta = t(n+1);
delta = t(n+2);

end
