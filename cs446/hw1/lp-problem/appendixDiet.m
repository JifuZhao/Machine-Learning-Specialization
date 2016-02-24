% This script solves the example linear program (eggs, pasta, and yogurt)
% This might help you write findLinearDiscriminant function.

c = [0.1; 0.05; 0.25];
A = [3 1 2; 1 3 2; -2 0 -1];
b = [7; 9; -4];
lowerBound = zeros(3, 1);
[t, z] = linprog(c, -A, -b, [], [], lowerBound) % no semi-colon, so we see output
