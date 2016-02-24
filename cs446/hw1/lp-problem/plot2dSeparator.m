% This function plots the linear discriminant.
% YOU NEED TO IMPLEMENT THIS FUNCTION

function plot2dSeparator(w, theta)

n = length(w);
if n ~= 2
    disp('only 2d data supported.')
else
    x = -0.1:0.01:1.1;
    y = -w(1) * x / w(2) - theta / w(2);
    plot(x, y, 'LineWidth',2);
    
    disp(w);
end
