% Perception analysis function

function [w, theta, N, R] = Perceptron(iternation, x, y, threshold)
    [rows, cols] = size(x);
    w = zeros(cols, 1);
    theta = 0;
    eta = 1.0;
    N = 0; %
    R = 0; %
    for i = 1:iternation
        if R == threshold  %
            break     %
        end           %
        for j = 1:rows
            if R == threshold %
                break; %
            end        %
            if y(j).*(x(j, :)*w + theta) <= 0
                R = 0;  %
                N = N + 1;
                w = w + eta.*y(j)*x(j, :)';
                theta = theta + eta.*y(j);
            else
                R = R + 1;  %
            end

        end
    end

end