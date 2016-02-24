%This function is going to generate a hann window

function [h] = hannWindow(L)

    h = zeros(L, 1);
    N = L-1;

    for i = 0:N
        h(i+1) = 0.5*(1 - cos(2*pi*i/N));
    end

end

