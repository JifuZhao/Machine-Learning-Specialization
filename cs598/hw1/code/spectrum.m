%This function is going to generate a spectrum

function [Y] = spectrum(N, L, y)
    
    Y = zeros(N, L);
    
    for j = 1:L
        Max = min(j+N-1, L-1);
        if rem(j-1, N/2) == 0
            h = hannWindow(Max + 2 - j);
            s = y(j:Max+1).* h;
            Y(:, j) = abs(fft(s, N)/N);
            Y(2:(N/2), j) = 2 * Y(2:(N/2), j);
        else
            Y(:, j) = Y(:, floor(j/(N/2))*N/2+1);
        end
    end
    
end

