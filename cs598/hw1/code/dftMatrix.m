%This function is going to generate a Gabor filter

function [dft] = dftMatrix(size)
    dft = zeros(size);

    for j = 1:size
        for k = 1:size
            dft(j, k) = exp(-i*(j-1)*(k-1)*2*pi/size)/sqrt(size);
        end  
    end
    
end
