% Lecture 3, Slide 18

clear all;

% Read original data Source/18.wav
[y, Fs] = audioread('18.wav');
%sound(y);

L = length(y);
t = (0:L-1)./Fs;
N0 = [64, 512, 2048];

figure;
for i = 1:3
    N = N0(i); 
    interval = N/2;
    Y = spectrum(N, L, y).^0.45;
    f = (0:N/2) * Fs / N;

    subplot(3, 1, i);    
    xx = (0:interval:(L-1))/Fs;
    yy = f;

    surf(xx, yy, Y(1:(N/2+1), 1:interval:L), 'EdgeColor', 'none');
    view(0, 90);
    xlim([0, 2.6]);
    ylim([0, 5500]);
    if i == 1
        title('Small window, N = 64');
    else
        if i == 2
            title('Medium window, N = 512');
        else
            title('Large window, N = 2048')
            xlabel('Time(sec)');
        end
    end            
    ylabel('Frequency (Hz)');
end

colormap(flipud(bone))
saveas(gcf, '18Self.png')