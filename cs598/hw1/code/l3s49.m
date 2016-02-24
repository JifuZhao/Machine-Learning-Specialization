% Lecture 3, Slide 49

clear all;

% Read the original data from Source/39.wav
[y, Fs] = audioread('49.mp3');
%sound(y);

M = 8;
L = length(y);
t = (0:L-1)./Fs;
N = 200;
interval = N/2;
f = (0:N/2) * Fs / N;
xx = (0:interval:L-1)/Fs;
yy = f;

% Downsampling signal
N2 = 200;
y2 = y(1:M:L);
L2 = length(y2);
f2 = (0:N2/2) * Fs / (M*N2);
xx2 = (0:interval:L2-1)/(Fs/M);
yy2 = f2;

% Generate the filter
low = fir1(49, 1/M);

% Convolution with the signal
N3 = N2;
M = 8;
Out3 = conv(y, low, 'same');
y3 = Out3(1:M:L); 
L3 = length(y3);
f3 = (0:N3/2) * Fs / (M*N3);
xx3 = (0:interval:L3-1)/(Fs/M);
yy3 = f3;

% Using self-defined function spectrum to get the spectrum
Y1 = spectrum(N, L, y).^0.6;
Y2 = spectrum(N2, L2, y2).^0.6;
Y3 = spectrum(N3, L3, y3).^0.6;

figure
subplot(3, 1, 1)
surf(xx, yy, Y1(1:(N/2+1), 1:interval:L), 'EdgeColor', 'none');
view(0, 90);
xlim([0, 5.8])
ylim([0, 11000]);
title('Original input');
xlabel('Time(sec)');
ylabel('Frequency(Hz)');

subplot(3, 1, 2)
surf(xx2, yy2, Y2(1:(N2/2+1), 1:interval:L2), 'EdgeColor', 'none');
view(0, 90);
xlim([0, 5.8])
ylim([0, 2800]);
title('Decimated');
xlabel('Time(sec)');
ylabel('Frequency(Hz)');

subplot(3, 1, 3)
surf(xx3, yy3, Y3(1:(N3/2+1), 1:interval:L3), 'EdgeColor', 'none');
view(0, 90);
xlim([0, 5.8])
ylim([0, 2800]);
title('Filtered & Decimated');
xlabel('Time(sec)');
ylabel('Frequency(Hz)');

colormap(flipud(bone));
saveas(gcf, '49Self.png')