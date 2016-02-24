% Lecture 3, Slide 39

clear all;

% Read the original data from Source/39.wav
[y, Fs] = audioread('39.wav');
%sound(y);

L = length(y);
t = (0:L-1)./Fs;
N = 50;
interval = N/2;
f = (0:N/2) * Fs / N;
xx = (0:interval:L-1)/Fs;
yy = f;

% Generate the filter
low = fir1(49, 0.4);
high = fir1(50, 0.4, 'high');
band = fir1(50, [0.3, 0.5]);
stop = fir1(50, [0.2, 0.6], 'stop');

% Convolution with the signal
Out1 = conv(low, y);
Out2 = conv(high, y);
Out3 = conv(band, y);
Out4 = conv(stop, y);

% Using self-defined function spectrum to get the spectrum
Y = spectrum(N, L, y).^0.6;
Y1 = spectrum(N, L, Out1).^0.6;
Y2 = spectrum(N, L, Out2).^0.6;
Y3 = spectrum(N, L, Out3).^0.6;
Y4 = spectrum(N, L, Out4).^0.6;

figure
subplot(3, 2, [1,2])
surf(xx, yy, Y(1:(N/2+1), 1:interval:L), 'EdgeColor', 'none');
view(0, 90);
xlim([0, 2.8])
ylim([0, 20000]);
title('Original input');
xlabel('Time(sec)');
ylabel('Frequency(Hz)');

subplot(3, 2, 3)
surf(xx, yy, Y1(1:(N/2+1), 1:interval:L), 'EdgeColor', 'none');
view(0, 90);
xlim([0, 2.8])
ylim([0, 20000]);
title('Lowpass');
xlabel('Time(sec)');
ylabel('Frequency(Hz)');

subplot(3, 2, 4)
surf(xx, yy, Y2(1:(N/2+1), 1:interval:L), 'EdgeColor', 'none');
view(0, 90);
xlim([0, 2.8])
ylim([0, 20000]);
title('Highpass');
xlabel('Time(sec)');
ylabel('Frequency(Hz)');

subplot(3, 2, 5)
surf(xx, yy, Y3(1:(N/2+1), 1:interval:L), 'EdgeColor', 'none');
view(0, 90);
xlim([0, 2.8])
ylim([0, 20000]);
title('Bandpass');
xlabel('Time(sec)');
ylabel('Frequency(Hz)');

subplot(3, 2, 6)
surf(xx, yy, Y4(1:(N/2+1), 1:interval:L), 'EdgeColor', 'none');
view(0, 90);
xlim([0, 2.8])
ylim([0, 20000]);
title('Band-reject');
xlabel('Time(sec)');
ylabel('Frequency(Hz)');

colormap(flipud(bone));
saveas(gcf, '39Self.png')