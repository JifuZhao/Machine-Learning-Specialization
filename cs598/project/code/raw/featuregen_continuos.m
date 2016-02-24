function [ features ] = featuregen_continuos( filename_acc,filename_gyro,cuttingEdge )
%FEATUREGEN_CONTINUOS Summary of this function goes here
% filename_acc: full path of accelerometer csv file
% filename_gyro: full path of gyro csv file (In this version, this is a
% dummy, you can put anything in
% cuttingEdge: number of ending sample you want to cut off
% assume that the first 500 samples are ignored.
% part of CSP 598 fall 2015 project code 

    c=  csvread(filename_acc,500,0);
    acc_y = c(1:end,3);
    acc_z = c(1:end,4);
    acc_x = c(1:end,2);
    L = length(acc_x);
    threshold1 = 3;
    threshold2 = 1;
    z= sqrt(acc_y.^2+acc_z.^2);
    windows_size = 150;
    features=[];
    isPossible = 0;
    for i = 100:L-cuttingEdge
        if isPossible ==1
            if acc_z(i)<threshold1-0.5 % hand going out of watching time position
                isPossible =0; 
            end
            continue
        end
        if (acc_z(i)>threshold1&& acc_y(i)>threshold2)
            if (z(i)>9.0)
                fet = zeros(4,1);
                first_ind=i;
                second_ind=i+windows_size;
                
                fet(1) = max(acc_z(first_ind:second_ind))- min(acc_z(first_ind:second_ind));
                fet(2) = sum(acc_z(first_ind:second_ind))/(windows_size+1);
                fet(3) = sum(acc_y(first_ind:second_ind))/(windows_size+1);
                fet(4) = sum(acc_x(first_ind:second_ind))/(windows_size+1);
                
                features = [features fet];
                isPossible =1;
            end
        end
    end
    if isempty(features)
        features = 25*ones(4,1);
    end
end

