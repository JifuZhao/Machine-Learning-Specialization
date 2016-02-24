classdef DataObj < handle %( handle class)
    %DATAIOBJECT class represent an instance of datum 
    %Information of the instance including 
    % label: true watching
    % feature: precomputed feature
    %
    %notes: 
    
    % developed by Duc Phan ver 1.0 Nov 20th 2015
    properties (SetAccess = public)
       label
       windows_size
       feature
       fd %file descriptor 
    end
    
    methods
        function obj = DataObj(label,filename_gyro,filename_acc,cuttingEdge)
            obj.label = label;
            obj.windows_size = 150;
            obj.fd = {filename_gyro,filename_acc};
            % todo 
           % c = csvread(filename_gyro,600,0);
             c=  csvread(filename_acc,500,0);
            acc_y = c(1:end-cuttingEdge,3);
            acc_z = c(1:end-cuttingEdge,4);
            acc_x = c(1:end-cuttingEdge,2);
            % calculate features
            threshold1 = 3;
            threshold2 = 1;
            z= sqrt(acc_y.^2+acc_z.^2);
            n = find(acc_z>threshold1&acc_y>threshold2&z>9.0,1);
            fet = 25*ones(4,1);
            if ~isempty(n)
                first_ind=n;
                second_ind=n+obj.windows_size;
                if second_ind>length(acc_y);
                     second_ind=length(acc_y);
                end
                fet(1) = max(acc_z(first_ind:second_ind))- min(acc_z(first_ind:second_ind));
                fet(2) = sum(acc_z(first_ind:second_ind))/(obj.windows_size+1);
                fet(3) = sum(acc_y(first_ind:second_ind))/(obj.windows_size+1);
                fet(4) = sum(acc_x(first_ind:second_ind))/(obj.windows_size+1);
            end
            obj.feature =fet;
                                           
        end 
    end 
    
end
