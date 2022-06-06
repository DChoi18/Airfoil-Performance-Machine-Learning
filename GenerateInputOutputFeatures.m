function IO_Feat = GenerateInputOutputFeatures(pdata,geom,mode)
% Generate block of data for use in Python for neural network training
%
% Inputs:
%        pdata = sorted airfoil performance polars
%        geom = airfoil geometry struct
% Outputs:
%        IO_Feat = array of data with input and output features for neural
%                  network training
% Author: Derrick Choi

switch mode
    case 'pressure'
        % Preallocate
        IO_Feat = zeros(size(pdata{1,1},1)*100,11);

        IO_Feat(:,2) = (pdata{1,2});% Re
        IO_Feat(:,3) = (pdata{1,3});% Ncrit
        % airfoil geom
        IO_Feat(:,4) = geom.t_max;
        IO_Feat(:,5) = geom.max_camb;
        IO_Feat(:,6) = geom.pos_t;
        IO_Feat(:,7) = geom.pos_c;

        for i =  1:length(pdata{1,1}.alpha)
            % Assign other inputs
            IO_Feat(1+(i-1)*100:100*i,1) = pdata{1,1}.alpha(i); % aoa
            IO_Feat(1+(i-1)*100:100*i,8) = geom.t;
            IO_Feat(1+(i-1)*100:100*i,9) = geom.mean_camber;

            % Assign outputs
            IO_Feat(1+(i-1)*100:100*i,10) = pdata{1,1}.Cp_ps{i};
            IO_Feat(1+(i-1)*100:100*i,11) = pdata{1,1}.Cp_ss{i};
        end
    case 'polars'
        IO_Feat = zeros(size(pdata{1,1},1)*100,13);

        IO_Feat(:,2) = (pdata{1,2});% Re
        IO_Feat(:,3) = (pdata{1,3});% Ncrit
        % airfoil geom
        IO_Feat(:,4) = geom.t_max;
        IO_Feat(:,5) = geom.max_camb;
        IO_Feat(:,6) = geom.pos_t;
        IO_Feat(:,7) = geom.pos_c;

        for i =  1:length(pdata{1,1}.alpha)
            % Assign other inputs
            IO_Feat(1+(i-1)*100:100*i,1) = pdata{1,1}.alpha(i); % aoa
            IO_Feat(1+(i-1)*100:100*i,8) = geom.t;
            IO_Feat(1+(i-1)*100:100*i,9) = geom.mean_camber;

            % Assign outputs
            IO_Feat(1+(i-1)*100:100*i,10) = pdata{1,1}.Cd(i);
            IO_Feat(1+(i-1)*100:100*i,11) = pdata{1,1}.Cdp(i);
            IO_Feat(1+(i-1)*100:100*i,12) = pdata{1,1}.Cl(i);
            IO_Feat(1+(i-1)*100:100*i,13) = pdata{1,1}.Cm(i);
        end
        
end

end