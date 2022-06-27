% Purpose: Read json files containing airfoil data and parse through them
%
% Author: Derrick Choi
%
% Things left to do:
% sort data
% create input output data for training
% write training neural net code
% write evaluation code
% write post processing code to assess accuracy
% run
clc; close all; clear
%% Get files

% Get directory
directory = [pwd '\json\'];
% Get filenames
allfiles_json = dir("json\*.json");
% allfiles_json.name = 'naca2421-il - NACA 2421.json';

% Test code on sample dataset
% str = fileread([directory allfiles_json(1).name]);
outDirectory = [pwd '\h5\'];
OutFile_Name = [outDirectory 'All_Airfoils_Polars.h5'];

% Modes of operation
% polars - training model for polars
% pressure - training model for pressure coefficient
mode = 'polars';

switch mode
    case 'pressure'
        %% Construct learning dataset
        IO_Feat = [];
        for i = 1:length(allfiles_json)
            fprintf(['Reading airfoil: ',allfiles_json(i).name,'\n\n'])
            str = fileread([directory allfiles_json(i).name]);
            data = jsondecode(str);
            pdata = struct2table(data.polars);

            fprintf('Sorting data by Re\n\n')
            pdata = sortbyRe(pdata);
            fprintf('Sorting data by Ncrit\n\n')
            pdata = sortbyNcrit(pdata);

            fprintf('Computing Airfoil Geometry Characteristics\n\n')

            airfoil_geom = ComputeAirfoilGeomStats(data.xps,data.yps,data.yss);

            for j = 1:size(pdata,1)
                fprintf('Generating Input and Output Features!\n\n')
                temp = GenerateInputOutputFeatures(data,pdata(j,:),airfoil_geom,mode);
                IO_Feat = [IO_Feat; temp];
            end
        end
        % f1 = figure('Name','Cd vs alpha');
        % plot(pdata{2,1}.alpha,pdata{2,1}.Cd,'LineWidth',1.5)
        % labelplot(f1,'$\alpha$','$c_d$','Drag Coefficient vs AOA',0)
        %% Rewrite sorted into hdf5 file format
        fprintf('Writing Output File!\n')
        %OutFile_Name = strcat([outDirectory data.name '_Re'],pdata{i,2},'_Ncrit',pdata{i,3});

        h5create(OutFile_Name,'/Re',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/Ncrit',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/thickness',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/camber',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/max_thick',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/max_camb',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/pos_max_camb',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/xu_coord',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/xl_coord',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/yu_coord',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/yl_coord',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/pos_max_t',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/Cp_ps',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/Cp_ss',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/alpha',size(IO_Feat(:,1),1))

        h5write(OutFile_Name,'/Re',IO_Feat(:,2))
        h5write(OutFile_Name,'/Ncrit',IO_Feat(:,3))
        h5write(OutFile_Name,'/thickness',IO_Feat(:,8))
        h5write(OutFile_Name,'/camber',IO_Feat(:,9))
        h5write(OutFile_Name,'/max_thick',IO_Feat(:,4))
        h5write(OutFile_Name,'/max_camb',IO_Feat(:,5))
        h5write(OutFile_Name,'/pos_max_camb',IO_Feat(:,7))
        h5write(OutFile_Name,'/pos_max_t',IO_Feat(:,6))
        h5write(OutFile_Name,'/Cp_ps',IO_Feat(:,14))
        h5write(OutFile_Name,'/Cp_ss',IO_Feat(:,15))
        h5write(OutFile_Name,'/alpha',IO_Feat(:,1))
        h5write(OutFile_Name,'/xu_coord',IO_Feat(:,10))
        h5write(OutFile_Name,'/xl_coord',IO_Feat(:,11))
        h5write(OutFile_Name,'/yu_coord',IO_Feat(:,12))
        h5write(OutFile_Name,'/yl_coord',IO_Feat(:,13))
    case 'polars'
        %% Construct learning dataset
        IO_Feat = [];
        for i = 1:length(allfiles_json)
            fprintf(['Reading airfoil: ',allfiles_json(i).name,'\n\n'])
            str = fileread([directory allfiles_json(i).name]);
            data = jsondecode(str);
            data.name
            pdata = struct2table(data.polars);

            fprintf('Sorting data by Re\n\n')
            pdata = sortbyRe(pdata);
            fprintf('Sorting data by Ncrit\n\n')
            pdata = sortbyNcrit(pdata);

            fprintf('Computing Airfoil Geometry Characteristics\n\n')
            airfoil_geom = ComputeAirfoilGeomStats(data.xps,data.yps,data.yss);

            for j = 1:size(pdata,1)
                fprintf('Generating Input and Output Features!\n\n')
                temp = GenerateInputOutputFeatures(data,pdata(j,:),airfoil_geom,mode);
                IO_Feat = [IO_Feat; temp];
            end
        end
        % f1 = figure('Name','Cd vs alpha');
        % plot(pdata{2,1}.alpha,pdata{2,1}.Cd,'LineWidth',1.5)
        % labelplot(f1,'$\alpha$','$c_d$','Drag Coefficient vs AOA',0)
        %% Rewrite sorted into hdf5 file format
        fprintf('Writing Output File!\n')
        %OutFile_Name = strcat([outDirectory data.name '_Re'],pdata{i,2},'_Ncrit',pdata{i,3});

        h5create(OutFile_Name,'/Re',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/Ncrit',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/thickness',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/camber',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/max_thick',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/max_camb',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/pos_max_camb',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/pos_max_t',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/Cd',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/Cdp',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/Cl',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/Cm',size(IO_Feat(:,1),1))
        h5create(OutFile_Name,'/alpha',size(IO_Feat(:,1),1))

        h5write(OutFile_Name,'/Re',IO_Feat(:,2))
        h5write(OutFile_Name,'/Ncrit',IO_Feat(:,3))
        h5write(OutFile_Name,'/thickness',IO_Feat(:,8))
        h5write(OutFile_Name,'/camber',IO_Feat(:,9))
        h5write(OutFile_Name,'/max_thick',IO_Feat(:,4))
        h5write(OutFile_Name,'/max_camb',IO_Feat(:,5))
        h5write(OutFile_Name,'/pos_max_camb',IO_Feat(:,7))
        h5write(OutFile_Name,'/pos_max_t',IO_Feat(:,6))
        h5write(OutFile_Name,'/Cd',IO_Feat(:,10))
        h5write(OutFile_Name,'/Cdp',IO_Feat(:,11))
        h5write(OutFile_Name,'/Cl',IO_Feat(:,12))
        h5write(OutFile_Name,'/Cm',IO_Feat(:,13))
        h5write(OutFile_Name,'/alpha',IO_Feat(:,1))
end

