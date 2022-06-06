% Script to post process results
% Author: Derrick Choi
clc;clear;close all
%% Get Airfoil Coordinates
jsonfile = 'naca0010-il - NACA 0010.json';
str = fileread([pwd '\json\' jsonfile]);
data = jsondecode(str);
pdata = struct2table(data.polars);
pdata = sortbyRe(pdata);
pdata = sortbyNcrit(pdata);

x_coord = data.xps;
c = max(x_coord)-min(x_coord);
yu_coord = data.yps;
yl_coord = data.yss;

% angles of attack, Re, and Ncrit
AOA = [-10.25:0.25:10.25];
Re = 100000;
Ncrit = 5;

% average values from NN prediction
Cd_avg = zeros(length(AOA),1);
Cdp_avg = zeros(length(AOA),1);
Cl_avg = zeros(length(AOA),1);
Cm_avg = zeros(length(AOA),1);

pdataRe_Ncrit = zeros(size(pdata,1),2);

for k = 1:size(pdata,1)
    pdataRe_Ncrit(k,1) = pdata{k,2};
    pdataRe_Ncrit(k,2) = pdata{k,3};
end

idx = find(pdataRe_Ncrit(:,1) == Re & pdataRe_Ncrit(:,2) == Ncrit);
pdata = pdata(idx,:);

%% Get Data files for each AOA at a specified Re and Ncrit and post-process
PostProcessDataFiles_Actual = GetPostProcessDataNames('Test_Actual_AOA_',AOA,Re,Ncrit);
PostProcessDataFiles_NNPredict = GetPostProcessDataNames('Test_NNpredict_AOA_',AOA,Re,Ncrit);
for i = 1:length(AOA)

    NNpredict.Cd = h5read(['NNpredict/',PostProcessDataFiles_NNPredict(i).name],'/Cd');
    NNpredict.Cdp = h5read(['NNpredict/',PostProcessDataFiles_NNPredict(i).name],'/Cdp');
    NNpredict.Cl = h5read(['NNpredict/',PostProcessDataFiles_NNPredict(i).name],'/Cl');
    NNpredict.Cm = h5read(['NNpredict/',PostProcessDataFiles_NNPredict(i).name],'/Cm');
    NNpredict.Cp_lower = h5read(['NNpredict/',PostProcessDataFiles_NNPredict(i).name],'/Cp_upper');
    NNpredict.Cp_upper = h5read(['NNpredict/',PostProcessDataFiles_NNPredict(i).name],'/Cp_lower');

    ActualData.Cd = h5read(['ActualData/',PostProcessDataFiles_Actual(i).name],'/Cd');
    ActualData.Cdp = h5read(['ActualData/',PostProcessDataFiles_Actual(i).name],'/Cdp');
    ActualData.Cl = h5read(['ActualData/',PostProcessDataFiles_Actual(i).name],'/Cl');
    ActualData.Cm = h5read(['ActualData/',PostProcessDataFiles_Actual(i).name],'/Cm');
    ActualData.Cp_upper = h5read(['ActualData/',PostProcessDataFiles_Actual(i).name],'/Cp_lower');
    ActualData.Cp_lower = h5read(['ActualData/',PostProcessDataFiles_Actual(i).name],'/Cp_upper');

    %% Plotting Pressure Coefficient
    figure('Position',[100 100 1300 600]);
    tiledlayout(1,2)
    nexttile

    plot(x_coord/c,NNpredict.Cp_upper,'--','LineWidth',1.5)
    hold on
    plot(x_coord/c,NNpredict.Cp_lower,'--','LineWidth',1.5)
    xlabel('$x/c$','Interpreter','latex','FontSize',12)
    ylabel('$C_p$','Interpreter','latex','FontSize',12)
    title('\textbf{Neural Network Prediction}','Interpreter','latex','FontSize',14)
    set(gca,'TickLabelInterpreter','latex','FontSize',12,'YDir','reverse')
    legend('Upper surface','Lower surface','Interpreter','latex','FontSize',12)
    grid on

    nexttile
    plot(x_coord/c,ActualData.Cp_upper,'.','MarkerSize',10)
    hold on
    plot(x_coord/c,ActualData.Cp_lower,'.','MarkerSize',10)
    xlabel('$x/c$','Interpreter','latex','FontSize',12)
    ylabel('$C_p$','Interpreter','latex','FontSize',12)
    title('\textbf{Actual Data}','Interpreter','latex','FontSize',14)
    set(gca,'TickLabelInterpreter','latex','FontSize',12,'YDir','reverse')
    legend('Upper surface','Lower surface','Interpreter','latex','FontSize',12)
    grid on

    sgtitle(['\textbf{Coefficient of Pressure Distribution at $\alpha$ = 10.25}'],'Interpreter','latex','FontSize',16)
end
%% Plot the remaining Polars
p1 = figure('Name','Cd','Position',[400 100 800 600]);
plot(pdata{1,1}.alpha,pdata{1,1}.Cd,'.','MarkerSize',10)
hold on
plot(AOA,Cd_avg,'--','LineWidth',1.5)
labelplot(p1,'$\alpha$','$C_d$','$C_d$ vs $\alpha$',1,{'Actual Data','Neural Network Prediction'})

p2 = figure('Name','Cdp','Position',[400 100 800 600]);
plot(pdata{1,1}.alpha,pdata{1,1}.Cdp,'.','MarkerSize',10)
hold on
plot(AOA,Cdp_avg,'--','LineWidth',1.5)
labelplot(p2,'$\alpha$','$C_{dp}$','$C_{dp}$ vs $\alpha$',1,{'Actual Data','Neural Network Prediction'})

p3 = figure('Name','Cl','Position',[400 100 800 600]);
plot(pdata{1,1}.alpha,pdata{1,1}.Cl,'.','MarkerSize',10)
hold on
plot(AOA,Cl_avg,'--','LineWidth',1.5)
labelplot(p3,'$\alpha$','$C_l$','$C_l$ vs $\alpha$',1,{'Actual Data','Neural Network Prediction'})

p1 = figure('Name','Cm','Position',[400 100 800 600]);
plot(pdata{1,1}.alpha,pdata{1,1}.Cm,'.','MarkerSize',10)
hold on
plot(AOA,Cm_avg,'--','LineWidth',1.5)
labelplot(p1,'$\alpha$','$C_m$','$C_m$ vs $\alpha$',1,{'Actual Data','Neural Network Prediction'})

%% Plot NN training Results
plotNNTrainingResults
