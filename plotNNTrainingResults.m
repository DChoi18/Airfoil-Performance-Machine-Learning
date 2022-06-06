function plotNNTrainingResults()
results = readmatrix('file_conv.txt','NumHeaderLines',1);

fig = figure('Position',[400 100 800 600]);
semilogy(results(:,1),results(:,2),'LineWidth',1.5)
labelplot(fig,'Epoch','Mean Squared Error','Training Loss Function',0)

end