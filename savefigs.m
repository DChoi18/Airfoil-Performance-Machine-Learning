function savefigs(foldername)
%function to save all figures generated from a script into a folder 

mkdir(foldername)
%% Get number of figures generated
n = get(gcf,'Number');
%% Save Figures
for i = 1:n
    figure(i)
    set(gcf,'PaperPositionMode','auto','InvertHardCopy','off','Color','w')
    pic_name = get(gcf,'Name');
    print(gcf,'-dpng','-r600',['./',foldername,'/',pic_name])
end
end