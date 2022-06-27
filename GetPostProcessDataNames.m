function filenames = GetPostProcessDataNames(testset,AOA,Re,NCrit)

% filenames = struct('name','');
% for i = 1:length(AOA)
%     filenames(i).name = [testset num2str(AOA(i)) '_Re_' num2str(Re) '.0_Ncrit_' num2str(NCrit) '.0.h5'];
% end

names = dir([ testset,'*.h5']);
temp = cell(length(names),2);

for i = 1:length(names)
    temp{i,1} = names(i).name;
    str_temp = temp{i};
    alpha = strsplit(str_temp,'_');
    alpha = alpha{4};
    temp{i,2} = str2double(alpha);
end

temp = sortrows(temp,2);
filenames = struct('name','');
for i = 1:length(names)
    filenames(i).name = temp{i,1};
end