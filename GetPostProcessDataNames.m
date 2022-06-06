function filenames = GetPostProcessDataNames(testset,AOA,Re,NCrit)

filenames = struct('name','');
for i = 1:length(AOA)
    filenames(i).name = [testset num2str(AOA(i)) '_Re_' num2str(Re) '.0_Ncrit_' num2str(NCrit) '.0.h5'];
end

end