function airfoilData = sortbyRe(data)
% Purpose split airfoil data by Re
%
% Input:
%       data = table variable:
%              - polars (Cd,Cp,Cl,etc) at each alpha, Re, and Ncrit
% Output:
%       airfoilData = data table variable rearranged as a cell array
%                     with data sets split by Re
% Author: Derrick Choi

idx = find(diff(data.Re)~=0); % indices of were Re changes

start_stop = [0;idx;length(data.Re)]; % indices to grab data

airfoilData = cell(length(idx)+1,2); % preallocate

% Split data by Re
for i = 2:length(start_stop)
    cutData = data(start_stop(i-1)+1:start_stop(i),:);
    airfoilData{i-1,1} = cutData;
    airfoilData{i-1,2} = (data.Re(start_stop(i-1)+1));
end

end