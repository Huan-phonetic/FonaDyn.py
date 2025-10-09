function [names, vrpArray]=FonaDynLoadVRP(filename)
%% function [names, vrpArray]=FonaDynLoadVRP(filename)
%  Import VRP data from _VRP.csv text file into an array (FonaDyn v(3.0.6). 
%  <filename> is the name of the _VRP.CSV file
%  Returns a cell array "names" containing the column names found on the first line,
%  and a 2D array "vrpArray" with the numeric data as found on the remaining lines.
  
%% Initialize variables.
%filename = '<your-path-here>/<your-filename-here>_VRP.csv';
delimiter = ';';
startRow = 2;
  
%% Open the text file.
fileID = fopen(filename,'r');
FirstLine=fgetl(fileID);
names=strsplit(FirstLine,delimiter);
% Deal with trailing delimiter at EOL
if isempty(names{end})
    VarLength=length(names)-1;
else
    VarLength=length(names);
end
  
names = names(1:VarLength);
  
formatSpec=repmat('%f',1,VarLength);
  
%fileID = fopen(filename,'r');
  
%% Read columns of data according to format string.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'ReturnOnError', false);
  
%% Close the text file.
fclose(fileID);
  
%% Allocate imported array to column variable names
  
arr = zeros(size(dataArray{1}, 1), size(dataArray,2)); 
for i = 1 : VarLength
    arr(:,i) = dataArray{:, i};
end;
  
vrpArray = arr;
  
%% Clear temporary variables
clearvars delimiter startRow formatSpec fileID dataArray;
  
end
