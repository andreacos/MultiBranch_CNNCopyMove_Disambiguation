function [file_path, file_name] = get_file_list(path, file_path, file_name, ext)
% This function extracts all files (path and name) recursively
% Input
%   path: the root directory
%   file_path: the current list of file paths
%   file_name: the current list of file names
%   ext: file extension of interest [optional]. If not indicated, all
%   files will be considered
% Output:
%  file_path: list of all file paths
%  file_name: list of all file names
% Usage: [file_path, file_name] = get_file_list('example/jpeg', [], [], '*.jpg')
    
    if nargin == 4 % list all files with indicated extension
        list_dir = dir([path filesep ext]);
    else % list all files
        list_dir = dir(path);
    end
    
    for f = list_dir'
        if strcmp(f.name, '.') || strcmp(f.name, '..') || strcmp(f.name, '.DS_Store') || ...
                strcmp(f.name, '._.DS_Store') || ~isempty(strfind(f.name, '._'))
            continue
        end
        if f.isdir == 1
            [file_path, file_name] = get_file_list([path filesep f.name], file_path, file_name);
        else
            file_path = [file_path; cellstr([path filesep f.name])];
            file_name = [file_name; cellstr(f.name)];
        end
    end
end