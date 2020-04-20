function list = read_file_list(path)
%   This function returns content of a file
%   path: file path

    fileID = fopen(path);
    cell = textscan(fileID, '%s', 'EndOfLine', '\n', 'Whitespace', '\t');
    fclose(fileID);
    list = cell{:};
end
