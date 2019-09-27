function test_dataset = read_txt_oxford(txt_path)
    fileID = fopen(txt_path);
    txt_content = textscan(fileID, '%f\t%f\t%s\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t', 'HeaderLines', 1);
    fclose(fileID);
    
    test_dataset = cell2mat(txt_content(:, [1,2,5,6,7,8,9,10,11]));
end