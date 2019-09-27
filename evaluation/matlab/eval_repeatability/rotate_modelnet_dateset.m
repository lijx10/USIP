dataset_root = '/ssd/jiaxin/TSF_datasets/modelnet40-normal_numpy';

output_root = '/ssd/jiaxin/TSF_datasets/modelnet40-test_numpy';
output_original_root = fullfile(output_root, 'original');
output_rotated_root = fullfile(output_root, 'rotated');

test_obj_list_txt = fullfile(dataset_root, 'modelnet40_test.txt');
fid = fopen(test_obj_list_txt);
test_obj_list = textscan(fid, '%s');
test_obj_list = test_obj_list{1,1};
fclose(fid);

modelnet_info = {};
for i=1:1:length(test_obj_list)
    obj_name = test_obj_list{i};
    obj_label = obj_name(1:end-5);
    
    obj = readNPY(fullfile(dataset_root, obj_label, sprintf("%s.npy", obj_name)));
    obj_output_path = fullfile(output_original_root, sprintf("%d.npy", i-1));
    writeNPY(obj, obj_output_path);
    
    random_angle = (rand(1, 3) - 0.5) * 2 * pi;
    random_R = eul2rotm(random_angle);
    obj_rotated = [obj(:, 1:3) * random_R, obj(:, 4:6) * random_R];
    obj_rotated_output_path = fullfile(output_rotated_root, sprintf("%d.npy", i-1));
    writeNPY(obj_rotated, obj_rotated_output_path);
    
    T = [random_R, [0;0;0]];
    
    modelnet_info{i, 1} = obj_output_path;
    modelnet_info{i, 2} = obj_rotated_output_path;
    modelnet_info{i, 3} = T;
    
    % visualization
%     figure(1);
%     pcshow(obj(:, 1:3));
%     figure(2);
%     pcshow(obj_rotated(:, 1:3));
%     break;
end

save(fullfile(output_root, 'modelnet_info.mat'), 'modelnet_info');