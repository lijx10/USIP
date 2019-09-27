function pc_keypoint_pair = build_filepath(dataset)
pc_keypoint_pair = {};

if strcmp(dataset, 'oxford')
    keypoint_folder = '/ssd/keypoint_results/oxford';
    pc_folder = '/ssd/dataset/oxford/storage/oxford_test_models_20k';
    
    for i=1:1:828
        keypoint_file = fullfile(keypoint_folder, sprintf('%d.bin', i-1));
        pc_file = fullfile(pc_folder, sprintf('%d.bin', i-1));
        
        pc_keypoint_pair{i} = {pc_file, keypoint_file};
    end
    
elseif strcmp(dataset, 'kitti')
    keypoint_folder = '/ssd/keypoint_results/kitti';
    pc_folder = '/ssd/dataset/kitti-reg-test';
    for seq=0:1:10
        keypoint_seq_folder = fullfile(keypoint_folder, sprintf('%02d', seq));
        pc_seq_folder = fullfile(pc_folder, sprintf('%02d', seq));
        
        content_keypoints = dir(fullfile(keypoint_seq_folder, '*.bin'));
        for i=1:1:length(content_keypoints)
            keypoint_file = fullfile(keypoint_seq_folder, content_keypoints(i).name);
            pc_file = fullfile(pc_seq_folder, content_keypoints(i).name);
            
            pc_keypoint_pair = [pc_keypoint_pair, {{pc_file, keypoint_file}}];
        end
    end
    
elseif strcmp(dataset, 'redwood')
    keypoint_folder = '/ssd/keypoint_results/redwood';
    pc_folder = '/ssd/dataset/redwood/numpy_gt_normal';
    
    scene_list = {'livingroom1', 'livingroom2', 'office1', 'office2'};
    for scene_idx=1:1:4
        scene = scene_list{scene_idx};
        content_keypoints = dir(fullfile(keypoint_folder, scene, '*.bin'));
        for i=1:1:length(content_keypoints)
            keypoint_file = fullfile(keypoint_folder, scene, content_keypoints(i).name);
            pc_file = fullfile(pc_folder, scene, [content_keypoints(i).name(1:end-3), 'npy']);
            
            pc_keypoint_pair = [pc_keypoint_pair, {{pc_file, keypoint_file}}];
        end
    end
    
elseif strcmp(dataset, 'modelnet')
    keypoint_folder = '/ssd/keypoint_results/modelnet';
    pc_folder = '/ssd/dataset/modelnet40-normal_numpy';
    
    test_txt = fopen(fullfile(pc_folder, 'modelnet40_test.txt'), 'r');
    test_sequence = textscan(test_txt, '%s\n');
    test_sequence = test_sequence{1};
    fclose(test_txt);
    
    for i=1:1:size(test_sequence, 1)
        keypoint_file = fullfile(keypoint_folder, sprintf('%d.bin', i-1));
        
        class_name = test_sequence{i,1}(1:end-5);
        pc_file = fullfile(pc_folder, class_name, [test_sequence{i,1}, '.npy']);
        
        pc_keypoint_pair{i} = {pc_file, keypoint_file};
    end
end
    
end