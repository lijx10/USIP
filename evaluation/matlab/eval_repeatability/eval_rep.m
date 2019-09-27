% suppress warning of parfor
warning('off', 'all')

% configurations
dataset_type = 'kitti';
is_visualize = false;  
inlier_radius = 0.5;

pc_parent_folder = '/ssd/jiaxin/TSF_datasets';
keypoint_parent_folder = '/ssd/jiaxin/keypoints_knn_ball';
keypoint_method_folder = 'tsf_BALL-16384-512-r2k64-k16';
% keypoint_method_folder = 'tsf_98B75DFF';
if strcmp(dataset_type, 'oxford')
    PC_FOLDER = fullfile(pc_parent_folder, 'oxford/oxford_test_models_20k');
    KEYPOINT_FOLDER = fullfile(keypoint_parent_folder, 'oxford', keypoint_method_folder);
    test_dataset = read_txt_oxford(fullfile(PC_FOLDER, 'groundtruths.txt'));
elseif strcmp(dataset_type, 'kitti')
    PC_FOLDER = fullfile(pc_parent_folder, 'kitti-reg-test');
    KEYPOINT_FOLDER = fullfile(keypoint_parent_folder, 'kitti', keypoint_method_folder);
    test_dataset = read_txts_correct_kitti(PC_FOLDER);
elseif strcmp(dataset_type, 'redwood')
    PC_FOLDER = fullfile(pc_parent_folder, 'redwood/numpy_gt_normal');
    KEYPOINT_FOLDER = fullfile(keypoint_parent_folder, 'redwood', keypoint_method_folder);
    test_dataset = build_redwood_dataset(fullfile(pc_parent_folder, 'redwood'));
elseif strcmp(dataset_type, '3dmatch')
    PC_FOLDER = fullfile(pc_parent_folder, '3DMatch_eval_npy');
    KEYPOINT_FOLDER = fullfile(keypoint_parent_folder, '3dmatch', keypoint_method_folder);
    test_dataset = build_3dmatch_dataset(fullfile(pc_parent_folder, '3DMatch_eval_npy'));
elseif strcmp(dataset_type, 'modelnet')
    PC_FOLDER = fullfile(pc_parent_folder, 'modelnet40-test_rotated_numpy');
    KEYPOINT_FOLDER = fullfile(keypoint_parent_folder, 'modelnet', keypoint_method_folder);
    load(fullfile(PC_FOLDER, 'modelnet_info.mat'));
    test_dataset = modelnet_info;
end


repeatability_array = [];
keypoint_num_array = [];
parfor i = 1:1:size(test_dataset, 1)
    % load pc & keypoint
    if strcmp(dataset_type, 'oxford')
        anc_idx = int64(test_dataset(i, 1));
        anc_pc_path = fullfile(PC_FOLDER, sprintf('%d.bin', anc_idx));
        anc_keypoint_path = fullfile(KEYPOINT_FOLDER, sprintf('%d.bin', anc_idx));
        anc_pc = Utils.loadPointCloud(anc_pc_path, 6);
        anc_keypoint = Utils.loadPointCloud(anc_keypoint_path, 3);
        anc_pc = anc_pc(:, 1:3);
        anc_keypoint = coord_cam2enu(anc_keypoint);
        
        pos_idx = int64(test_dataset(i, 2));
        pos_pc_path = fullfile(PC_FOLDER, sprintf('%d.bin', pos_idx));
        pos_keypoint_path = fullfile(KEYPOINT_FOLDER, sprintf('%d.bin', pos_idx));
        pos_pc = Utils.loadPointCloud(pos_pc_path, 6);
        pos_keypoint = Utils.loadPointCloud(pos_keypoint_path, 3);
        pos_pc = pos_pc(:, 1:3);
        pos_keypoint = coord_cam2enu(pos_keypoint);
        
        t_gt = test_dataset(i, 3:5);
        q_gt = test_dataset(i, 6:9);
        T_gt = [quat2rotm(q_gt), t_gt'];  % 3x4
    elseif strcmp(dataset_type, 'kitti')
        seq = int64(test_dataset(i, 1));
        
        anc_idx = int64(test_dataset(i, 2));
        anc_pc_path = fullfile(PC_FOLDER, sprintf('%02d', seq), sprintf('%06d.bin', anc_idx));
        anc_keypoint_path = fullfile(KEYPOINT_FOLDER, sprintf('%02d', seq), sprintf('%06d.bin', anc_idx));
        anc_pc = Utils.loadPointCloud(anc_pc_path, 6);
        anc_keypoint = Utils.loadPointCloud(anc_keypoint_path, 3);
        anc_pc = anc_pc(:, 1:3);
        calib = read_kitti_calib(fullfile(pc_parent_folder, ['kitti/calib/', sprintf('%02d', seq), '/calib.txt']));
        Tr = calib.Tr;
        anc_keypoint = cam2velodyne(anc_keypoint, Tr);
        
        pos_idx = int64(test_dataset(i, 3));
        pos_pc_path = fullfile(PC_FOLDER, sprintf('%02d', seq), sprintf('%06d.bin', pos_idx));
        pos_keypoint_path = fullfile(KEYPOINT_FOLDER, sprintf('%02d', seq), sprintf('%06d.bin', pos_idx));
        pos_pc = Utils.loadPointCloud(pos_pc_path, 6);
        pos_keypoint = Utils.loadPointCloud(pos_keypoint_path, 3);
        pos_pc = pos_pc(:, 1:3);
        calib = read_kitti_calib(fullfile(pc_parent_folder, ['kitti/calib/', sprintf('%02d', seq), '/calib.txt']));
        Tr = calib.Tr;
        pos_keypoint = cam2velodyne(pos_keypoint, Tr);
        
        t_gt = test_dataset(i, 4:6);
        q_gt = test_dataset(i, 7:10);
        T_gt = [quat2rotm(q_gt), t_gt'];  % 3x4
    elseif strcmp(dataset_type, 'redwood')
        scene = test_dataset{i, 1};
        anc_idx = test_dataset{i, 2};
        anc_pc = readNPY(fullfile(PC_FOLDER, scene, sprintf('%d.npy', anc_idx)));
        anc_pc = anc_pc(:, 1:3);
        anc_keypoint = Utils.loadPointCloud(fullfile(KEYPOINT_FOLDER, scene, sprintf('%d.bin', anc_idx)), 3);
        
        pos_idx = test_dataset{i, 3};
        pos_pc = readNPY(fullfile(PC_FOLDER, scene, sprintf('%d.npy', pos_idx)));
        pos_pc = pos_pc(:, 1:3);
        pos_keypoint = Utils.loadPointCloud(fullfile(KEYPOINT_FOLDER, scene, sprintf('%d.bin', pos_idx)), 3);
        
        T = test_dataset{i, 4};
        T_gt = T(1:3, :);  % 3x4
    elseif strcmp(dataset_type, '3dmatch')
        scene = test_dataset{i, 1};
        anc_idx = test_dataset{i, 2};
        anc_pc = readNPY(fullfile(PC_FOLDER, scene, sprintf('cloud_bin_%d.npy', anc_idx)));
        anc_pc = anc_pc(:, 1:3);
        anc_keypoint = Utils.loadPointCloud(fullfile(KEYPOINT_FOLDER, scene, sprintf('%d.bin', anc_idx)), 3);
        
        pos_idx = test_dataset{i, 3};
        pos_pc = readNPY(fullfile(PC_FOLDER, scene, sprintf('cloud_bin_%d.npy', pos_idx)));
        pos_pc = pos_pc(:, 1:3);
        pos_keypoint = Utils.loadPointCloud(fullfile(KEYPOINT_FOLDER, scene, sprintf('%d.bin', pos_idx)), 3);
        
        T = test_dataset{i, 4};
        T_gt = T(1:3, :);  % 3x4
    elseif strcmp(dataset_type, 'modelnet')
        anc_idx = i - 1;
        anc_pc = readNPY(fullfile(PC_FOLDER, 'original', sprintf('%d.npy', anc_idx)));
        anc_pc = anc_pc(:, 1:3);
        anc_keypoint = Utils.loadPointCloud(fullfile(KEYPOINT_FOLDER, 'original', sprintf('%d.bin', anc_idx)), 3);
        
        pos_idx = i - 1;
        pos_pc = readNPY(fullfile(PC_FOLDER, 'rotated', sprintf('%d.npy', pos_idx)));
        pos_pc = pos_pc(:, 1:3);
        pos_keypoint = Utils.loadPointCloud(fullfile(KEYPOINT_FOLDER, 'rotated', sprintf('%d.bin', pos_idx)), 3);
        
        T_gt = test_dataset{i, 3};
    end
    
    % visualization
    if is_visualize
        fig = figure(1);
        pcshow(anc_pc);
        hold on
        pcshow(anc_keypoint, [1, 0, 0], 'MarkerSize', 128);
        hold on
        pcshow(Utils.apply_transform(pos_pc, T_gt))
        hold on
        pcshow(Utils.apply_transform(pos_keypoint, T_gt), [0, 1, 0], 'MarkerSize', 128)
        hold off;
    end
    
    % compute repeatability
    keypoint_anc_to_pos = pdist2(Utils.apply_transform(pos_keypoint, T_gt), anc_keypoint, 'euclidean', 'Smallest', 1);
    inlier_number = sum(keypoint_anc_to_pos < inlier_radius);
    repeatability_array = [repeatability_array, inlier_number / size(anc_keypoint, 1)];
    keypoint_num_array = [keypoint_num_array, size(anc_keypoint, 1)];
    
    if is_visualize
        % break;
    end
end
fprintf('%s\n', KEYPOINT_FOLDER);
fprintf("%s: average keypoint number: %d, repeatability: %f\n", dataset_type, mean(keypoint_num_array), mean(repeatability_array));
fprintf("%s: min repeatability: %f, max repeatability: %f\n", dataset_type, min(repeatability_array), max(repeatability_array));