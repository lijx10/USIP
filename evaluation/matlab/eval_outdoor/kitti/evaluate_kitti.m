DATA_FOLDER = '/ssd/jiaxin/TSF_datasets/kitti-reg-test';
RESULT_FOLDER = '/ssd/jiaxin/keypoint_descriptor/kitti/shot/iss_256';
FEATURE_DIM = 352;

warning('off', 'MATLAB:mir_warning_maybe_uninitialized_temporary')
global maxTrials;
if isempty(maxTrials) || maxTrials == 10001
    maxTrials = 10001;
    isPrintWrong = true;
else
    fprintf('\nrunning in automation mode, maxTrials %d ......\n', maxTrials);
    isPrintWrong = false;
end


% test_dataset = read_txts_kitti(DATA_FOLDER);
test_dataset = read_txts_correct_kitti(DATA_FOLDER);

wrong_counter = 0;
inlier_ratio_array = [];
trial_count_array = [];
delta_t_array = [];
delta_deg_array = [];
parfor i = 1:1:size(test_dataset, 1)
    
    seq = int64(test_dataset(i, 1));
    
    anc_idx = int64(test_dataset(i, 2));
    anc_desc_fname = fullfile(RESULT_FOLDER, sprintf('%02d', seq), sprintf('%06d.bin', anc_idx));
    anc_xyz_features = Utils.load_descriptors(anc_desc_fname, sum(FEATURE_DIM+3));
    anc_result_desc = anc_xyz_features(:, 4:end);

    pos_idx = int64(test_dataset(i, 3));
    pos_desc_fname = fullfile(RESULT_FOLDER, sprintf('%02d', seq), sprintf('%06d.bin', pos_idx));
    pos_xyz_features = Utils.load_descriptors(pos_desc_fname, sum(FEATURE_DIM+3));
    pos_result_desc = pos_xyz_features(:, 4:end);
    
    % for my case, cam -> velodyne transform
    % if ~isempty(strfind(RESULT_FOLDER, 'my'))
    if true
        calib = read_kitti_calib(['/ssd/jiaxin/TSF_datasets/kitti/calib/', sprintf('%02d', seq), '/calib.txt']);
        Tr = calib.Tr;
        anc_result_xyz = cam2velodyne(anc_xyz_features(:, 1:3), Tr);
        pos_result_xyz = cam2velodyne(pos_xyz_features(:, 1:3), Tr);
    else
        % normal situation, no need to do coordinate transform
        anc_result_xyz = anc_xyz_features(:, 1:3);
        pos_result_xyz = pos_xyz_features(:, 1:3);
    end
    
    
     % 1NN Match
    [~, matches12] = pdist2(pos_result_desc, anc_result_desc, 'euclidean', 'smallest', 1);
    matches12 = [1:length(matches12); matches12]';  
    
    
    
% %     % kNN matching
%     k = 5;
%     [~, matches12] = pdist2(pos_result_desc, anc_result_desc, 'euclidean', 'smallest', k);
%     matches12 = reshape(matches12', [], 1);  % Nxk -> kNx1
%     matches12 = [repmat([1:length(anc_result_xyz)]', k, 1), matches12];
% 
%     [~, matches21] = pdist2(anc_result_desc, pos_result_desc, 'euclidean', 'smallest', k);
%     matches21 = reshape(matches21', [], 1);  % Nxk -> kNx1
%     matches21 = [matches21, repmat([1:length(pos_result_xyz)]', k, 1)];
% 
%     % % union of matches21 & matches12
%     matches12 = union(matches12, matches21, 'rows');
    % matches12 = [matches12; matches21];

    % % intersection of matches21 & matches12
%     matches12 = intersect(matches12, matches21, 'rows');




    % RANSAC
    cloud1_pts = anc_result_xyz(matches12(:,1), :);
    cloud2_pts = pos_result_xyz(matches12(:,2), :);
    [estimateRt, inlierIdx, trialCount] = ransacfitRt([cloud1_pts'; cloud2_pts'], 1.0, false, maxTrials);
%     fprintf('Number of inliers: %i / %i (Proportion: %.3f. #RANSAC trials: %i)\n', ...
%             length(inlierIdx), size(matches12, 1), ...
%             length(inlierIdx)/size(matches12, 1), trialCount);
        
        
        
    t_gt = test_dataset(i, 4:6);
    q_gt = test_dataset(i, 7:10);
    T_gt = [quat2rotm(q_gt), t_gt'];
    
    try
        [delta_t, delta_deg] = Utils.compareTransform(T_gt, estimateRt);
    catch
        delta_t = 3;
        delta_deg = 6;
    end
    
    % catch wrong GT
    if isPrintWrong && ((delta_t < 3 && delta_t > 2 && delta_deg < 5) || (delta_deg < 6 && delta_deg > 5 && delta_t < 2))
        fprintf('!!! %d !!!', i);
    end
    
    if delta_t > 2 || delta_deg > 5
        wrong_counter = wrong_counter + 1;
%         fprintf('wrong counter %d / %d \n', wrong_counter, i);
        if isPrintWrong
            fprintf('--- %d --- Number of inliers: %i / %i (Proportion: %.3f. #RANSAC trials: %i)\n', ...
                i, ...
                length(inlierIdx), size(matches12, 1), ...
                length(inlierIdx)/size(matches12, 1), trialCount);
        end
    else
        % record the inlier/outlier ratio
        inlier_ratio_array = [inlier_ratio_array, length(inlierIdx)/size(matches12, 1)];
        trial_count_array = [trial_count_array, trialCount];
        
        delta_t_array = [delta_t_array, delta_t];
        delta_deg_array = [delta_deg_array, delta_deg];
    end
    
%     if i > 100
%         break;
%     end

%     break
    
end

fprintf('wrong counter: %d\ninlier ratio %f\ntrial count %f\n', wrong_counter, mean(inlier_ratio_array), mean(trial_count_array));
fprintf('successful alignment RTE: %f +- %f, RRE: %f +- %f\n', mean(delta_t_array), std(delta_t_array), mean(delta_deg_array), std(delta_deg_array));