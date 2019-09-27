% Visualizes matches from descriptors
%
% i.e. no augmentations, etc
% clear all
close all

addpath('./external');
DATA_FOLDER = '/ssd/dataset/kitti-reg-test';

test_dataset = read_txts_correct_kitti(DATA_FOLDER);
dataset_idx = 1390;
seq = test_dataset(dataset_idx, 1);
anc_idx = test_dataset(dataset_idx, 2);
pos_idx = test_dataset(dataset_idx, 3);
t_gt = test_dataset(dataset_idx, 4:6);
q_gt = test_dataset(dataset_idx, 7:10);
T_gt = [quat2rotm(q_gt), t_gt'];


% clear;
m = 6;  % Dimensionality of raw data (all datasets are XYZNxNyNz)
DRAW_ALL_PUTATIVE = false;  % If true, will draw all inlier/outlier matches
MAX_MATCHES = 1000; % Maximum number of inlier+outlier matches to draw

DATA_FOLDER = '/ssd/dataset/kitti-reg-test';
% RESULT_FOLDER = '/ssd/3dfeatnet_data/kitti/my-k1k5(k1k9)-k1k9(2m64)';
RESULT_FOLDER = '/ssd/3dfeatnet_data/kitti/my';
DATA_PAIRS = {sprintf('%02d/%06d', seq, anc_idx), sprintf('%02d/%06d', seq, pos_idx)};

FEATURE_DIM = 128;

%% Load pairs and runs matching+RANSAC
for iPair = 1 : size(DATA_PAIRS, 1)
    
    pair = DATA_PAIRS(iPair,:);
    cloud_fnames = {[fullfile(DATA_FOLDER, pair{1}), '.bin'], ...
                [fullfile(DATA_FOLDER, pair{2}), '.bin']};
    desc_fnames = {[fullfile(RESULT_FOLDER, pair{1}), '.bin'], ...
                   [fullfile(RESULT_FOLDER, pair{2}), '.bin']};
               
               
    % Load point cloud and descriptors
    fprintf('Running on frames:\n');
    fprintf('- %s\n', cloud_fnames{1});
    fprintf('- %s\n', cloud_fnames{2});

    for i = 1 : 2
        pointcloud{i} = Utils.loadPointCloud(cloud_fnames{i}, m);
        xyz_features = Utils.load_descriptors(desc_fnames{i}, sum(FEATURE_DIM+3));
        result{i}.desc = xyz_features(:, 4:end);

        % for my case, cam -> velodyne transform
        if ~isempty(strfind(RESULT_FOLDER, 'my'))
            calib = read_kitti_calib(['/ssd/dataset/odometry/calib/', pair{i}(1:2), '/calib.txt']);
            Tr = calib.Tr;
            
            xyz_cam = xyz_features(:, 1:3);
            xyz_cam = cam2velodyne(xyz_cam, Tr);
            result{i}.desc = xyz_features(:, 4:end);
            result{i}.xyz = xyz_cam;
        else
            % normal situation, no need to do coordinate transform
            result{i}.xyz = xyz_features(:, 1:3);
        end
    end

    % Match
    [~, matches12] = pdist2(result{2}.desc, result{1}.desc, 'euclidean', 'smallest', 1);
    matches12 = [1:length(matches12); matches12]';  

    %  RANSAC
    cloud1_pts = result{1}.xyz(matches12(:,1), :);
    cloud2_pts = result{2}.xyz(matches12(:,2), :);
    [estimateRt, inlierIdx, trialCount] = ransacfitRt([cloud1_pts'; cloud2_pts'], 1.0, false);
    fprintf('Number of inliers: %i / %i (Proportion: %.3f. #RANSAC trials: %i)\n', ...
            length(inlierIdx), size(matches12, 1), ...
            length(inlierIdx)/size(matches12, 1), trialCount);
    estimateRt
    T_gt
    [delta_t, delta_deg] = Utils.compareTransform(T_gt, estimateRt)
    

    
    % Shows result
    figure(iPair * 2 - 1); clf
    if DRAW_ALL_PUTATIVE
        Utils.pcshow_matches(pointcloud{1}, pointcloud{2}, ...
                         result{1}.xyz, result{2}.xyz, ...
                         matches12, 'inlierIdx', inlierIdx, 'k', MAX_MATCHES);
    else
        Utils.pcshow_matches(pointcloud{1}, pointcloud{2}, ...
                             result{1}.xyz, result{2}.xyz, ...
                             matches12(inlierIdx, :), 'k', MAX_MATCHES);
    end
    title('Matches')

    % Show alignment
    figure(iPair * 2); clf
    Utils.pcshow_multiple(pointcloud, {eye(4), estimateRt});
    title('Alignment')
        
end


