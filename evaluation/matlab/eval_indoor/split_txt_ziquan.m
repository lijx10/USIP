clear all

% % Synthetic data benchmark
pcRoot = '/ssd/dataset/redwood/numpy_gt_normal';
resultRoot = '/ssd/redwood_data/1024k1k5-512s512-0.75m448-0.767';
sceneList = {'livingroom1', 'livingroom2', 'office1', 'office2'};

inlierThreshold = 0.05;
method = 'point';  % 'feature' / 'point'

dataPath = '/ssd/redwood_data';
gtPath = '/ssd/dataset/redwood/original';
for sceneIdx = 1:length(sceneList)
    result = mrLoadLogMy(fullfile(dataPath,sprintf('%s.log', sceneList{sceneIdx})));
    
    gt = mrLoadLog(fullfile(gtPath, sprintf('%s-evaluation', sceneList{sceneIdx}), 'gt.log'));
    gt_info = mrLoadInfo(fullfile(gtPath, sprintf('%s-evaluation', sceneList{sceneIdx}), 'gt.info'));
    
    result_odom = [];
    information_odom = [];
    result_loop = [];
    information_loop = [];
    parfor i=1:1:length(result)
        seq_array = result(i).info;
        if seq_array(2)-seq_array(1) == 1
            % odometry
            srcIdx = result(i).info(1);
            dstIdx = result(i).info(2);
            estimateRt = result(i).trans;
            
            % use gt pose
            for j=1:1:length(gt)
                if isequal(gt(j).info, result(i).info)
                    estimateRt = gt(j).trans;
                    break
                end
            end
            
            correspondence = computeCorrepondence(pcRoot, resultRoot, sceneList{sceneIdx}, srcIdx, dstIdx, estimateRt, inlierThreshold);
            
            if ~isempty(correspondence)
                inlierNum = size(correspondence, 1);
                inlierRatio = result(i).inlierRatio;
                outlierNum = round(inlierNum / inlierRatio - inlierNum);

                result_odom = [result_odom, struct('info', result(i).info, ...
                    'inlier', [inlierNum, outlierNum, inlierRatio], ...
                    'rt', estimateRt(1:3, 1:4), ...
                    'correspondence', correspondence)];
            end
            
            
        else
            % loop closure     
            srcIdx = result(i).info(1);
            dstIdx = result(i).info(2);
            estimateRt = result(i).trans;
            
            correspondence = computeCorrepondence(pcRoot, resultRoot, sceneList{sceneIdx}, srcIdx, dstIdx, estimateRt, inlierThreshold);
            
            if ~isempty(correspondence)
                inlierNum = size(correspondence, 1);
                inlierRatio = result(i).inlierRatio;
                outlierNum = round(inlierNum / inlierRatio - inlierNum);

                result_loop = [result_loop, struct('info', result(i).info, ...
                    'inlier', [inlierNum, outlierNum, inlierRatio], ...
                    'rt', result(i).trans(1:3, 1:4), ...
                    'correspondence', correspondence)];
            end
            
        end
    end
    
    % fix missing in odometry data
    for i=1:1:length(gt)
        seq_array = gt(i).info;
        if seq_array(2)-seq_array(1) == 1
            % search in result_odom
            is_exist = 0;
            for j=1:1:length(result_odom)
                if isequal(gt(i).info, result_odom(j).info)
                    is_exist = 1;
                    break;
                end
            end
            
            % missing in result_odom
            if is_exist == 0
                srcIdx = gt(i).info(1);
                dstIdx = gt(i).info(2);
                estimateRt = gt(i).trans;
                correspondence = computeCorrepondence(pcRoot, resultRoot, sceneList{sceneIdx}, srcIdx, dstIdx, estimateRt, inlierThreshold);
                
                if ~isempty(correspondence)
                    inlierNum = size(correspondence, 1);
                    inlierRatio = 0.3;
                    outlierNum = round(inlierNum / inlierRatio - inlierNum);

                    result_odom = [result_odom, struct('info', result(i).info, ...
                        'inlier', [inlierNum, outlierNum, inlierRatio], ...
                        'rt', result(i).trans(1:3, 1:4), ...
                        'correspondence', correspondence)];
                end
            end
        end
    end
    
    txtWriteZiquan(result_odom, fullfile(dataPath,sprintf('%s_odom.txt', sceneList{sceneIdx})));
    txtWriteZiquan(result_loop, fullfile(dataPath,sprintf('%s_loop.txt', sceneList{sceneIdx})));
    
end

function correspondence = computeCorrepondence(pcRoot, resultRoot, scene, srcIdx, dstIdx, estimateRt, inlierThreshold)
FEATURE_DIM = 128;

% % load keypoints and descriptors of fragment 1
result1path = fullfile(resultRoot, scene, [num2str(srcIdx), '.bin']);
fragment1_xyz_features = Utils.load_descriptors(result1path, sum(FEATURE_DIM+3));
fragment1Keypoints = fragment1_xyz_features(:, 1:3);
fragment1Descriptors = fragment1_xyz_features(:, 4:end);

% load keypoints and descriptors of fragment 2
result2path = fullfile(resultRoot, scene, [num2str(dstIdx), '.bin']);
fragment2_xyz_features = Utils.load_descriptors(result2path, sum(FEATURE_DIM+3));
fragment2Keypoints = fragment2_xyz_features(:, 1:3);
fragment2Descriptors = fragment2_xyz_features(:, 4:end);



% kNN matching
k = 5;
[~, matches12] = pdist2(fragment2Descriptors, fragment1Descriptors, 'euclidean', 'smallest', k);
matches12 = reshape(matches12', [], 1);  % Nxk -> kNx1
matches12 = [repmat([1:length(fragment1Keypoints)]', k, 1), matches12];

[~, matches21] = pdist2(fragment1Descriptors, fragment2Descriptors, 'euclidean', 'smallest', k);
matches21 = reshape(matches21', [], 1);  % Nxk -> kNx1
matches21 = [matches21, repmat([1:length(fragment2Keypoints)]', k, 1)];

% % union of matches21 & matches12
matches12 = union(matches12, matches21, 'rows');
% matches12 = [matches12; matches21];

% % intersection of matches21 & matches12
% matches12 = intersect(matches12, matches21, 'rows');

fragment1MatchKeypoints = fragment1Keypoints(matches12(:,1), :);
fragment2MatchKeypoints = fragment2Keypoints(matches12(:,2), :);

fragment2MatchKeypoints_T = (estimateRt(1:3,1:3) * fragment2MatchKeypoints' + repmat(estimateRt(1:3,4),1,size(fragment2MatchKeypoints',2)))';
dist = vecnorm(fragment1MatchKeypoints - fragment2MatchKeypoints_T, 2, 2);
inlierIdx = find(dist < inlierThreshold);

if isempty(inlierIdx)
    fprintf('%s - %d - %d\n', scene, srcIdx, dstIdx);
    correspondence = [];
else
    correspondence = [fragment1MatchKeypoints(inlierIdx, :), fragment2MatchKeypoints(inlierIdx, :), dist(inlierIdx)];
end



end

