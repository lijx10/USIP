pcRoot = '/ssd/dataset/redwood/numpy_gt_normal';
resultRoot = '/ssd/redwood_data/my';
sceneName = 'office1';  % {'livingroom1', 'livingroom2', 'office1', 'office2'};
fragment1Idx = 33;
fragment2Idx = 35;
FEATURE_DIM = 128;
inlierThreshold = 0.2;

pc1path = fullfile(pcRoot, sceneName, sprintf('%d.npy', fragment1Idx));
pc2path = fullfile(pcRoot, sceneName, sprintf('%d.npy', fragment2Idx));
resutl1path = fullfile(resultRoot, sceneName, sprintf('%d.bin', fragment1Idx));
result2path = fullfile(resultRoot, sceneName, sprintf('%d.bin', fragment2Idx));

% Load fragment point clouds
fragment1PointCloud = readNPY(pc1path);
fragment1PointCloud = pointCloud(fragment1PointCloud(:, 1:3));
fragment2PointCloud = readNPY(pc2path);
fragment2PointCloud = pointCloud(fragment2PointCloud(:, 1:3));

% load keypoints and descriptors of fragment 1
fragment1_xyz_features = Utils.load_descriptors(resutl1path, sum(FEATURE_DIM+3));
fragment1Keypoints = fragment1_xyz_features(:, 1:3);
fragment1Descriptors = fragment1_xyz_features(:, 4:end);

% load keypoints and descriptors of fragment 1
fragment2_xyz_features = Utils.load_descriptors(result2path, sum(FEATURE_DIM+3));
fragment2Keypoints = fragment2_xyz_features(:, 1:3);
fragment2Descriptors = fragment2_xyz_features(:, 4:end);

% Find mutually closest keypoints in 3DMatch feature descriptor space
% fragment2KDT = KDTreeSearcher(fragment2Descriptors);
% fragment1KDT = KDTreeSearcher(fragment1Descriptors);
% fragment1NNIdx = knnsearch(fragment2KDT,fragment1Descriptors);
% fragment2NNIdx = knnsearch(fragment1KDT,fragment2Descriptors);
% fragment2MatchIdx = find((1:size(fragment2NNIdx,1))' == fragment1NNIdx(fragment2NNIdx));
% fragment2MatchKeypoints = fragment2Keypoints(fragment2MatchIdx,:);
% fragment1MatchKeypoints = fragment1Keypoints(fragment2NNIdx(fragment2MatchIdx),:);

% 1NN matching
% [~, matches12] = pdist2(fragment2Descriptors, fragment1Descriptors, 'euclidean', 'smallest', 1);
% matches12 = [1:length(matches12); matches12]';
% 
% [~, matches21] = pdist2(fragment1Descriptors, fragment2Descriptors, 'euclidean', 'smallest', 1);
% matches21 = [matches21; 1:length(matches21)]';
% 
% % union of matches21 & matches12
% matches12 = [matches12; matches21];
% 
% % intersection of matches21 & matches12
% % matches12 = intersect(matches12, matches21, 'rows');

% kNN matching
k = 5;
[~, matches12] = pdist2(fragment2Descriptors, fragment1Descriptors, 'euclidean', 'smallest', k);
matches12 = reshape(matches12', [], 1);  % Nxk -> kNx1
matches12 = [repmat([1:length(fragment1Keypoints)]', k, 1), matches12];

[~, matches21] = pdist2(fragment1Descriptors, fragment2Descriptors, 'euclidean', 'smallest', k);
matches21 = reshape(matches21', [], 1);  % Nxk -> kNx1
matches21 = [matches21, repmat([1:length(fragment2Keypoints)]', k, 1)];

% union of matches21 & matches12
% matches12 = union(matches12, matches21, 'rows');
matches12 = [matches12; matches21];

% intersection of matches21 & matches12
% matches12 = intersect(matches12, matches21, 'rows');


fragment1MatchKeypoints = fragment1Keypoints(matches12(:,1), :);
fragment2MatchKeypoints = fragment2Keypoints(matches12(:,2), :);

% Estimate initial transformation with RANSAC to align fragment 2 keypoints to fragment 1 keypoints
try
    [estimateRt,inlierIdx] = ransacfitRt([fragment1MatchKeypoints';fragment2MatchKeypoints'], inlierThreshold, 0);
    estimateRt = [estimateRt;[0,0,0,1]];
catch
    fprintf('Error: not enough mutually matching keypoints!\n');
    estimateRt = eye(4);
    inlierIdx = [];
end

% % Refine rigid transformation with ICP
% if useGPU
%     [tform,movingReg,icpRmse] = pcregrigidGPU(pcdownsample(pointCloud(fragment1Points'),'gridAverage',0.01),pcdownsample(pointCloud(fragment2Points'),'gridAverage',0.01),'InlierRatio',0.3,'Verbose',true,'Tolerance',[0.0001,0.00009],'Extrapolate',true,'MaxIterations',50);
% else
%     [tform,movingReg,icpRmse] = pcregrigid(pcdownsample(pointCloud(fragment1Points'),'gridAverage',0.01),pcdownsample(pointCloud(fragment2Points'),'gridAverage',0.01),'InlierRatio',0.3,'Verbose',false,'Tolerance',[0.0001,0.00009],'Extrapolate',true,'MaxIterations',50); %,
% end
% icpRt = inv(tform.T');
% fragment2Points = icpRt(1:3,1:3) * fragment2Points + repmat(icpRt(1:3,4),1,size(fragment2Points,2));
% estimateRt = icpRt * estimateRt;

fragment2Points = estimateRt(1:3,1:3) * fragment2PointCloud.Location' + repmat(estimateRt(1:3,4),1,size(fragment2PointCloud.Location',2));
fragment1Points = fragment1PointCloud.Location';

% resultsPath = fullfile(intermPath,sprintf('%s-registration-results',descriptorName));
% 
% if ~exist(resultsPath)
%     mkdir(resultsPath);
% end

% % Visualize alignment result
% % results
% figure(1); clf
% Utils.pcshow_matches(fragment1PointCloud.Location, fragment2PointCloud.Location, ...
%     fragment1Keypoints, fragment2Keypoints, ...
%     matches12, 'inlierIdx', inlierIdx, 'k', 1000);

figure(1); clf
Utils.pcshow_matches(fragment1PointCloud.Location, fragment2PointCloud.Location, ...
    fragment1Keypoints, fragment2Keypoints, ...
    matches12(inlierIdx, :), 'k', 1000);


% Show alignment
figure(2); clf
Utils.pcshow_multiple({fragment1PointCloud.Location, fragment2PointCloud.Location}, {eye(4), estimateRt});
title('Alignment')

                         
% Compute alignment percentage
ratioAligned = zeros(1,2);
[nnIdx,sqrDists] = multiQueryKNNSearchImpl(pointCloud(fragment2Points'),fragment1Points',1);
dists = sqrt(sqrDists);
ratioAligned(1) = sum(dists < 0.1)/size(fragment1Points,2); % relative overlap on first fragment
[nnIdx,sqrDists] = multiQueryKNNSearchImpl(pointCloud(fragment1Points'),fragment2Points',1);
dists = sqrt(sqrDists);
ratioAligned(2) = sum(dists < 0.1)/size(fragment2Points,2); % relative overlap on second fragment

% Compute several additional heuristics for loop closure detection
numInliers = length(inlierIdx);
inlierRatio = numInliers/size(matches12, 1);


fprintf('%d %f %f %f\n',numInliers,inlierRatio,ratioAligned(1),ratioAligned(2));