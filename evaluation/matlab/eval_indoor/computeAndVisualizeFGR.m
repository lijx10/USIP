pcRoot = '/ssd/dataset/redwood/numpy_gt_normal';
resultRoot = '/ssd/redwood_data/my';
sceneName = 'livingroom1';  % {'livingroom1', 'livingroom2', 'office1', 'office2'};
fragment1Idx = 1;
fragment2Idx = 8;
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


% FGR, Fast Global Registration
estimateRt = fast_global_registration(fragment1Keypoints, fragment1Descriptors, fragment2Keypoints, fragment2Descriptors); 


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
numInliers = 1;
inlierRatio = 0.99;


fprintf('%d %f %f %f\n',numInliers,inlierRatio,ratioAligned(1),ratioAligned(2));