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
    
    result_odom = [];
    information_odom = [];
    result_loop = [];
    information_loop = [];
    for i=1:1:length(result)
        seq_array = result(i).info;
        if seq_array(2)-seq_array(1) == 1
            % odometry
            result_odom = [result_odom, struct('info', result(i).info, 'trans', result(i).trans)];
            
            srcIdx = result(i).info(1);
            dstIdx = result(i).info(2);
            estimateRt = result(i).trans;
            information_matrix = computeInformation(pcRoot, resultRoot, sceneList{sceneIdx}, srcIdx, dstIdx, estimateRt, inlierThreshold);
            information_odom = [information_odom, struct('info', result(i).info, 'mat', information_matrix)];
        else
            % loop closure
            result_loop = [result_loop, struct('info', result(i).info, 'trans', result(i).trans)];
            
            srcIdx = result(i).info(1);
            dstIdx = result(i).info(2);
            estimateRt = result(i).trans;
            information_matrix = computeInformation(pcRoot, resultRoot, sceneList{sceneIdx}, srcIdx, dstIdx, estimateRt, inlierThreshold);
            information_loop = [information_loop, struct('info', result(i).info, 'mat', information_matrix)];
        end
    end
    
    % fix missing in odometry data
    gt = mrLoadLog(fullfile(gtPath, sprintf('%s-evaluation', sceneList{sceneIdx}), 'gt.log'));
    gt_info = mrLoadInfo(fullfile(gtPath, sprintf('%s-evaluation', sceneList{sceneIdx}), 'gt.info'));
    for i=1:1:length(gt)
        % replace information matrix wit gt
        %         for j=1:1:length(result_odom)
        %             if isequal(gt(i).info, result_odom(j).info)
        %                 information_odom(j) = gt_info(i);
        %                 break;
        %             end
        %         end
        %         for j=1:1:length(result_loop)
        %             if isequal(gt(i).info, result_loop(j).info)
        %                 information_loop(j) = gt_info(i);
        %                 break;
        %             end
        %         end
        
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
                result_odom = [result_odom, struct('info', gt(i).info, 'trans', gt(i).trans)];
                information_odom = [information_odom, struct('info', gt_info(i).info, 'mat', gt_info(i).mat)];
            end
        end
    end
    
    mrWriteLog(result_odom, fullfile(dataPath,sprintf('%s_odom.log', sceneList{sceneIdx})));
    mrWriteInfo(information_odom, fullfile(dataPath,sprintf('%s_odom.info', sceneList{sceneIdx})));
    mrWriteLog(result_loop, fullfile(dataPath,sprintf('%s_loop.log', sceneList{sceneIdx})));
    mrWriteInfo(information_loop, fullfile(dataPath,sprintf('%s_loop.info', sceneList{sceneIdx})));
    
end

function information_matrix = computeInformation(pcRoot, resultRoot, scene, srcIdx, dstIdx, estimateRt, inlierThreshold)
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


% try using gt Rt


if strcmp(method, 'feature')
    % kNN matching
    k = 1;
    [~, matches12] = pdist2(fragment2Descriptors, fragment1Descriptors, 'euclidean', 'smallest', k);
    matches12 = reshape(matches12', [], 1);  % Nxk -> kNx1
    matches12 = [repmat([1:length(fragment1Keypoints)]', k, 1), matches12];
    
    [~, matches21] = pdist2(fragment1Descriptors, fragment2Descriptors, 'euclidean', 'smallest', k);
    matches21 = reshape(matches21', [], 1);  % Nxk -> kNx1
    matches21 = [matches21, repmat([1:length(fragment2Keypoints)]', k, 1)];
    
    % % union of matches21 & matches12
    % matches12 = union(matches12, matches21, 'rows');
    % matches12 = [matches12; matches21];
    
    % % intersection of matches21 & matches12
    matches12 = intersect(matches12, matches21, 'rows');
    
    fragment1MatchKeypoints = fragment1Keypoints(matches12(:,1), :);
    fragment2MatchKeypoints = fragment2Keypoints(matches12(:,2), :);
    
    fragment2MatchKeypoints_T = (estimateRt(1:3,1:3) * fragment2MatchKeypoints' + repmat(estimateRt(1:3,4),1,size(fragment2MatchKeypoints',2)))';
    dist = vecnorm(fragment1MatchKeypoints - fragment2MatchKeypoints_T, 2, 2);
    inlierIdx = find(dist < inlierThreshold);
    
    if isempty(inlierIdx)
        fprintf('%s - %d - %d\n', scene, srcIdx, dstIdx);
        information_matrix = eye(6);
    else
        % loop all the inlier feature points in fragment1 to get information matrix
        information_matrix = zeros(6, 6);
        parfor i=1:1:length(inlierIdx)
            idx = inlierIdx(i);
            sx = fragment1MatchKeypoints(idx, 1);
            sy = fragment1MatchKeypoints(idx, 2);
            sz = fragment1MatchKeypoints(idx, 3);
            
            A = [1, 0, 0, 0, 2 * sz, - 2 * sy;
                0, 1, 0, - 2 * sz, 0, 2 * sx;
                0, 0, 1, 2 * sy, - 2 * sx, 0;];
            information_matrix = information_matrix + A' * A;
        end
    end
    
    
elseif strcmp(method, 'point')
    % try ICP
    pc1Path = fullfile(pcRoot, scene, sprintf('%d.npy', srcIdx));
    pc2Path = fullfile(pcRoot, scene, sprintf('%d.npy', dstIdx));
    pc1 = readNPY(pc1Path);
    pc2 = readNPY(pc2Path);
    pc1_cloud = pointCloud(pc1(:, 1:3), 'Normal', pc1(:, 4:6));
    pc2_cloud = pointCloud(pc2(:, 1:3), 'Normal', pc2(:, 4:6));
    
    pc1_cloud = pcdownsample(pc1_cloud, 'gridAverage', 0.04);
    pc2_cloud = pcdownsample(pc2_cloud, 'gridAverage', 0.04);
    pc1 = pc1_cloud.Location;
    pc2 = pc2_cloud.Location;
    
    R = eul2rotm(rotm2eul(estimateRt(1:3, 1:3)));
    t = estimateRt(1:3, 4);
    estimateRt = [[R, t];[0,0,0,1]];
    [t_estimated, movingReg, rmse] = pcregrigid(pc2_cloud, pc1_cloud, ...
        'InitialTransform', affine3d(estimateRt'), ...
        'InlierRatio', 0.3, 'Verbose', false);
    
    estimateRt = t_estimated.T';
    
    fragment1Keypoints = pc1(:, 1:3);
    fragment2Keypoints_T = movingReg.Location;
    
%     figure(1)
%     pcshow(fragment1Keypoints)
%     hold on
%     pcshow(fragment2Keypoints_T, 'r')
    
    [I_matches21, D_matches21] = knnsearch(fragment1Keypoints, fragment2Keypoints_T, 'K', 1);
    inlierMask = D_matches21 < inlierThreshold;
    matches21 = I_matches21(inlierMask);
    
    if isempty(matches21)
        fprintf('%s - %d -%d\n', scene, srcIdx, dstIdx);
    end
    
    % loop all the inlier feature points in fragment1 to get information matrix
    information_matrix = zeros(6, 6);
    parfor i=1:1:length(matches21)
        idx = matches21(i);
        sx = fragment1Keypoints(idx, 1);
        sy = fragment1Keypoints(idx, 2);
        sz = fragment1Keypoints(idx, 3);
        
        A = [1, 0, 0, 0, 2 * sz, - 2 * sy;
            0, 1, 0, - 2 * sz, 0, 2 * sx;
            0, 0, 1, 2 * sy, - 2 * sx, 0;];
        information_matrix = information_matrix + A' * A;
    end
    
    
else
    fprintf('wrong method!');
end


end

