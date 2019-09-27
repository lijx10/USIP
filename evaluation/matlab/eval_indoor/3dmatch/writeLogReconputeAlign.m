% Script to run create evaluation .log files from predicted rigid
% transformations for the geometric registration benchmark
%
% ---------------------------------------------------------
% Copyright (c) 2016, Andy Zeng
%
% This file is part of the 3DMatch Toolbox and is available
% under the terms of the Simplified BSD License provided in
% LICENSE. Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% Configuration options (change me)
descriptorName = 'my';
pcRoot = '/ssd/dataset/redwood/numpy_gt_normal';
dataPath = '/ssd/dataset/redwood/numpy_gt_normal'; % Location of scene files
txtPath = '/ssd/redwood_data/my-txt'; % Location of txt files
savePath = '/ssd/redwood_data'; % Location to save evaluation .log file
% % Synthetic data benchmark
sceneList = {'livingroom1', 'livingroom2', 'office1', 'office2'};

totalRecall = []; totalPrecision = [];
for sceneIdx = 1:length(sceneList)
    
    % List fragment files
    scenePath = fullfile(dataPath,sceneList{sceneIdx});
    sceneDir = dir(fullfile(scenePath,'*.npy'));
    numFragments = length(sceneDir);
    fprintf('%s\n',scenePath);
    
    % Loop through registration results and write a log file
    valid_mask = zeros(numFragments, numFragments);
    parfor fragment1Idx = 0:numFragments-1
        for fragment2Idx = (fragment1Idx+1):numFragments-1
            fragment1Name = sprintf('%d',fragment1Idx);
            fragment2Name = sprintf('%d',fragment2Idx);
            
            resultPath = fullfile(txtPath, sceneList{sceneIdx}, sprintf('%s-%s.rt.txt',fragment1Name,fragment2Name));
            
            try
                resultRt = dlmread(resultPath,'\t',[2,0,5,3]);
            catch
                continue
            end
            resultNumInliers = dlmread(resultPath,'\t',[1,0,1,0]);
            resultInlierRatio = dlmread(resultPath,'\t',[1,1,1,1]);
            resultAlignRatio = dlmread(resultPath,'\t',[1,2,1,3]);
            resultInformation = dlmread(resultPath, '\t', [6,0,11,5]);
            
            
            
            % try ICP
            inlierThreshold = 0.05;
            estimateRt = resultRt;
            
            pc1Path = fullfile(pcRoot, sceneList{sceneIdx}, sprintf('%s.npy', fragment1Name));
            pc2Path = fullfile(pcRoot, sceneList{sceneIdx}, sprintf('%s.npy', fragment2Name));
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
            
            resultAlignRatio(1) = sum(inlierMask) / length(fragment1Keypoints);
            resultAlignRatio(2) = sum(inlierMask) / length(fragment2Keypoints_T);
            
            
            % Check if surface overlap is above some threshold
            % Note: set threshold to 0.23 to reproduce our numbers
            % for the synthetic benchmark from Choi et al.
            if resultAlignRatio(1) > 0.15 && resultInlierRatio>0.025
                tmp_mask = zeros(numFragments, numFragments)
                tmp_mask(fragment1Idx+1, fragment2Idx+1) = 1;
                valid_mask = valid_mask + tmp_mask;
            end
        end
    end
    
    
    % Loop through registration results and write a log file
    logPath = fullfile(savePath, sprintf('%s.log',sceneList{sceneIdx}));
    fid = fopen(logPath,'w');
    for fragment1Idx = 0:numFragments-1
        for fragment2Idx = (fragment1Idx+1):numFragments-1
            if valid_mask(fragment1Idx+1, fragment2Idx+1) == 1
                fragment1Name = sprintf('%d',fragment1Idx);
                fragment2Name = sprintf('%d',fragment2Idx);
                
                resultPath = fullfile(txtPath, sceneList{sceneIdx}, sprintf('%s-%s.rt.txt',fragment1Name,fragment2Name));
                
                try
                    resultRt = dlmread(resultPath,'\t',[2,0,5,3]);
                catch
                    continue
                end
                resultNumInliers = dlmread(resultPath,'\t',[1,0,1,0]);
                resultInlierRatio = dlmread(resultPath,'\t',[1,1,1,1]);
                resultAlignRatio = dlmread(resultPath,'\t',[1,2,1,3]);
                resultInformation = dlmread(resultPath, '\t', [6,0,11,5]);
                
                fprintf(fid,'%d\t %d\t %d\t\n',fragment1Idx,fragment2Idx,numFragments);
                fprintf(fid,'%.10f\t%.10f\t%.10f\t%.10f\n',resultRt');
                fprintf(fid, '%d\t%f\n', resultNumInliers, resultInlierRatio);
                fprintf(fid,'%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\n',resultInformation);
            end
        end
    end
    fclose(fid);
end
