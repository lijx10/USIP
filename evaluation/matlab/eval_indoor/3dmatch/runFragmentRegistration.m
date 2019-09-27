% Script to run RANSAC over keypoints and descriptors to predict rigid
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

% Location of scene files (change this and the parameters in clusterCallback.m)
pcRoot = '/ssd/dataset/redwood/numpy_gt_normal';
resultRoot = '/ssd/redwood_data/1024k1k5-512s512-0.75m448-0.767';
outputTxtPath = '/ssd/redwood_data/my-txt';
sceneList = {'livingroom1', 'livingroom2', 'office1', 'office2'};
numScenes = length(sceneList);

% Get fragment pairs
fragmentPairs = {};
fragmentPairIdx = 1;
for sceneIdx = 1:numScenes
    numFragments = length(dir(fullfile(pcRoot, sceneList{sceneIdx}))) - 2;
    for fragment1Idx = 0:numFragments-1
        for fragment2Idx = (fragment1Idx+1):numFragments-1
            fragmentPairs{fragmentPairIdx,1} = sceneList{sceneIdx};
            fragmentPairs{fragmentPairIdx,2} = fragment1Idx;
            fragmentPairs{fragmentPairIdx,3} = fragment2Idx;
            fragmentPairIdx = fragmentPairIdx + 1;
        end
    end
end

% evaluate only the overlapped pairs
% gtPath = '/ssd/dataset/redwood/original';
% 
% fragmentPairs = {};
% fragmentPairIdx = 1;
% for sceneIdx = 1:numScenes
%     gt_traj = mrLoadLog(fullfile(gtPath, sprintf('%s-evaluation', sceneList{sceneIdx}), 'gt.log'));
%     
%     for i=1:1:length(gt_traj)
%         fragmentPairs{fragmentPairIdx,1} = sceneList{sceneIdx};
%         fragmentPairs{fragmentPairIdx,2} = gt_traj(i).info(1);
%         fragmentPairs{fragmentPairIdx,3} = gt_traj(i).info(2);
%         fragmentPairIdx = fragmentPairIdx + 1;
%     end
% end

% Run registration for all fragment pairs
parfor fragmentPairIdx = 1:size(fragmentPairs,1)    
    clusterCallback(pcRoot, resultRoot, outputTxtPath, ...
                    fragmentPairs{fragmentPairIdx,1}, ...
                    fragmentPairs{fragmentPairIdx,2}, ...
                    fragmentPairs{fragmentPairIdx,3});
end