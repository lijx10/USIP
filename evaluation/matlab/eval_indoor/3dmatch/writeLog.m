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
dataPath = '/ssd/dataset/redwood/numpy_gt_normal'; % Location of scene files
txtPath = '/ssd/redwood_data/my-txt'; % Location of txt files
savePath = '/ssd/redwood_data'; % Location to save evaluation .log file
% % Synthetic data benchmark
sceneList = {'livingroom1', 'livingroom2', 'office1', 'office2'};

totalRecall = []; totalPrecision = [];
parfor sceneIdx = 1:length(sceneList)
    
    % List fragment files
    scenePath = fullfile(dataPath,sceneList{sceneIdx});
    sceneDir = dir(fullfile(scenePath,'*.npy'));
    numFragments = length(sceneDir);
    fprintf('%s\n',scenePath);

    % Loop through registration results and write a log file
    logPath = fullfile(savePath, sprintf('%s.log',sceneList{sceneIdx}));
    fid = fopen(logPath,'w');
    for fragment1Idx = 0:numFragments-1
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
            
            % Check if surface overlap is above some threshold
            % Note: set threshold to 0.23 to reproduce our numbers 
            % for the synthetic benchmark from Choi et al.
            if resultAlignRatio(1) > 0.23 && resultInlierRatio>0.025
                fprintf(fid,'%d\t %d\t %d\t\n',fragment1Idx,fragment2Idx,numFragments);
                fprintf(fid,'%.10f\t%.10f\t%.10f\t%.10f\n',resultRt');
                fprintf(fid, '%d\t%f\n', resultNumInliers, resultInlierRatio);
                fprintf(fid,'%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\n',resultInformation);
            end
        end
    end
    fclose(fid);
    
end
