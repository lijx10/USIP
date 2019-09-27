% ---------------------------------------------------------
% Copyright (c) 2016, Andy Zeng
% 
% This file is part of the 3DMatch Toolbox and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

function clusterCallback(pcRoot, resultRoot, outputTxtPath, sceneName, fragment1Idx, fragment2Idx)
    FEATURE_DIM = 128;
    fprintf('Registering %s - %d and %d: ',sceneName,fragment1Idx,fragment2Idx);

    % Get results file
    resultPath = fullfile(outputTxtPath, sceneName, sprintf('%s-registration-results'), ...
        sprintf('%d-%d.rt.txt',fragment1Idx,fragment2Idx));
%     if exist(resultPath,'file')
%         fprintf('\n');
%         return;
%     end

    % Compute rigid transformation that aligns fragment 2 to fragment 1
    pc1Path = fullfile(pcRoot, sceneName, sprintf('%d.npy', fragment1Idx));
    pc2Path = fullfile(pcRoot, sceneName, sprintf('%d.npy', fragment2Idx));
    fragment1Path = fullfile(resultRoot, sceneName, sprintf('%d.bin', fragment1Idx));
    fragment2Path = fullfile(resultRoot, sceneName, sprintf('%d.bin', fragment2Idx));
    [estimateRt,numInliers,inlierRatio,ratioAligned, information_matrix] = register2Fragments(pc1Path, pc2Path, fragment1Path, fragment2Path, FEATURE_DIM);
    fprintf('%d %f %f %f\n',numInliers,inlierRatio,ratioAligned(1),ratioAligned(2));

    % Save rigid transformation
    fid = fopen(resultPath,'w');
    fprintf(fid,'%d\t %d\t\n%d\t %15.8e\t %15.8e\t %15.8e\t\n', fragment1Idx,fragment2Idx,numInliers,inlierRatio,ratioAligned(1),ratioAligned(2));
    fprintf(fid,'%15.8e\t %15.8e\t %15.8e\t %15.8e\t\n',estimateRt');
    fprintf(fid,'%15.8e\t %15.8e\t %15.8e\t %15.8e\t %15.8e\t %15.8e\t\n',information_matrix);
    fclose(fid);
end