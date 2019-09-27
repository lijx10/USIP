% Script to evaluate .log files for the geometric registration benchmarks,
% in the same spirit as Choi et al 2015. Please see:
%
% http://redwood-data.org/indoor/regbasic.html
% https://github.com/qianyizh/ElasticReconstruction/tree/master/Matlab_Toolbox

descriptorName = 'my'; % 3dmatch, spin, fpfh

% Locations of evaluation files
dataPath = '/ssd/redwood_data';
gtPath = '/ssd/dataset/redwood/original';

% % Synthetic data benchmark
sceneList = {'livingroom1', 'livingroom2', 'office1', 'office2'};
         
% Load Elastic Reconstruction toolbox
addpath(genpath('external'));

% Compute precision and recall
totalRecall = []; totalPrecision = []; totalInlierNum = []; totalInlierRatio = [];
for sceneIdx = 1:length(sceneList)   
    % Compute registration error
    fprintf(['--- ', sceneList{sceneIdx}, '\n']);
    gt = mrLoadLog(fullfile(gtPath, sprintf('%s-evaluation', sceneList{sceneIdx}), 'gt.log'));
    gt_info = mrLoadInfo(fullfile(gtPath, sprintf('%s-evaluation', sceneList{sceneIdx}), 'gt.info'));

%     result = mrLoadLogMy(fullfile(dataPath,sprintf('%s.log', sceneList{sceneIdx})));
%     [recall,precision, inlierNum, inlierRatio] = mrEvaluateRegistrationMy(result,gt,gt_info);
%     
%     totalRecall = [totalRecall;recall];
%     totalPrecision = [totalPrecision;precision];
%     totalInlierNum = [totalInlierNum;inlierNum];
%     totalInlierRatio = [totalInlierRatio;inlierRatio];
    
    result = mrLoadLog(fullfile(dataPath,sprintf('%s_reg_refine_all.log', sceneList{sceneIdx})));
    [recall,precision] = mrEvaluateRegistration(result,gt,gt_info);
    
    totalRecall = [totalRecall;recall];
    totalPrecision = [totalPrecision;precision];
    totalInlierNum = 0;
    totalInlierRatio = 0;
end
fprintf('===\nMean registration recall: %f precision: %f inlierNum: %d inlierRatio: %f\n',mean(totalRecall),mean(totalPrecision), round(mean(totalInlierNum)), mean(totalInlierRatio));
