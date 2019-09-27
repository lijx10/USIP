clear all

% % Synthetic data benchmark
pcRoot = '/ssd/dataset/redwood/numpy_gt_normal';
resultRoot = '/ssd/redwood_data/1024k1k5-512s512-0.75m448-0.767';
sceneList = {'livingroom1', 'livingroom2', 'office1', 'office2'};
inlierThreshold = 0.05;

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
            information_odom = [information_odom, struct('info', result(i).info, 'mat', result(i).information)];
        else
            % loop closure
            result_loop = [result_loop, struct('info', result(i).info, 'trans', result(i).trans)];
            information_loop = [information_loop, struct('info', result(i).info, 'mat', result(i).information)];
        end
    end
    
    % fix missing in odometry data
    gt = mrLoadLog(fullfile(gtPath, sprintf('%s-evaluation', sceneList{sceneIdx}), 'gt.log'));
    gt_info = mrLoadInfo(fullfile(gtPath, sprintf('%s-evaluation', sceneList{sceneIdx}), 'gt.info'));
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

