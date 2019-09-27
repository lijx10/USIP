sceneList = {'livingroom1', 'livingroom2', 'office1', 'office2'};
numScenes = length(sceneList);

gtPath = '/ssd/dataset/redwood/original';

rot_angles = [];
for sceneIdx = 1:numScenes
    gt_traj = mrLoadLog(fullfile(gtPath, [sceneList{sceneIdx}, '-evaluation'], 'gt.log'));
    for i=1:length(gt_traj)
        T = gt_traj(i).trans;
        R = T(1:3, 1:3);
        axang = rotm2axang(R);  % The default order for Euler angle rotations is "ZYX"
%         rot_angles = [rot_angles; axang];
        
        if abs(axang(1))>abs(axang(2)) || abs(axang(3))>abs(axang(2))
            fprintf('%s - %d, %d - %f, %f, %f, %f\n', sceneList{sceneIdx}, gt_traj(i).info(1), gt_traj(i).info(2), ...
                axang(1), axang(2), axang(3), axang(4))
            
            rot_angles = [rot_angles; axang];
        end
    end
end

rot_angles = abs(rot_angles);
angle_mean = mean(rot_angles, 1)
angle_max = max(rot_angles, [], 1)
angle_min = min(rot_angles, [], 1)