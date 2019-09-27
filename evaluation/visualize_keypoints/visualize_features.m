close all;
clear all;

% dataset = 'modelnet';
% sigma_threshold = 0.04;
% verticalAxis= 'Y';
% verticalAxisDir = 'Up';

% dataset = 'oxford';
% sigma_threshold = 0.25;
% verticalAxis= 'Y';
% verticalAxisDir = 'Down';

dataset = 'kitti';
sigma_threshold = 0.3;
verticalAxis= 'Y';
verticalAxisDir = 'Down';

% dataset = 'redwood';
% sigma_threshold = 0.08;
% verticalAxis= 'Y';
% verticalAxisDir = 'Down';

% build {pc_file, keypoint_file}
pc_keypoint_pair = build_filepath(dataset);

random_idx = randperm(length(pc_keypoint_pair));
for j=1:1:length(random_idx)
    i = random_idx(j);
%     i=916;
    
    pair = pc_keypoint_pair{i};
    
    pc = load_pc(pair{1});
    keypoints_sigmas = load_keypoint(pair{2}, dataset);
    
    valid_idx = find(keypoints_sigmas(:, 4) < sigma_threshold);
    keypoints = keypoints_sigmas(valid_idx, 1:3);
    
    sigmas = keypoints_sigmas(valid_idx, 4);
    sigmas_max = mean(sigmas) * 2;
    sigmas_normalized = sigmas .* (sigmas < sigmas_max) + sigmas_max * (sigmas > sigmas_max);
    sigmas_normalized = sigmas_normalized / sigmas_max;
    sigmas_color = [1-sigmas_normalized, zeros(length(sigmas_normalized), 2)];
    
    
    % rotate point clouds and keypoints
%     R = angle2dcm(pi/3, pi/3, pi/3);
%     pc(:, 1:3) = pc(:, 1:3) * R;
%     keypoints = keypoints * R;
    
    fprintf('%s - i%d - n%d - %s, %s\n', dataset, i, length(valid_idx), pair{1}, pair{2});
%     ax = pcshow(pc(:,1:3), [0.75, 0.75, 0.75]); hold on;
    ax = pcshow(pc(:,1:3), 'VerticalAxis', verticalAxis, 'VerticalAxisDir', verticalAxisDir, 'MarkerSize', 12); hold on;
    scatter3(keypoints(:,1), keypoints(:,2), keypoints(:,3), 100, 'r.')
%     scatter3(keypoints(:,1), keypoints(:,2), keypoints(:,3), 600, sigmas_color, '.')
    hold off;
    axis off;
    set(gcf,'color','w');
    waitfor(ax)
    
    
end