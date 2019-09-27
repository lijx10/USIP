close all;
clear all;

markerSize = 12;
keypointSize = 400;

% dataset = 'modelnet';
% sigma_threshold = 0.04;
% verticalAxis= 'Y';
% verticalAxisDir = 'Up';
% keypointSize = 600;
% classes = {};

% dataset = 'oxford';
% sigma_threshold = 0.25;
% verticalAxis= 'Y';
% verticalAxisDir = 'Down';
% markerSize = 6;

dataset = 'kitti';
sigma_threshold = 0.3;
verticalAxis= 'Y';
verticalAxisDir = 'Down';
fig_size = [10, 8];
keypointSize = 800;

% dataset = 'redwood';
% sigma_threshold = 0.08;
% verticalAxis= 'Y';
% verticalAxisDir = 'Down';

% build {pc_file, keypoint_file}
pc_keypoint_pair = build_filepath(dataset);

random_idx = randperm(length(pc_keypoint_pair));
for j=1:1:length(random_idx)
    i = random_idx(j);
%     i=53;
    
    pair = pc_keypoint_pair{i};
    
    pc = load_pc(pair{1});
    keypoints_sigmas = load_keypoint(pair{2}, dataset);
    
    % radius threshold
    
    
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
    ax = pcshow(pc(:,1:3), 'VerticalAxis', verticalAxis, 'VerticalAxisDir', verticalAxisDir, 'MarkerSize', markerSize); hold on;
    scatter3(keypoints(:,1), keypoints(:,2), keypoints(:,3), keypointSize, 'r.')
%     scatter3(keypoints(:,1), keypoints(:,2), keypoints(:,3), keypointSize, sigmas_color, '.')
    hold off;
    axis off;
    
    set(gcf, 'Units', 'Inches', 'Position', [0, 0, fig_size(1), fig_size(2)], 'PaperUnits', 'Inches', 'PaperSize', [fig_size(1), fig_size(2)])
    
    set(gcf,'color','w');
    
    
    set(gcf, 'Visible', 'off');
    
    
    
    if strcmp(dataset, 'modelnet')==0
        filename = fullfile('auto_imgs', dataset, [num2str(j), '.png']);
        saveas(gcf, filename, 'png');
        imwrite(crop_img(imread(filename)), filename);
        if j>=40
            break
        end
    end
    
    if strcmp(dataset, 'modelnet')
        pc_path_split = strsplit(pair{1}, '/');
        class = {pc_path_split{5}};
        
        if any(cellfun(@(x) strcmp(x, class{1}), classes, 'UniformOutput', true))==0
            fprintf('%s\n', class{1});
            filename = fullfile('auto_imgs', dataset, [num2str(length(classes)+1), '.png']);
            saveas(gcf, filename, 'png');
            imwrite(crop_img(imread(filename)), filename);
        end
        classes = union(classes, class);
        
        if length(classes) == 40
            break;
        end
    end
    
%     waitfor(ax)
    
end