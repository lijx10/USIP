close all
clear all

root = '/ssd/dataset/kitti-reg-test';
test_dataset = read_txts_kitti(root);

load('kitti/invalid_gt_idx.mat');
invalid_gt_idx = [invalid_gt_idx, [1424, 1422, 1421, 1419, 1389, 1383, 1374, ...
                                  258, 259, 2370, 1385, 1386, 1388, 1391, ... 
                                  1392, 1394, 1395, 1397, 1398, 1380, 1382, 1371 ...
                                  1379, 235]];
for i = 1:1:size(test_dataset, 1)
%     i = 258;
    
    if any(invalid_gt_idx == i)
        seq = int64(test_dataset(i, 1));
        anc_idx = int64(test_dataset(i, 2));
        pos_idx = int64(test_dataset(i, 3));
        
        anc_pc_path = fullfile(root, sprintf('%02d', seq), sprintf('%06d.bin', anc_idx));
        anc_pc = Utils.loadPointCloud(anc_pc_path, 6);
        anc_pc = anc_pc(:, 1:3);
        
        pos_pc_path = fullfile(root, sprintf('%02d', seq), sprintf('%06d.bin', pos_idx));
        pos_pc = Utils.loadPointCloud(pos_pc_path, 6);
        pos_pc = pos_pc(:, 1:3);
        
        t = test_dataset(i, 4:6);
        q = test_dataset(i, 7:10);
        R = quat2rotm(q);
        T = [[R, t']; [0, 0, 0, 1]];
        T = inv(T)';
        
        anc_pc_sampled = pcdownsample(pointCloud(anc_pc), 'gridAverage', 0.4);
        pos_pc_sampled = pcdownsample(pointCloud(pos_pc), 'gridAverage', 0.4);

        inlierRatio = 0.95;
        if i==165
            inlierRatio = 0.6;
        end
        if i==166
            inlierRatio = 0.8;
        end
        if i==2371 || i==2367
            inlierRatio = 1;
        end
        [t_estimated, anc_pc_reg, rmse] = pcregrigid(anc_pc_sampled, pos_pc_sampled, ...
            'InitialTransform', affine3d(T), ...
            'InlierRatio', inlierRatio, ...
            'Verbose', false);
        
        [delta_t, delta_deg] = Utils.compareTransform(T', t_estimated.T');
%         if (delta_t>2 || delta_deg>5) && rmse<2
        if 1
            fprintf('%d - delta_t %f, delta_deg %f, rmse %f\n', i, delta_t, delta_deg, rmse);
%             invalid_gt_idx = [invalid_gt_idx, i];
            
            % save the correct T to txt
            T_correct = inv(t_estimated.T');
            R_correct = rotm2quat(T_correct(1:3, 1:3));
            t_correct = T_correct(1:3, 4);
            test_dataset(i, 4:6) = t_correct;
            test_dataset(i, 7:10) = R_correct;
            
        end
        
%         figure(1)
%         ax = pcshow(anc_pc_reg.Location, 'r');
%         hold on;
%         ax = pcshow(pos_pc_sampled.Location, 'b');
%         hold on;
%         anc_pc_transformed = hom2cart((cart2hom(anc_pc_sampled.Location) * T));
%         ax = pcshow(anc_pc_transformed, 'g');
%         hold off;
%         fprintf('\n%d - delta_t %f, delta_deg %f, rmse %f\n', i, delta_t, delta_deg, rmse);
%         break
    end
    
end

% save correct_gt.txt
root = '/ssd/dataset/kitti-reg-test';

gt_seq = {};
for i = 1:1:11
    gt_seq{i} = [];
end
for i = 1:1:size(test_dataset, 1)
    seq = int64(test_dataset(i, 1));
    gt_seq{seq+1} = [gt_seq{seq+1}; test_dataset(i, 2:end)];
end
for i = 1:1:11
    seq_str = sprintf('%02d', i-1);
    output_path = fullfile(root, seq_str, 'correct_gt.txt');
    
    gt = gt_seq{i};
    fileID = fopen(output_path, 'w');
    fprintf(fileID, 'idx1	idx2	t_1	t_2	t_3	q_1	q_2	q_3	q_4\n');
    fclose(fileID);
    dlmwrite(output_path, gt, '-append');
end