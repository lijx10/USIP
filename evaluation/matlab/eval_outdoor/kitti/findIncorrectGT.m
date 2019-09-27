root = '/ssd/dataset/kitti-reg-test';
test_dataset = read_txts_kitti(root);

invalid_gt_idx = [];
parfor i = 1:1:size(test_dataset, 1)
%     i = 2359;
    
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
    
    [t_estimated, anc_pc_reg, rmse] = pcregrigid(anc_pc_sampled, pos_pc_sampled, ...
        'InitialTransform', affine3d(T), ... 
        'InlierRatio', 0.5, ...
        'Verbose', false);

    [delta_t, delta_deg] = Utils.compareTransform(T', t_estimated.T');
    if (delta_t>2 || delta_deg>5) && rmse<2
        fprintf('%d - delta_t %f, delta_deg %f, rmse %f\n', i, delta_t, delta_deg, rmse);
        invalid_gt_idx = [invalid_gt_idx, i];        
    end
    
%     figure(1)
%     ax = pcshow(anc_pc_reg.Location, 'r');
%     hold on;
%     ax = pcshow(pos_pc_sampled.Location, 'b');
%     hold on;
%     anc_pc_transformed = hom2cart((cart2hom(anc_pc_sampled.Location) * T));
%     ax = pcshow(anc_pc_transformed, 'g');
%     hold off;
%     fprintf('\n%d - delta_t %f, delta_deg %f, rmse %f\n', i, delta_t, delta_deg, rmse);
%     break

end

invalid_gt_idx