function dataset = build_redwood_dataset(dataset_path)
sceneList = {'livingroom1', 'livingroom2', 'office1', 'office2'};

dataset = {};
counter = 1;
for sceneIdx = 1:length(sceneList)   
    scene = sceneList{sceneIdx};
    gt = mrLoadLog(fullfile(dataset_path, 'original', sprintf('%s-evaluation', scene), 'gt.log'));
    
    for i=1:length(gt)
        anc_idx = gt(i).info(1);
        pos_idx = gt(i).info(2);
        T = gt(i).trans;
        % anc_npy_path = fullfile(dataset_path, 'numpy_gt_normal', scene, sprintf('%d.npy', anc_idx));
        % pos_npy_path = fullfile(dataset_path, 'numpy_gt_normal', scene, sprintf('%d.npy', pos_idx));
        
        dataset{counter, 1} = scene;
        dataset{counter, 2} = anc_idx;
        dataset{counter, 3} = pos_idx;
        dataset{counter, 4} = T;
        counter = counter + 1;
    end

end
end