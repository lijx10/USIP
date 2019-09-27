function dataset = build_3dmatch_dataset(dataset_path)
sceneList = {'7-scenes-redkitchen', 'sun3d-home_at-home_at_scan1_2013_jan_1', 
              'sun3d-home_md-home_md_scan9_2012_sep_30', 'sun3d-hotel_uc-scan3',
              'sun3d-hotel_umd-maryland_hotel1', 'sun3d-hotel_umd-maryland_hotel3', 
              'sun3d-mit_76_studyroom-76-1studyroom2', 'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'};

dataset = {};
counter = 1;
for sceneIdx = 1:length(sceneList)   
    scene = sceneList{sceneIdx};
    gt = mrLoadLog(fullfile(dataset_path, sprintf('%s-evaluation', scene), 'gt.log'));
    
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