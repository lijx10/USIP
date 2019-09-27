function keypoints_sigmas = load_keypoint(filepath, dataset)
keypoints_sigmas = Utils.load_descriptors(filepath, 4);
if strcmp(dataset, 'oxford')
    keypoints_sigmas_converted = keypoints_sigmas;
    keypoints_sigmas_converted(:, 1) = keypoints_sigmas(:, 1);
    keypoints_sigmas_converted(:, 2) = keypoints_sigmas(:, 3);
    keypoints_sigmas_converted(:, 3) = keypoints_sigmas(:, 2) * -1;
    keypoints_sigmas = keypoints_sigmas_converted;
elseif strcmp(dataset, 'kitti')
    seq_str = filepath(end-12:end-11);
    calib = read_kitti_calib(['/ssd/dataset/odometry/calib/', seq_str, '/calib.txt']);
    Tr = calib.Tr;
    keypoints_sigmas(:, 1:3) = cam2velodyne(keypoints_sigmas(:, 1:3), Tr);
end
end