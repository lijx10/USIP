function calib_params = read_kitti_calib(fname)
% Read calibration file for KITTI odometry dataset

fid = fopen(fname);
param = textscan(fid, '%s %f %f %f %f %f %f %f %f %f %f %f %f', 5);
values = cat(2, param{2:end});


for idx = 1:5
    name = param{1}{idx}(1:end-1);
    P = [reshape(values(idx,:), [4,3])'; 0 0 0 1];
    
    calib_params.(name) = P;
end
fclose(fid);
end