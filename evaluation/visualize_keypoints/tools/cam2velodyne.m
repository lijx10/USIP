function pc_velodyne = cam2velodyne(pc_cam, Tr)
% pc_cam: Nx3, Tr: 4x4

Tr_inv = inv(Tr);
pc_cam_homo = cart2hom(pc_cam);  % Nx4
pc_velodyne_homo = (Tr_inv * pc_cam_homo')';  % 4x4 * 4xN -> 4xN -> Nx4
pc_velodyne = hom2cart(pc_velodyne_homo);

end

