function pc_enu = coord_cam2enu(pc_cam)

pc_enu = pc_cam;
pc_enu(:, 1) = pc_cam(:, 1);
pc_enu(:, 2) = pc_cam(:, 3);
pc_enu(:, 3) = pc_cam(:, 2)*-1;

end