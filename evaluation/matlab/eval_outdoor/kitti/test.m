root = '/ssd/dataset/kitti-reg-test';
test_dataset = read_txts(root);

i = 20;
seq = int64(test_dataset(i, 1));
anc_idx = int64(test_dataset(i, 2));
pos_idx = int64(test_dataset(i, 3));

anc_pc_path = fullfile(root, sprintf('%02d', seq), sprintf('%06d.bin', anc_idx));
anc_pc = Utils.loadPointCloud(anc_pc_path, 3);

pos_pc_path = fullfile(root, sprintf('%02d', seq), sprintf('%06d.bin', pos_idx));
pos_pc = Utils.loadPointCloud(pos_pc_path, 3);

t = test_dataset(i, 4:6);
q = test_dataset(i, 7:10);
R = quat2rotm(q);
T = [[R, t']; [0, 0, 0, 1]];

% pos_pc_transformed = hom2cart((T * cart2hom(pos_pc)')');
% % pos_pc_transformed = Utils.transform(pos_pc, q, t, 1);
% 
% figure(1)
% ax = pcshow(anc_pc, 'r');
% hold on;
% ax = pcshow(pos_pc_transformed, 'b');
% hold off;

anc_pc_transformed = hom2cart((cart2hom(anc_pc) * inv(T)'));
figure(1)
ax = pcshow(anc_pc_transformed, 'r');
hold on;
ax = pcshow(pos_pc, 'b');
hold off;