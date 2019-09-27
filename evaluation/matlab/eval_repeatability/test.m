redwood_file = '/ssd/jiaxin/TSF_datasets/redwood/numpy_gt_normal/livingroom1/0.npy';
redwood_pc = readNPY(redwood_file);
redwood_pc = redwood_pc(:, 1:3);

sceneNN_file = '/ssd/jiaxin/TSF_datasets/SceneNN-DS-compact/frames_test/400.npy';
sceneNN_pc = readNPY(sceneNN_file);
sceneNN_pc = sceneNN_pc(:, 1:3);

figure(1);
pcshow(redwood_pc);
xlabel('x');
ylabel('y');
zlabel('z');
figure(2);
pcshow(sceneNN_pc);
xlabel('x');
ylabel('y');
zlabel('z');