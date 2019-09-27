% Example script to show how to interpret translation T and rotation q
% in our provided groundtruths
%
% We illustrate using test case 26 in our Oxford dataset, which is provided
% in the example data
%
% Author: Zi Jian Yew <zijian.yew@comp.nus.edu.sg>
%

%% Set the model and groundtruth transforms
% The translation T is given by t_1, t_2, t_3
% The Rotation quaternion q is given by q_1, q_2, q_3, q_4
%
model1 = Utils.loadPointCloud('../example_data/oxford_270.bin');
model2 = Utils.loadPointCloud('../example_data/oxford_456.bin');
T = [11.2205663032545, 4.91111082490333, 0.113109519604];
q = [0.949471387446567, -0.00459102089212152, -0.000201595593149864, -0.313819958426285];

%% Plots the point clouds
% Before alignment
figure(1), hold off
pcshow(model1(:, 1:3), [0, 1, 0], 'MarkerSize', 2)
hold on
pcshow(model2(:, 1:3), [1, 0, 0], 'MarkerSize', 2)
title('Before alignment')

% After alignment
R = quat2rotm(q);
transform = [R T'];
model2_transformed = (transform * ...
                      padarray(model2(:,1:3), [0, 1], 1, 'post')')';
figure(2), hold off
pcshow(model1(:, 1:3), [0, 1, 0], 'MarkerSize', 2)
hold on
pcshow(model2_transformed(:, 1:3), [1, 0, 0], 'MarkerSize', 2)
title('After alignment')
