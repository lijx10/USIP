% Script for processing KITTI dataset
%
% Takes pointcloud every METERS_PER_POINT_CLOUD=10m, downsamples it
% using a voxelgrid filter with grid size=0.2m, computes normals and
% saves the resulting pointcloud together with pairwise poses.
%
% Author: Zi Jian Yew <zijian.yew@comp.nus.edu.sg>
%

clear
addpath('common')

KITTI_FOLDER = '/ssd/dataset/odometry';  % Change this to point at the dataset directory
OUTPUT_FOLDER = '/ssd/dataset/kitti-reg-test';
METERS_PER_POINT_CLOUD = 10;

%% Check datasets
datasets = dir(fullfile(KITTI_FOLDER, 'poses'));
datasets = {datasets(endsWith({datasets.name}, '.txt')).name};
for iSet = 1 : length(datasets)
    datasets{iSet} = datasets{iSet}(1:end-4);
end

% datasets: {'00'    '01'    '02'    '03'    '04'    '05'    '06'    '07'
% '08'    '09'    '10'}
datasets

%%
% Load poses
for iSet = 1 : length(datasets)
    dataset = datasets{iSet};
    poses = dlmread(fullfile(KITTI_FOLDER, 'poses',  sprintf('%s.txt', dataset)));
    calib = read_kitti_calib(fullfile(KITTI_FOLDER, 'calib',  dataset, 'calib.txt'));

    poses = reshape(poses, [size(poses, 1), 4, 3]);
    poses = permute(poses, [3,2,1]);
    positions = squeeze(poses(:,4,:))';
    numScans = size(positions, 1);

    distTable = pdist2(positions, positions);
    distTable = triu(distTable);
    iCur = 1;
    scans = zeros(numScans,1);  % 0-based indexing as per kitti
    numScansRetained = 1;

    while iCur <= numScans
        iCur = find(distTable(iCur, :) > METERS_PER_POINT_CLOUD, 1, 'first');
        iCur = iCur - 1;
        if isempty(iCur)
            break
        end
        numScansRetained = numScansRetained + 1;
        scans(numScansRetained) = iCur-1;
    end
    scans = scans(1:numScansRetained);

    %%
    srcFolder = fullfile(KITTI_FOLDER, 'data_odometry_velodyne', 'sequences', dataset, 'velodyne');
    dstFolder = fullfile(OUTPUT_FOLDER, dataset);
    fprintf('Source folder: %s, dstFolder: %s\n', srcFolder, dstFolder);
    mkdir(dstFolder);

    %% Pairs
    positionsFiltered = positions(scans+1, :);
    distTableFiltered = triu(pdist2(positionsFiltered, positionsFiltered));
    [r, c] = ind2sub(size(distTableFiltered), find(distTableFiltered > 0 & distTableFiltered < 10));

    pairsIdx = [scans(r) scans(c)];
    groundtruths = zeros(size(pairsIdx, 1), 9);
    for iPair = 1 : size(pairsIdx, 1)

        a = pairsIdx(iPair, 1);
        b = pairsIdx(iPair, 2);
        pose1 = poses2velo(poses(:,:,a+1), calib);
        pose2 = poses2velo(poses(:,:,b+1), calib);
        transform_12 = pose1 \ pose2;  % Multiply points in 2 by this transform to get 1

        q = rotm2quat(transform_12(1:3, 1:3));
        t = transform_12(1:3, 4);

        groundtruths(iPair, :) = [a b t' q];
    end

    groundtruths = array2table(groundtruths, 'VariableNames', ...
        {'idx1', 'idx2', 't_1', 't_2', 't_3', 'q_1', 'q_2', 'q_3', 'q_4'});
    writetable(groundtruths, fullfile(dstFolder, 'groundtruths.txt'), 'delimiter', '\t');

    %% Generate the scans
    for iBin = 1 : length(scans)
        %%
        idx_6 = sprintf('%06i', scans(iBin));
        binFname = sprintf('%s.bin', idx_6);
        srcBin = fullfile(srcFolder, binFname);
        dstBin = fullfile(dstFolder, binFname);
        xyzi = Utils.loadPointCloud(srcBin, 4);
        
        normals = findPointNormals(xyzi(:,1:3), 9, [0, 0, 1]);
        pc = pointCloud(xyzi(:,1:3), 'normal', normals);
        %%

        pc = pcdownsample(pc, 'gridAverage', 0.2);

        %%
        xyz = pc.Location;    
        normals = pc.Normal;
        xyzn = [xyz, normals];
        fid = fopen(dstBin, 'w');
        fwrite(fid, xyzn', 'float');
        fclose(fid);

        fprintf('Processing %i out of %i\n', iBin, length(scans));
    end
end


%%

function poses_velo = poses2velo(poses_cam0, calib)
% Transform poses in cam0 frame to velodyne frame

if all(size(poses_cam0) == [3, 4])
    poses_cam0 = [poses_cam0; 0 0 0 1];
end

poses_velo = zeros(size(poses_cam0));
Tr = calib.Tr;
TrI = [Tr(1:3,1:3)', -Tr(1:3,1:3)'*Tr(1:3,4); 0 0 0 1];

for i = 1 : size(poses_cam0, 3)
    poses_velo(:,:,i) = TrI * poses_cam0(:,:,i) * Tr;
end
end

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