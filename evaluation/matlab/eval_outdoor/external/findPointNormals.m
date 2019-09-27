function [ normals, curvature, normalized_curvature ] = findPointNormals(points, numNeighbours, viewPoint, dirLargest)
%FINDPOINTNORMALS Estimates the normals of a sparse set of n 3d points by
% using a set of the closest neighbours to approximate a plane.
%
%   Required Inputs:
%   points- nx3 set of 3d points (x,y,z)
%
%   Optional Inputs: (will give default values on empty array [])
%   numNeighbours- number of neighbouring points to use in plane fitting
%       (default 9)
%   viewPoint- location all normals will point towards (default [0,0,0])
%   dirLargest- use only the largest component of the normal in determining
%       its direction wrt the viewPoint (generally provides a more stable
%       estimation of planes near the viewPoint, default true)
%
%   Outputs:
%   normals- nx3 set of normals (nx,ny,nz)
%   curvature- nx1 set giving the curvature
%
%   References-
%   The implementation closely follows the method given at
%   http://pointclouds.org/documentation/tutorials/normal_estimation.php
%   This code was used in generating the results for the journal paper
%   Multi-modal sensor calibration using a gradient orientation measure 
%   http://www.zjtaylor.com/welcome/download_pdf?pdf=JFR2013.pdf
%
%   This code was written by Zachary Taylor
%   zacharyjeremytaylor@gmail.com
%   http://www.zjtaylor.com

%% check inputs
validateattributes(points, {'numeric'},{'ncols',3});

if(nargin < 2)
    numNeighbours = [];
end
if(isempty(numNeighbours))
    numNeighbours = 9;
else
    validateattributes(numNeighbours, {'numeric'},{'scalar','positive'});
    if(numNeighbours > 100)
        warning(['%i neighbouring points will be used in plane'...
            ' estimation, expect long run times, large ram usage and'...
            ' poor results near edges'],numNeighbours);
    end
end

if(nargin < 3)
    viewPoint = [];
end
if(isempty(viewPoint))
    viewPoint = [0,0,0];
else
    validateattributes(viewPoint, {'numeric'},{'size',[1,3]});
end

if(nargin < 4)
    dirLargest = [];
end
if(isempty(dirLargest))
    dirLargest = true;
else
    validateattributes(dirLargest, {'logical'},{'scalar'});
end

%% setup

%ensure inputs of correct type
points = double(points);
viewPoint = double(viewPoint);

%create kdtree
kdtreeobj = KDTreeSearcher(points,'distance','euclidean');

%get nearest neighbours
n = knnsearch(kdtreeobj,points,'k',(numNeighbours+1));

%remove self
n = n(:,2:end);

%find difference in position from neighbouring points
p = repmat(points(:,1:3),numNeighbours,1) - points(n(:),1:3);
p = reshape(p, size(points,1),numNeighbours,3);

%calculate values for covariance matrix
C = zeros(size(points,1),6);
C(:,1) = sum(p(:,:,1).*p(:,:,1),2);
C(:,2) = sum(p(:,:,1).*p(:,:,2),2);
C(:,3) = sum(p(:,:,1).*p(:,:,3),2);
C(:,4) = sum(p(:,:,2).*p(:,:,2),2);
C(:,5) = sum(p(:,:,2).*p(:,:,3),2);
C(:,6) = sum(p(:,:,3).*p(:,:,3),2);
C = C ./ numNeighbours;

%% normals and curvature calculation

normals = zeros(size(points));
curvature = zeros(size(points,1),1);
for i = 1:(size(points,1))
    
    %form covariance matrix
    Cmat = [C(i,1) C(i,2) C(i,3);...
        C(i,2) C(i,4) C(i,5);...
        C(i,3) C(i,5) C(i,6)];  
    
    %get eigen values and vectors
    [v,d] = eig(Cmat);
    d = diag(d);
    [lambda,k] = min(d);
    
    %store normals
    normals(i,:) = v(:,k)';
    
    %store curvature
    curvature(i) = lambda / sum(d);
end

%% flipping normals

%ensure normals point towards viewPoint
points = points - repmat(viewPoint,size(points,1),1);
if(dirLargest)
    [~,idx] = max(abs(normals),[],2);
    idx = (1:size(normals,1))' + (idx-1)*size(normals,1);
    dir = normals(idx).*points(idx) > 0;
else
    dir = sum(normals.*points,2) > 0;
end

normals(dir,:) = -normals(dir,:);

%%
% normalize curvature
normalized_curvature = (curvature - min(curvature)) / (max(curvature) - min(curvature));
normalized_curvature = 1 ./ (1 + exp(-10*(normalized_curvature - mean(normalized_curvature))));


end