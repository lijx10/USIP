function img2 = crop_img(img,border)
% 
% Crop image by removing edges with homogeneous intensity
% 
% 
%USAGE
%-----
% img2 = crop_img(img)
% img2 = crop_img(img,border)
% 
% 
%INPUT
%-----
% - IMG: MxNxC matrix, where MxN is the size of the image and C is the
%   number of color layers
% - BORDER: maximum number of pixels at the borders (default: 0)
% 
% 
%OUPUT
%-----
% - IMG2: cropped image
% 
% 
%EXAMPLE
%-------
% >> img  = imread('my_pic.png');
% >> img2 = crop_img(img,0);
% >> imwrite(img2,'my_cropped_pic.png')
% 

% Guilherme Coco Beltramini (guicoco@gmail.com)
% 2013-May-29, 12:29 pm


% Input
%==========================================================================
if nargin<2
    border = 0;
end


% Initialize
%==========================================================================
[MM,NN,CC] = size(img);
edge_col   = zeros(2,CC); % image edges (columns)
edge_row   = edge_col;    % image edges (rows)


% Find the edges
%==========================================================================
for cc=1:CC % loop for the colors
    
    
    % Top left corner
    %================
    
    % Find the background
    %--------------------
    img_bg = img(:,:,cc) == img(1,1,cc);
    
    % Columns
    %--------
    cols = sum(img_bg,1);
    if cols(1)==MM % first column is background
        tmp = find(diff(cols),1,'first'); % background width on the left
        if ~isempty(tmp)
            edge_col(1,cc) = tmp + 1 - border;
        else % no background
            edge_col(1,cc) = 1;
        end
    else % no background
        edge_col(1,cc) = 1;
    end
    
    % Rows
    %-----
    rows = sum(img_bg,2);
    if rows(1)==NN % first row is background
        tmp = find(diff(rows),1,'first'); % background height at the top
        if ~isempty(tmp)
            edge_row(1,cc) = tmp + 1 - border;
        else % no background
            edge_row(1,cc) = 1;
        end
    else % no background
        edge_row(1,cc) = 1;
    end
    
    
    % Bottom right corner
    %====================
    
    % Find the background
    %--------------------
    img_bg = img(:,:,cc) == img(MM,NN,cc);
    
    % Columns
    %--------
    cols = sum(img_bg,1);
    if cols(end)==MM % last column is background
        tmp = find(diff(cols),1,'last'); % background width on the right
        if ~isempty(tmp)
            edge_col(2,cc) = tmp + border;
        else % no background
            edge_col(2,cc) = NN;
        end
    else % no background
        edge_col(2,cc) = NN;
    end
    
    % Rows
    %-----
    rows = sum(img_bg,2);
    if rows(end)==NN % last row is background
        tmp = find(diff(rows),1,'last'); % background height at the bottom
        if ~isempty(tmp)
            edge_row(2,cc) = tmp + border;
        else % no background
            edge_row(2,cc) = MM;
        end
    else % no background
        edge_row(2,cc) = MM;
    end
    
    
    % Identify homogeneous color layers
    %==================================
    if edge_col(1,cc)==1 && edge_col(2,cc)==NN && ...
            edge_row(1,cc)==1 && edge_row(2,cc)==MM && ...
            ~any(any(diff(img(:,:,cc),1)))
        edge_col(:,cc) = [NN;1];
        edge_row(:,cc) = [MM;1]; % => ignore layer
    end
    
    
end


% Indices of the edges
%==========================================================================

% Columns
%--------
tmp      = min(edge_col(1,:),[],2);
edge_col = [tmp ; max(edge_col(2,:),[],2)];
if edge_col(1)<1
   edge_col(1) = 1;
end
if edge_col(2)>NN
   edge_col(2) = NN;
end

% Rows
%-----
tmp      = min(edge_row(1,:),[],2);
edge_row = [tmp ; max(edge_row(2,:),[],2)];
if edge_row(1)<1
   edge_row(1) = 1;
end
if edge_row(2)>MM
   edge_row(2) = MM;
end


% Crop the edges
%==========================================================================
img2 = zeros(edge_row(2)-edge_row(1)+1,...
    edge_col(2)-edge_col(1)+1,CC,class(img));
for cc=1:CC % loop for the colors
    img2(:,:,cc) = img(edge_row(1):edge_row(2),edge_col(1):edge_col(2),cc);
end