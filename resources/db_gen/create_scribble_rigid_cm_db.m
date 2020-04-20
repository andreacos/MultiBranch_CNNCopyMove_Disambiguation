function create_scribble_rigid_cm_db(input_path, output_path, crop_size, roi_size, n_pairs, para)
% This function generate rigid copy-move
% Input:
%   input_path: the path to host images
%   output_path: the path to store forged images and 4 patches
%   crop_size: crop the host image to this size
%   roi_size: one dimension of image input to the network
%   n_pairs: the number of forged images. If n_pairs > # of images inside
%   input_path, host images will be reused. The location of source/target
%   however is generated randomly
%   para: all other parameters

    warning off;
    assert(roi_size < 0.5*crop_size, 'ROI is too large')
    
    % get all images, prepare the output folders
    if ~contains(input_path, '.txt')
        [file_path, file_name] = get_file_list(input_path, [], []);
    else
        file_path = read_file_list(input_path);
        file_name = cell(length(file_path),1);
        for i = 1:length(file_path)
            [~,name,ext] = fileparts(file_path{i});
            file_name{i} = [name ext];
        end
    end
    [file_path, file_name] = prepare_files(file_path, file_name, n_pairs);
    
    if ~exist(output_path, 'dir')
        mkdir(output_path);
    end
    if ~exist([output_path filesep 'pos'], 'dir')
        mkdir([output_path filesep 'pos']);
    end
    if para.img_save == true && ~exist([output_path filesep 'img'], 'dir')
        mkdir([output_path filesep 'img']);
    end
    
    parfor i = 1:n_pairs
        fprintf('\nProcessing: %s in ', file_name{i});
        try
            %newfp = strrep(file_path{i}, '/quoctin/', '/quoctin.phan/');
            tic;
            I = imread(file_path{i});
        catch ME
            fprintf('\nReading error: %s\n', ME.identifier);
            continue
        end
        [h,w,~] = size(I);
        if h < crop_size || w < crop_size
            continue;
        end
        
        % crop the image
        top = randi(h - crop_size - 1);
        left = randi(w - crop_size - 1);
        I = I(top:top+crop_size-1, left:left+crop_size-1, :);
        [H,W,~] = size(I);
        ort_shift = {[W/4; H/4],[3*W/4; H/4], [3*W/4; 3*H/4], [W/4; 3*H/4]};
        
        % generate rigid copy move
        T     = [1 0 0; 0 1 0; 0 0 1];
        
        % generate polygon from n_points, shift to a random orthant
        coords = gen_polygon(20, roi_size+10, roi_size+10);
        dt=delaunayTriangulation(coords(1,:)',coords(2,:)');
        [k, ~] = convexHull(dt); % return vertices of convex hull
        coords = coords(:,k);
            
        src_orthant = randi(length(ort_shift));
        src_coords = round(bsxfun(@plus, bsxfun(@minus, coords, mean(coords,2)), ort_shift{src_orthant}));
        dst_orthant  = mod(src_orthant, 4) + 1;
        tran = round(-mean(src_coords,2) + ort_shift{dst_orthant});
        
        % copy-move creation
        dst_coords = src_coords + tran;
        dst_mask = poly2mask(dst_coords(1,:), dst_coords(2,:), H, W); % find the internal region
        src_mask = poly2mask(src_coords(1,:), src_coords(2,:), H, W); % find the internal region
        Tf = T; Tf(1:2,3) = tran;
        
        [src_r,src_c] = find(src_mask==1);
        [dst_r,dst_c] = find(dst_mask==1);
        I_flat = reshape(I, H*W, []);
        I_flat(sub2ind([H,W], dst_r, dst_c), :) = I_flat(sub2ind([H,W], src_r, src_c), :);
        I_forged = reshape(I_flat, H, W, []);
        
        % flip a coin, blur the border
        if round(rand(1)) > 0.5
            I_forged = blurring_border(I_forged, dst_mask);
        end
        
        % global blending
        if para.gl_blending
            I_forged = global_blending(I_forged);
        end
        
        % extract pairs
        I_src = standardize(I_forged, src_mask, roi_size);
        I_dst = standardize(I_forged, dst_mask, roi_size);
        
        % transformation matrix for saving
        [~,random_string] = fileparts(tempname);

        % save pairs
        save_pairs([output_path filesep 'pos'], [random_string '.hdf5'], ...
                        uint8(I_dst), uint8(I_src), Tf);
        if para.img_save
            imwrite(uint8(I_forged), [output_path filesep 'img' filesep random_string '.png']);
            mask = zeros(size(I_forged), 'uint8');
            bg_mask = ~(src_mask | dst_mask);
            mask(:,:,1) = dst_mask.*255;
            mask(:,:,2) = src_mask.*255;
            mask(:,:,3) = bg_mask.*255;
            imwrite(mask, [output_path filesep 'img' filesep random_string '_tm.png']);
            dlmwrite([output_path filesep 'img' filesep random_string '_trans.txt'], Tf);
        end
        %figure;imshowpair(uint8(I_src), uint8(I_dst), 'montage');
        %close all
        fprintf(' %.2f s\n', toc);
    end 
end

function I_out = blurring_border(I, mask)
    ker_sizes = [3,5,7,9,11];
    idx = randi(numel(ker_sizes));
    C = size(I,3);
    % edge detection
    h_ed = -ones(5); h_ed(3,3) = 24;
    mask_edge = imfilter(double(mask), h_ed, 'replicate') > 0;
    count = 1;
    while count < 6
        mask_edge = imdilate(mask_edge,[0,1,0;1,1,1;0,1,0]);
        count = count + 1;
    end
    
    % applying blurring
    h = fspecial('average', ker_sizes(idx));
    I_out = zeros(size(I));
    for c = 1:C
        I_out(:,:,c) = roifilt2(h, I(:,:,c), mask_edge);
    end
end

function I_out = standardize(I, mask, crop_size)
% This function extracts the center region of crop_size x crop_size
    if size(I,3) == 3 && size(mask,3) == 1
       mask = repmat(mask, [1,1,size(I,3)]);
    end
    [r, c] = find(mask(:,:,1) == 1);
    max_X = max(c(:)); min_X = min(c(:));
    max_Y = max(r(:)); min_Y = min(r(:));
    sx = floor((min_X+max_X)/2) - floor(crop_size/2); sx = sx + (sx==0);
    sy = floor((min_Y+max_Y)/2) - floor(crop_size/2); sy = sy + (sy==0);
    I_out = I(sy:sy+crop_size-1, sx:sx+crop_size-1, :);
end

function coords = gen_polygon(n_points, h, w)
% This function generates polygon from n_points
% All points stay in a box [h,w]
    coords = [round(rand(1,n_points)*w + 1);round(rand(1,n_points)*h + 1)];
    c = mean(coords,2);
    d = bsxfun(@minus, coords, c); % vectors connecting center point and given points
    angles = atan2(d(2,:), d(1,:));
    [~,idx] = sort(angles); % sort by angles
    coords = coords(:,idx);
    coords = [coords coords(:,1)]; % add the first to the end to close the polygon
end

function [file_path, file_name] = prepare_files(file_path, file_name, n_pairs)
% This function circularly makes a file list ready for creating copy move
    if length(file_path) > n_pairs
        file_path = file_path(1:n_pairs);
        file_name = file_name(1:n_pairs);
    else
        n_ext = n_pairs-length(file_path);
        ext_file_path = cell(n_ext, 1);
        ext_file_name = cell(n_ext, 1);
        for i = 1:n_ext
            ext_file_path{i} = file_path{mod(i-1, length(file_path))+1};
            ext_file_name{i} = file_name{mod(i-1, length(file_name))+1};
        end
        file_path = [file_path; ext_file_path];
        file_name = [file_name; ext_file_name];
    end
end

function I_dest = global_blending(I)
% This function performs global blending by different operations
    C = size(I,3);
    isModified = rand() > 0.5;
    if ~isModified
        I_dest = I;
        return;
    end
    pp = randi(5);
    switch pp
        case 1 % lowpass + highpass
            h = {fspecial('gaussian', 3, 0.5),...
                 fspecial('gaussian', 3, 1.0),...
                 fspecial('gaussian', 3, 1.5),...
                 fspecial('gaussian', 3, 2.0),...
                 fspecial('average', 3),...
                 fspecial('unsharp', 0.5)};
            idx = randi(numel(h));
            I_dest = imfilter(I, h{idx}, 'replicate');
            
        case 2 % denoising
            idx = randi(2);
            h1 = @(x) wiener2(x,[3,3]);
            h2 = @(x) wiener2(x,[5,5]);
            h = {h1,h2};
            I_dest = zeros(size(I));
            for c = 1:C
            	I_dest(:,:,c) = h{idx}(I(:,:,c));
            end
           
        case 3 % WGN
            h = @(x) imnoise(uint8(x), 'gaussian', 0, 0.001);
            I_dest = zeros(size(I));
            for c = 1:C
                I_dest(:,:,c) = h(I(:,:,c));
            end
           
        case 4 % tonal adjustment
            h1 = @(x) imadjust(uint8(x), stretchlim(x,2/100), [], 0.8);
            h2 = @(x) imadjust(uint8(x), stretchlim(x,6/100), [], 0.8);
            h3 = @(x) histeq(uint8(x));
            h = {h1,h2,h3};
            idx = randi(3);
            I_dest = zeros(size(I));
            for c = 1:C
                I_dest(:,:,c) = h{idx}(I(:,:,c));
            end
            
        case 5 % JPEG compression
            jpeg_qf     = 55:5:100;
            idx         = randi(length(jpeg_qf));
            I_dest      = jpeg_compress(I, jpeg_qf(idx));
           
    end
end

function I_comp = jpeg_compress(I, QF)
% This function compresses an image and return the compressed values
    [~,random_string] = fileparts(tempname);
    tmp_name = [random_string '.jpg'];
    imwrite(uint8(I), tmp_name, 'Quality', QF);
    I_comp = imread(tmp_name);
    delete(tmp_name);
end

function save_pairs(output_path, file_name, I1, I2, T)
    % create db
    h5create([output_path filesep file_name], '/T', size(T), 'Datatype', 'single');
    h5create([output_path filesep file_name], '/x1', size(I1), 'Datatype', 'uint8');
    h5create([output_path filesep file_name], '/x2', size(I2), 'Datatype', 'uint8');
    % write
    h5write([output_path filesep file_name], '/T', single(T));
    h5write([output_path filesep file_name], '/x1', uint8(I1));
    h5write([output_path filesep file_name], '/x2', uint8(I2));
end