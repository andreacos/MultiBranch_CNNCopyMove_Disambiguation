function create_scribble_all_db(input_path, output_path, crop_size, roi_size, n_pairs, para)
% This function generates copy-move of different kinds: rotation,
% resizing, their combination, rigid
% Input:
%   input_path: the path to host images
%   output_path: the path to store forged images and 4 patches
%   crop_size: crop the host image to this size
%   roi_size: one dimension of image input to the network
%   n_pairs: the number of forged images. If n_pairs > # of images inside
%   input_path, host images will be reused. The location of source/target
%   however is generated randomly
%   para: all other parameters


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
    if ~exist([output_path filesep 'pos_neg'], 'dir')
        mkdir([output_path filesep 'pos_neg']);
    end
    if para.img_save == true && ~exist([output_path filesep 'img'], 'dir')
        mkdir([output_path filesep 'img']);
    end
    
    for i = 1:n_pairs
        fprintf('\nProcessing: %s in ', file_name{i});
        try
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
        
        % generate copy move
        if ~isempty(para.tran)
            tran_type = para.tran;
        else
            tran_type = randi(3); % randomly select the kind of copy-move
        end
        angles = para.angles; 
        perb_angles = para.perb_angles;
        scale_factors = para.scale_factors;
        perb_scale_factors = para.perb_scale_factors;
        switch tran_type
            case 1 % rotation 
                theta = angles(randi(length(angles)));
                delta = perb_angles(randi(length(perb_angles)));
                T     = [cosd(theta) -sind(theta) 0; sind(theta) ...
                                  cosd(theta) 0; 0 0 1];
                T_d   = [cosd(theta+delta) -sind(theta+delta) 0; ...
                                  sind(theta+delta) cosd(theta+delta) 0; 0 0 1];
            case 2 % resizing
                sx      = scale_factors(randi(length(scale_factors)));
                sy      = scale_factors(randi(length(scale_factors)));
                delta_x = perb_scale_factors(randi(length(perb_scale_factors)));
                delta_y = perb_scale_factors(randi(length(perb_scale_factors)));
                T       = [sx 0 0; 0 sy 0; 0 0 1];
                T_d     = [sx+delta_x 0 0; 0 sy+delta_y 0; 0 0 1];
                
            case 3 % combination
                theta  = angles(randi(length(angles)));
                delta  = perb_angles(randi(length(perb_angles)));
                T1     = [cosd(theta) -sind(theta) 0; sind(theta) ...
                                   cosd(theta) 0; 0 0 1];
                T1_d   = [cosd(theta+delta) -sind(theta+delta) 0; ...
                                   sind(theta+delta) cosd(theta+delta) 0; 0 0 1];
                sx      = scale_factors(randi(length(scale_factors)));
                sy      = scale_factors(randi(length(scale_factors)));
                delta_x = perb_scale_factors(randi(length(perb_scale_factors)));
                delta_y = perb_scale_factors(randi(length(perb_scale_factors)));
                
                T2      = [sx 0 0; 0 sy 0; 0 0 1];
                T2_d    = [sx+delta_x 0 0; 0 sy+delta_y 0; 0 0 1];
                
                % random order
                it = round(rand());
                if it == 1
                    T   = T2*T1;
                    T_d = T2_d*T1_d;
                else
                    T   = T1*T2;
                    T_d = T1_d*T2_d;
                end
        end
        
        % generate polygon from n_points, shift to a random orthant
        while true
            coords = gen_polygon(20, H/6, W/6);
            dt=delaunayTriangulation(coords(1,:)',coords(2,:)');
            [k, ~] = convexHull(dt); % return vertices of convex hull
            coords = coords(:,k);
            if max(coords(1,:))-min(coords(1,:)) > roi_size*2 && ...
               max(coords(2,:))-min(coords(2,:)) > roi_size*2
                break
            end
        end
        src_orthant = randi(length(ort_shift));
        src_coords = round(bsxfun(@plus, bsxfun(@minus, coords, mean(coords,2)), ort_shift{src_orthant}));
        dst_orthant  = mod(src_orthant, 4) + 1;
        tran = round(-mean(src_coords,2) + ort_shift{dst_orthant});
        
        % copy-move creation
        [I_forged, src_mask, Tf, dst_mask, dst_coords, true_tran] = copy_move_creation(I, src_coords, T, tran, para.interp);
%       DEBUG
%       imshow(uint8(I_forged)); hold on
%       plot(src_coords(1,:), src_coords(2,:), '-g', 'LineWidth', 2);
%       plot(dst_coords(1,:), dst_coords(2,:), '-r', 'LineWidth', 2);
        
        % DO NOT blur the border because the area is big, center cropping
        % will not include the border
        % I_forged = blurring_border(I_forged, dst_mask);
        
        % global blending
        if para.gl_blending
            I_forged = global_blending(I_forged);
        end
        
        % finding Tf_d
        m = mean(src_coords, 2);
        Tt = eye(3); Tt(1:2, 3) = -m;
        Tf_d = inv(Tt) * T_d * Tt; % rotation, rescaling
        Tf_d(1:2,3) = Tf_d(1:2,3) + true_tran; % translation
        
        % extract pairs
        I_src = standardize(I_forged, src_mask, roi_size);
        I_dst = standardize(I_forged, dst_mask, roi_size);

        [I_est, dst_mask_est]  = get_est(I_forged, src_coords, Tf_d, 'linear');
        I_dst_est = standardize(I_est, dst_mask_est, roi_size);
        [I_est, src_mask_est]  = get_est(I_forged, dst_coords, inv(Tf_d), 'linear');
        I_src_est = standardize(I_est, src_mask_est, roi_size);
        
        % transformation matrix for saving
        [~,random_string] = fileparts(tempname);

        % save two pairs in HDF5 file, ready for training
        save_two_pairs([output_path filesep 'pos_neg'], [random_string '.hdf5'], ...
                        uint8(I_dst), uint8(I_dst_est), uint8(I_src), ...
                        uint8(I_src_est), Tf);
        
        % true if we want to save the forged image for testing
        if para.img_save
            imwrite(uint8(I_forged), [output_path filesep 'img' filesep random_string '.png']);
            mask = zeros(size(I_forged), 'uint8');
            src_mask = poly2mask(src_coords(1,:), src_coords(2,:), H, W);
            dst_mask = poly2mask(dst_coords(1,:), dst_coords(2,:), H, W);
            bg_mask = ~(src_mask | dst_mask);
            mask(:,:,1) = dst_mask.*255;
            mask(:,:,2) = src_mask.*255;
            mask(:,:,3) = bg_mask.*255;
            imwrite(mask, [output_path filesep 'img' filesep random_string '_tm.png']);
            dlmwrite([output_path filesep 'img' filesep random_string '_trans.txt'], Tf);
        end
%       DEBUG
%       subplot(2,1,1);
%       imshowpair(uint8(I_dst), uint8(I_dst_est), 'montage');
%       subplot(2,1,2);
%       imshowpair(uint8(I_src), uint8(I_src_est), 'montage');
%       figure; imshow(uint8(I_forged));
        fprintf(' %.2f s\n', toc);
%        close all
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
% If the region is smaller than crop_size, the region is resized
    if size(I,3) == 3 && size(mask,3) == 1
       mask = repmat(mask, [1,1,size(I,3)]);
    end
    [r, c] = find(mask(:,:,1) == 1);
    max_X = max(c(:)); min_X = min(c(:));
    max_Y = max(r(:)); min_Y = min(r(:));
    I_out = I(min_Y:max_Y, min_X:max_X,:);
    [h,w,~] = size(I_out);
    if crop_size > min([h,w])
        f = (crop_size+2)/min([h,w]);
        I_out = imresize(I_out, f, 'nearest');
        [h,w,~] = size(I_out);
    end
    sx = floor(w/2) - floor(crop_size/2); sx = sx + (sx==0);
    sy = floor(h/2) - floor(crop_size/2); sy = sy + (sy==0);
    I_out = I_out(sy:sy+crop_size-1, sx:sx+crop_size-1, :);
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

function [I_out, src_mask, Tf, dst_mask, dst_coords, true_tran] = copy_move_creation(I, src_coords, T, tran, interp)
% This function creates the copy-move
% Input:
%   I: the host image
%   src_coords: coordinates of source polygon
%   T: transformation matrix
%   tran: the translation distance
%   interp: interpolation method
% Output:
%   I_out: the forged image
%   src_mask: binary image highlighting the source
%   dst_mask: binary image highlighting the target
%   Tf: the actual transformation matrix (can be different to T because the
%   translation amount might be adjusted if violated)
%   dst_coords: coordinates of target polygon
%   true_tran: the actual translation distance (can be different to tran because the
%   translation amount might be adjusted if violated)


    warning off;
    [H,W,C] = size(I);
    I_flat = reshape(I, H*W, []);
    % get coordinates of points in src
    src_mask = poly2mask(src_coords(1,:), src_coords(2,:), H, W); % find the internal region
    [rows,cols] = find(src_mask == 1);
    src_poly_coords = [cols'; rows'];
    % T: affine2d of rotation and scaling
    m = mean(src_coords, 2);
    Tt = eye(3); Tt(1:2, 3) = -m;
    % translate to the origin before applying the rotation and scaling
    Tf = inv(Tt) * T * Tt;
    % apply transformation to vertices, and the inside points
    dst_scat_coords = Tf * [src_poly_coords; ones(1,size(src_poly_coords,2))];
    dst_scat_coords(3,:) = [];
    dst_coords = Tf * [src_coords; ones(1,size(src_coords,2))];
    dst_coords(3,:) = [];
    % translate and make sure they stay inside
    dst_scat_coords = dst_scat_coords + tran;
    dst_coords = dst_coords + tran;
    if max(dst_scat_coords(1,:)) > W - 32
        d = max(dst_scat_coords(1,:));
        dst_scat_coords(1,:) = dst_scat_coords(1,:) - (d-W+32);
        dst_coords(1,:) = dst_coords(1,:) - (d-W+32);
        tran(1) = tran(1) - (d-W+32);
    elseif min(dst_scat_coords(1,:)) < 32
        d = min(dst_scat_coords(1,:));
        dst_scat_coords(1,:) = dst_scat_coords(1,:) - d + 32;
        dst_coords(1,:) = dst_coords(1,:) - d + 32;
        tran(1) = tran(1) - d + 32;
    end
    if max(dst_scat_coords(2,:)) > H - 32
        d = max(dst_scat_coords(2,:));
        dst_scat_coords(2,:) = dst_scat_coords(2,:) - (d-H+32);
        dst_coords(2,:) = dst_coords(2,:) - (d-H+32);
        tran(2) = tran(2) - (d-H+32);
    elseif min(dst_scat_coords(2,:)) < 32
        d = min(dst_scat_coords(2,:));
        dst_scat_coords(2,:) = dst_scat_coords(2,:) - d + 32;
        dst_coords(2,:) = dst_coords(2,:) - d + 32;
        tran(2) = tran(2) - d + 32;
    end
    Tf(1:2,3) = Tf(1:2,3)+tran;
    true_tran = tran;
    % poly2mask(x,y,m,n) convention: x=col, y=row, m=height, n=width
    dst_mask = poly2mask(dst_coords(1,:), dst_coords(2,:), H, W); % find the internal region
    [rows,cols] = find(dst_mask == 1);
    dst_poly_coords = [cols'; rows'];
    
    % interpolate
    %dst_scat_coords = round(dst_scat_coords);
    I_out_flat = I_flat;
    V = double(I_flat(sub2ind([H,W], src_poly_coords(2,:), src_poly_coords(1,:)),:));
    for c = 1:C
        interp_data = griddata(dst_scat_coords(2,:), dst_scat_coords(1,:), V(:,c), ...
                               dst_poly_coords(2,:), dst_poly_coords(1,:), interp);
        % fill nan values by the background values
        nan_idx = find(isnan(interp_data)==1);
        if ~isempty(nan_idx)
            nan_idx_ = sub2ind([H,W], dst_poly_coords(2,nan_idx), dst_poly_coords(1,nan_idx));
            interp_data(nan_idx) = I_flat(nan_idx_,c);
        end
        % fill 0's values by the background values
%         zero_idx = find(interp_data==0);
%         if ~isempty(zero_idx)
%             interp_data(zero_idx) = griddata(dst_poly_coords(2,:), dst_poly_coords(1,:), interp_data, ...
%                           dst_poly_coords(2,zero_idx), dst_poly_coords(1,zero_idx), interp);
%         end
        I_out_flat(sub2ind([H,W], dst_poly_coords(2,:), dst_poly_coords(1,:)), c) = interp_data;
    end
    I_out = reshape(I_out_flat, H, W, C);
    dst_coords = round(dst_coords);
end

function [I_est,dst_est_mask]  = get_est(I, src_coords, T, interp)
% This function extracts the transform of a region
% Input:
%   I: forged image
%   src_coords: coordinates of polygon vertices
%   T: transformation matrix
%   interp: interpolation method
% Output:
%   I_est: I after transform src_coords by T
%   dst_est_mask: mask highlighting the transform region

    [H,W,C] = size(I);
    I_flat = reshape(I, H*W, []);
    
    % perform transformation and get the target mask
    tran_scat_coords = T*[src_coords; ones(1,size(src_coords,2))];
    tran_scat_coords(3,:) = [];
    dst_est_mask = poly2mask(tran_scat_coords(1,:), tran_scat_coords(2,:), H, W);
    
    % create a bounding box around the target
    max_X = max(tran_scat_coords(1,:)); min_X = min(tran_scat_coords(1,:));
    max_Y = max(tran_scat_coords(2,:)); min_Y = min(tran_scat_coords(2,:));
    [X, Y] = meshgrid(round(min_X:max_X), round(min_Y:max_Y));
    tran_grid_coords = [X(:)'; Y(:)'];
    
    % perform backward transformation to get source coordinates
    inv_tran_grid_coords = inv(T)*[tran_grid_coords; ones(1,size(tran_grid_coords,2))];
    inv_tran_grid_coords(3,:) = [];
    
    % perform interpolation using pixel values in the source. This step is to find the
    % unavailable values on the target
    V = double(I_flat);
    VV = double(I);
    I_out_flat = I_flat;
    [X, Y] = meshgrid(1:W, 1:H);
    for c = 1:C
        % 
        interpolant = griddedInterpolant(Y, X, VV(:,:,c), interp);
        interp_data = interpolant(inv_tran_grid_coords(2,:), inv_tran_grid_coords(1,:));
        % fill nan values by the background values
        nan_idx = find(isnan(interp_data)==1);
        if ~isempty(nan_idx)
            nan_idx_ = sub2ind([H,W], inv_tran_grid_coords(2,nan_idx), inv_tran_grid_coords(1,nan_idx));
            interp_data(nan_idx) = I_flat(nan_idx_,c);
        end
        I_out_flat(sub2ind([H,W], tran_grid_coords(2,:), tran_grid_coords(1,:)), c) = interp_data;
    end
    I_est = reshape(I_out_flat,H,W,C);
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

function save_two_pairs(output_path, file_name, I1, I2, I3, I4, T)
    % create db
    h5create([output_path filesep file_name], '/T', size(T), 'Datatype', 'single');
    h5create([output_path filesep file_name], '/x1', size(I1), 'Datatype', 'uint8');
    h5create([output_path filesep file_name], '/x2', size(I2), 'Datatype', 'uint8');
    h5create([output_path filesep file_name], '/x3', size(I3), 'Datatype', 'uint8');
    h5create([output_path filesep file_name], '/x4', size(I4), 'Datatype', 'uint8');
    % write
    h5write([output_path filesep file_name], '/T', single(T));
    h5write([output_path filesep file_name], '/x1', uint8(I1));
    h5write([output_path filesep file_name], '/x2', uint8(I2));
    h5write([output_path filesep file_name], '/x3', uint8(I3));
    h5write([output_path filesep file_name], '/x4', uint8(I4));
end