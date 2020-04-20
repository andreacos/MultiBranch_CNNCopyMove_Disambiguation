clear all;
close all;
addpath('export_fig');

% unchanged declarations
%input_path  = '../datasets/syn_set';
%output_path = '../datasets/syn_set_scribble';
input_path  = '../../../RAISE2K';
output_path = 'RAISE_DRESDEN_VISION_mixed_scribble_pair_interp';
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

% define parameters of the transform
para.tran = [];
para.angles = 2:2:180;
para.perb_angles = -5:1:5;
para.scale_factors = [0.5:0.01:1 1:0.02:2.0];
para.perb_scale_factors = -0.1:0.01:0.1;
para.gl_blending = true;
para.interp = 'linear';
para.save_pair = 2;

% no need to store the forged images for training (because we need
% source/target only
para.img_save = false;

% run
create_scribble_all_db(input_path, output_path, 1024, 64, 900000, para);