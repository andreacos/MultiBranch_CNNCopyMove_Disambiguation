clear all;
close all;
addpath('export_fig');

% unchanged declarations
%input_path  = '../datasets/syn_set';
%output_path = '../datasets/syn_set_scribble';
input_path  = '../datasets/RAISE_DRESDEN_VISION_mixed';
output_path = '../datasets/RAISE_DRESDEN_VISION_mixed_scribble_pair_rigid';
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

% define parameters of the transform
para.tran = [];
para.gl_blending = true;
para.interp = 'linear';

% no need to store the forged images for training (because we need
% source/target only
para.img_save = false;

% run
create_scribble_rigid_cm_db(input_path, output_path, 1024, 64, 900000, para);