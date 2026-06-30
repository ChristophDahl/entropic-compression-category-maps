%% run_revision_resnet50_controls_strong.m
% Reproduce the strong pooled-nuisance conditional ResNet-50 control.

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src', 'revision'));
run_resnet50_conditional_controls('full_extended');
