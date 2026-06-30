%% run_revision_resnet50_controls_mild.m
% Reproduce the mild nuisance conditional ResNet-50 control.

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src', 'revision'));
run_resnet50_conditional_controls('mild_control');
