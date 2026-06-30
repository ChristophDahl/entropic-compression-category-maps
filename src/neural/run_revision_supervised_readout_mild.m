%% run_revision_supervised_readout_mild.m
% Reproduce the mild balanced supervised-readout analysis.

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src', 'revision'));
run(fullfile(fileparts(mfilename('fullpath')), '..', 'src', 'revision', 'main_mild_balanced_supervised_readout_10class_v2.m'));
