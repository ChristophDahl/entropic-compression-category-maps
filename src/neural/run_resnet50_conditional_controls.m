function run_resnet50_conditional_controls(runMode)
%RUN_RESNET50_CONDITIONAL_CONTROLS Revision ResNet-50 conditional-control analysis.
%
%   run_resnet50_conditional_controls('full_extended') runs the strong pooled-nuisance control.
%   run_resnet50_conditional_controls('mild_control') runs the mild nuisance-control analysis.
%
% This file is a refactored version of main_nn_entropic_compression_resnet50_controls_v4.m.
% It keeps all helper functions local, but exposes runMode so the two manuscript
% controls can be reproduced without manually editing the source file.

if nargin < 1 || isempty(runMode)
    runMode = 'full_extended';
end

clc; close all;

% This file is intended as the final long Reviewer-3 control run.
% It writes files with the tag "full":
%   nn_resnet50_v4_controls_full_all_seeds.csv
%   nn_resnet50_v4_controls_full_summary.csv
%   nn_resnet50_v4_controls_full_main_table.csv
%   table_network_conditional_controls_full.tex
%   figure_nn_resnet50_v4_conditional_controls_full.png/pdf
%
% The first convolutional layer is deliberately excluded from full mode
% because it is computationally expensive and not needed for the manuscript
% control. It was used only in the earlier fast diagnostic.

cfg = make_cfg(runMode);
initialise_project(cfg);

if exist(cfg.logFile, 'file')
    delete(cfg.logFile);
end
diary(cfg.logFile);
cleanupDiary = onCleanup(@() diary('off')); %#ok<NASGU>

fprintf('\n=== ResNet-50 conditional network-control analysis: v4 SWITCHED ===\n');
fprintf('Project root: %s\n', cfg.projectRoot);
fprintf('Results dir : %s\n', cfg.resultsDir);
fprintf('Figures dir : %s\n', cfg.figDir);
fprintf('Log file    : %s\n', cfg.logFile);
fprintf('Run mode    : %s\n', cfg.runMode);
fprintf('Image dir   : %s\n', cfg.pooledImageDir);
fprintf('Label file  : %s\n', cfg.pooledLabelFile);

%% ------------------------------------------------------------------------
% Load labels and resolve images
% -------------------------------------------------------------------------
labelFile = cfg.pooledLabelFile;
if ~exist(labelFile, 'file')
    error('Label file not found: %s', labelFile);
end

L = readtable(labelFile, 'Delimiter', ',', 'PreserveVariableNames', true);
[L, imagePaths] = resolve_image_paths(L, cfg.pooledImageDir);
validate_image_paths(imagePaths);

targets = infer_targets(L);

fprintf('\nLoaded %d labelled images.\n', height(L));
fprintf('Target columns detected:\n');
fprintf('  object       : %s\n', targets.objectName);
fprintf('  superordinate: %s\n', targets.superName);
fprintf('  action/env   : %s\n', targets.actionName);
fprintf('  nuisance     : %s\n', targets.nuisanceName);

%% ------------------------------------------------------------------------
% Load pretrained network
% -------------------------------------------------------------------------
fprintf('\nLoading pretrained ResNet-50...\n');
try
    net = resnet50;
catch ME
    fprintf(2, '\nCould not load pretrained ResNet-50.\n');
    fprintf(2, 'Install: Deep Learning Toolbox Model for ResNet-50 Network.\n');
    fprintf(2, 'Do not use resnet50(''Weights'',''none'') for this manuscript analysis.\n\n');
    rethrow(ME);
end
inputSize = net.Layers(1).InputSize(1:2);

%% ------------------------------------------------------------------------
% Main analysis
% -------------------------------------------------------------------------
allRows = table();
summaryRows = table();

for iLayer = 1:numel(cfg.layers)
    layerName = cfg.layers{iLayer};
    fprintf('\n--- Layer %d/%d: %s ---\n', iLayer, numel(cfg.layers), layerName);

    Xraw = load_or_extract_activations(cfg, net, imagePaths, inputSize, layerName);
    X = prepare_activation_matrix(Xraw, layerName, cfg, numel(imagePaths));

    fprintf('  Activation matrix after preparation: %d images x %d features.\n', size(X,1), size(X,2));

    for iP = 1:numel(cfg.pcaDimList)
        pcaDim = cfg.pcaDimList(iP);
        pcaDimUse = min([pcaDim, size(X,1)-1, size(X,2)]);
        fprintf('  PCA dim requested %d, using %d.\n', pcaDim, pcaDimUse);

        % IMPORTANT: PCA must use Xuse, not Xraw, not logical, not gpuArray.
        Xuse = double(gather(X));
        if any(~isfinite(Xuse(:)))
            error('Activation matrix for layer %s contains NaN or Inf values before PCA.', layerName);
        end

        fprintf('  Starting PCA for %s with matrix %d x %d (%.2f GB as double).\n', ...
            layerName, size(Xuse,1), size(Xuse,2), numel(Xuse)*8/1e9);
        tPca = tic;
        [~, score] = pca(Xuse, 'NumComponents', pcaDimUse);
        fprintf('  Finished PCA for %s, dim %d, elapsed %.1f s.\n', ...
            layerName, pcaDimUse, toc(tPca));

        for iK = 1:numel(cfg.kList)
            k = cfg.kList(iK);
            fprintf('    k = %d; seeds = %d; permutations = %d\n', k, numel(cfg.seedList), cfg.nPerm);

            seedRows = table();

            for iSeed = 1:numel(cfg.seedList)
                seed = cfg.seedList(iSeed);
                rng(seed, 'twister');

                fprintf('      Seed %d/%d: k-means...\n', iSeed, numel(cfg.seedList));
                tSeed = tic;
                C = kmeans(score, k, ...
                    'Start', 'plus', ...
                    'Replicates', 1, ...
                    'MaxIter', cfg.kmeansMaxIter, ...
                    'Display', 'off', ...
                    'Options', statset('UseParallel', false));

                fprintf('      Seed %d/%d: metrics and permutations...\n', iSeed, numel(cfg.seedList));
                row = compute_control_metrics(C, targets, cfg.nPerm, seed, layerName, k, pcaDimUse);
                fprintf('      Seed %d/%d finished, elapsed %.1f s.\n', iSeed, numel(cfg.seedList), toc(tSeed));
                seedRows = [seedRows; row]; %#ok<AGROW>
            end

            allRows = [allRows; seedRows]; %#ok<AGROW>
            summaryRows = [summaryRows; summarise_seed_rows(seedRows, layerName, k, pcaDimUse)]; %#ok<AGROW>
        end
    end
end

%% ------------------------------------------------------------------------
% Save outputs
% -------------------------------------------------------------------------
tag = cfg.runMode;
allFile  = fullfile(cfg.resultsDir, sprintf('nn_resnet50_v4_controls_%s_all_seeds.csv', tag));
summaryFile = fullfile(cfg.resultsDir, sprintf('nn_resnet50_v4_controls_%s_summary.csv', tag));
mainFile = fullfile(cfg.resultsDir, sprintf('nn_resnet50_v4_controls_%s_main_table.csv', tag));

writetable(allRows, allFile);
writetable(summaryRows, summaryFile);

mainRows = summaryRows(summaryRows.k == cfg.mainK & summaryRows.pcaDim == cfg.mainPcaDim, :);
writetable(mainRows, mainFile);

fprintf('\nSaved:\n  %s\n  %s\n  %s\n', allFile, summaryFile, mainFile);

if ~isempty(mainRows)
    make_control_figure(cfg, mainRows);
    latexFile = fullfile(cfg.resultsDir, sprintf('table_network_conditional_controls_%s.tex', tag));
    write_latex_control_table(mainRows, latexFile);
    fprintf('  %s\n', latexFile);
else
    warning('No rows found for k=%d and PCA=%d. No main figure/table produced.', cfg.mainK, cfg.mainPcaDim);
end

fprintf('\nDone.\n');

end

%% ========================================================================
% Configuration
% ========================================================================
function cfg = make_cfg(runMode)

cfg = struct();

% Select run mode here.
%   'full_extended' = strong pooled-nuisance control.
%   'mild_control'  = mild balanced nuisance control.
if nargin < 1 || isempty(runMode)
    runMode = 'full_extended';
end
cfg.runMode = char(runMode);

cfg.projectRoot = 'I:\entropicCompression';
cfg.dataDir     = fullfile(cfg.projectRoot, 'data');
cfg.resultsDir  = fullfile(cfg.projectRoot, 'results');
cfg.figDir      = fullfile(cfg.projectRoot, 'figures');

% Keep activation caches separate across stimulus sets. The strong and mild
% controls can have different image sets; sharing caches would be unsafe.
cfg.cacheDir = fullfile(cfg.resultsDir, ...
    sprintf('activation_cache_resnet50_v4_controls_%s', cfg.runMode));

cfg.logFile = fullfile(cfg.resultsDir, ...
    sprintf('resnet50_controls_v4_%s_runlog.txt', cfg.runMode));

% Image and label files are selected automatically from cfg.runMode.
switch lower(cfg.runMode)
    case {'fast','full','full_extended'}
        % Original strong pooled-nuisance control.
        cfg.pooledImageDir  = fullfile(cfg.dataDir, ...
            'nn_images_resnet50_v3', ...
            'pooled_variants');

        cfg.pooledLabelFile = fullfile(cfg.resultsDir, ...
            'nn_resnet50_v3_pooled_variants_labels.csv');

    case 'mild_control'
        % Mild balanced nuisance control, matched to the supervised-readout setup.
        cfg.pooledImageDir  = fullfile(cfg.dataDir, ...
            'nn_images_resnet50_v7_10class_mild', ...
            'pooled_variants');

        cfg.pooledLabelFile = fullfile(cfg.resultsDir, ...
            'nn_resnet50_v7_10class_mild_labels.csv');

    otherwise
        error('Unknown cfg.runMode when assigning image set: %s', cfg.runMode);
end

switch lower(cfg.runMode)
    case 'fast'
        % Fast multi-layer diagnostic.
        % Runs all manuscript layers with light settings.
        cfg.layers = { ...
            'activation_1_relu', ...
            'activation_10_relu', ...
            'activation_22_relu', ...
            'activation_40_relu', ...
            'activation_49_relu', ...
            'avg_pool'};
        cfg.kList = 10;
        cfg.pcaDimList = 10;
        cfg.seedList = 1:2;
        cfg.nPerm = 20;
        cfg.mainK = 10;
        cfg.mainPcaDim = 10;

    case 'full'
        cfg.layers = { ...
            'activation_10_relu', ...
            'activation_22_relu', ...
            'activation_40_relu', ...
            'activation_49_relu', ...
            'avg_pool'};
        cfg.kList = [2 3 4 10];
        cfg.pcaDimList = [10 30 100];
        cfg.seedList = 1:20;
        cfg.nPerm = 200;
        cfg.mainK = 10;
        cfg.mainPcaDim = 30;
        
    case 'full_extended'
    % Focused manuscript control across the full sampled layer hierarchy.
    % This produces the primary pooled-nuisance conditional-control figure
    % at the main manuscript setting: k = 10, PCA = 30.
    %
    % This is not the full k-by-PCA sensitivity grid. It keeps the analysis
    % computationally tractable while sampling the hierarchy densely.

    cfg.layers = { ...
        'activation_4_relu',  ...
        'activation_7_relu',  ...
        'activation_10_relu', ...
        'activation_13_relu', ...
        'activation_16_relu', ...
        'activation_19_relu', ...
        'activation_22_relu', ...
        'activation_25_relu', ...
        'activation_28_relu', ...
        'activation_31_relu', ...
        'activation_34_relu', ...
        'activation_37_relu', ...
        'activation_40_relu', ...
        'activation_43_relu', ...
        'activation_46_relu', ...
        'activation_49_relu', ...
        'avg_pool'};

    cfg.kList = 10;
    cfg.pcaDimList = 30;

    cfg.seedList = 1:20;
    cfg.nPerm = 200;

    cfg.mainK = 10;
    cfg.mainPcaDim = 30;
    
    case 'mild_control'
    cfg.layers = { ...
        'activation_4_relu',  ...
        'activation_7_relu',  ...
        'activation_10_relu', ...
        'activation_13_relu', ...
        'activation_16_relu', ...
        'activation_19_relu', ...
        'activation_22_relu', ...
        'activation_25_relu', ...
        'activation_28_relu', ...
        'activation_31_relu', ...
        'activation_34_relu', ...
        'activation_37_relu', ...
        'activation_40_relu', ...
        'activation_43_relu', ...
        'activation_46_relu', ...
        'activation_49_relu', ...
        'avg_pool'};

    cfg.kList = 10;
    cfg.pcaDimList = 30;
    cfg.seedList = 1:20;
    cfg.nPerm = 200;

    cfg.mainK = 10;
    cfg.mainPcaDim = 30;

    otherwise
        error('Unknown cfg.runMode: %s', cfg.runMode);
end

cfg.miniBatchSize = 32;
cfg.kmeansMaxIter = 500;

% Safety limit before PCA. If a layer has more features than this, a fixed
% random feature subset is used. This prevents large convolutional layers from
% crashing MATLAB during conversion/PCA.
cfg.maxFeaturesForPCA = 50000;
cfg.featureSubsetSeed = 1001;

end

function initialise_project(cfg)
if ~exist(cfg.resultsDir, 'dir'), mkdir(cfg.resultsDir); end
if ~exist(cfg.figDir, 'dir'), mkdir(cfg.figDir); end
if ~exist(cfg.cacheDir, 'dir'), mkdir(cfg.cacheDir); end
end

%% ========================================================================
% Label and image handling
% ========================================================================
function [L, imagePaths] = resolve_image_paths(L, imageDir)

candidateCols = {'imagePath','ImagePath','path','Path','filename','fileName', ...
    'Filename','file','File','imageFile','ImageFile'};

colNames = L.Properties.VariableNames;
pathCol = '';
for i = 1:numel(candidateCols)
    if any(strcmp(colNames, candidateCols{i}))
        pathCol = candidateCols{i};
        break;
    end
end

if ~isempty(pathCol)
    raw = string(L.(pathCol));
    imagePaths = strings(height(L),1);
    for i = 1:height(L)
        p = raw(i);
        if strlength(p) == 0
            error('Empty image path in row %d of column %s.', i, pathCol);
        end
        if isfile(p)
            imagePaths(i) = p;
        else
            imagePaths(i) = string(fullfile(imageDir, char(p)));
        end
    end
else
    files = [dir(fullfile(imageDir, '**', '*.png')); ...
             dir(fullfile(imageDir, '**', '*.jpg')); ...
             dir(fullfile(imageDir, '**', '*.jpeg'))];
    if numel(files) ~= height(L)
        error(['No filename/path column detected and number of images (%d) ', ...
               'does not match number of label rows (%d).'], numel(files), height(L));
    end
    [~, ord] = sort({files.name});
    files = files(ord);
    imagePaths = strings(numel(files),1);
    for i = 1:numel(files)
        imagePaths(i) = string(fullfile(files(i).folder, files(i).name));
    end
end

L.imagePathResolved = imagePaths;

end

function validate_image_paths(imagePaths)
missing = ~isfile(imagePaths);
if any(missing)
    firstMissing = find(missing, 1);
    error('Image file not found, first missing row %d: %s', firstMissing, imagePaths(firstMissing));
end
end

function targets = infer_targets(L)

names = L.Properties.VariableNames;

targets = struct();
targets.objectName   = find_first(names, {'Ysub','Y_obj','Yobj','objectClass','class','label','category'});
targets.superName    = find_first(names, {'Ybasic','Y_sup','Ysup','superordinate','basic','animalVehicle'});
targets.actionName   = find_first(names, {'A','Aff','affordance','environment','env','action'});
targets.nuisanceName = find_first(names, {'V','variant','nuisance','condition','transform','imageCondition'});

if isempty(targets.objectName)
    error('Could not identify object-class target column in labels CSV.');
end
if isempty(targets.superName)
    error('Could not identify superordinate/basic target column in labels CSV.');
end
if isempty(targets.actionName)
    error('Could not identify action/environment target column in labels CSV.');
end
if isempty(targets.nuisanceName)
    error('Could not identify nuisance/variant target column in labels CSV.');
end

targets.Yobj = categorical(L.(targets.objectName));
targets.Ysup = categorical(L.(targets.superName));
targets.A    = categorical(L.(targets.actionName));
targets.V    = categorical(L.(targets.nuisanceName));

end

function out = find_first(names, candidates)
out = '';
for i = 1:numel(candidates)
    idx = strcmp(names, candidates{i});
    if any(idx)
        out = names{find(idx, 1)};
        return;
    end
end
end

%% ========================================================================
% Activation extraction and preparation
% ========================================================================
function X = load_or_extract_activations(cfg, net, imagePaths, inputSize, layerName)

safeLayer = regexprep(layerName, '[^A-Za-z0-9_]', '_');
cacheFile = fullfile(cfg.cacheDir, sprintf('pooled_%s_N%d_activations.mat', safeLayer, numel(imagePaths)));

if exist(cacheFile, 'file')
    fprintf('  Loading cached activations: %s\n', cacheFile);
    S = load(cacheFile, 'X', 'layerName');
    X = S.X;
    return;
end

fprintf('  Extracting activations for %d images...\n', numel(imagePaths));

imds = imageDatastore(cellstr(imagePaths), ...
    'ReadFcn', @(f) read_for_resnet(f, inputSize));

try
    X = activations(net, imds, layerName, ...
        'MiniBatchSize', cfg.miniBatchSize, ...
        'OutputAs', 'rows');
catch
    A = activations(net, imds, layerName, ...
        'MiniBatchSize', cfg.miniBatchSize);
    X = reshape_activations_to_rows(A);
end

X = single(gather(X));
save(cacheFile, 'X', 'layerName', '-v7.3');
fprintf('  Saved activation cache: %s\n', cacheFile);

end

function X = prepare_activation_matrix(Xraw, layerName, cfg, nExpectedImages)

fprintf('  Preparing activation matrix for %s...\n', layerName);
fprintf('    Raw class: %s; raw size:', class(Xraw));
fprintf(' %d', size(Xraw));
fprintf('\n');

X = gather(Xraw);

if ndims(X) > 2
    nObs = size(X, 1);
    X = reshape(X, nObs, []);
end

% Ensure rows are images. The activation extraction requests OutputAs rows.
if size(X,1) ~= nExpectedImages
    fprintf('    Warning: activation matrix has %d rows, expected %d rows. Continuing.\n', ...
        size(X,1), nExpectedImages);
end

% Convert logical/integer data to single first. Avoid immediate double
% conversion for huge convolutional layers.
X = single(X);

if any(~isfinite(X(:)))
    error('Activation matrix for layer %s contains NaN or Inf values after loading.', layerName);
end

fprintf('    After reshape/conversion: %d images x %d features.\n', size(X,1), size(X,2));
fprintf('    Approx memory as single: %.2f GB; as double: %.2f GB.\n', ...
    numel(X)*4/1e9, numel(X)*8/1e9);

% Remove constant columns.
fprintf('    Removing constant columns...\n');
colSd = std(X, 0, 1);
keep = colSd > 0 & isfinite(colSd);
if ~any(keep)
    error('No non-constant activation columns remain for layer %s.', layerName);
end
X = X(:, keep);
fprintf('    Remaining non-constant features: %d.\n', size(X,2));

% Feature cap before PCA for large convolutional layers.
if size(X,2) > cfg.maxFeaturesForPCA
    rng(cfg.featureSubsetSeed, 'twister');
    idx = randperm(size(X,2), cfg.maxFeaturesForPCA);
    idx = sort(idx);
    X = X(:, idx);
    fprintf('    Feature cap applied: using %d fixed randomly selected features.\n', size(X,2));
end

% Z-score manually in single precision.
fprintf('    Z-scoring features...\n');
mu = mean(X, 1);
sd = std(X, 0, 1);
sd(sd == 0 | ~isfinite(sd)) = 1;
X = (X - mu) ./ sd;
X(isnan(X)) = 0;

if any(~isfinite(X(:)))
    error('Activation matrix for layer %s contains NaN or Inf after z-scoring.', layerName);
end

fprintf('    Prepared matrix: %d images x %d features.\n', size(X,1), size(X,2));

end

function I = read_for_resnet(filename, inputSize)
I = imread(filename);
if ndims(I) == 2
    I = repmat(I, [1 1 3]);
end
if size(I,3) == 4
    I = I(:,:,1:3);
end
I = imresize(I, inputSize);
end

function X = reshape_activations_to_rows(A)
if ndims(A) == 4
    n = size(A,4);
    X = reshape(A, [], n)';
elseif ndims(A) == 2
    X = A;
else
    sz = size(A);
    n = sz(end);
    X = reshape(A, [], n)';
end
end

%% ========================================================================
% Metrics
% ========================================================================
function row = compute_control_metrics(C, targets, nPerm, seed, layerName, k, pcaDim)

C = categorical(C);
Yobj = targets.Yobj;
Ysup = targets.Ysup;
A    = targets.A;
V    = targets.V;

HC = entropy_discrete(C);

NMI_Yobj = nmi_target(C, Yobj);
NMI_Ysup = nmi_target(C, Ysup);
NMI_A    = nmi_target(C, A);
NMI_V    = nmi_target(C, V);

CNMI_Yobj_given_V = conditional_nmi_target(C, Yobj, V);
CNMI_Ysup_given_V = conditional_nmi_target(C, Ysup, V);

null_Yobj = permutation_nmi(C, Yobj, nPerm);
null_Ysup = permutation_nmi(C, Ysup, nPerm);
null_A    = permutation_nmi(C, A,    nPerm);
null_V    = permutation_nmi(C, V,    nPerm);

null_CYobjV = permutation_conditional_nmi(C, Yobj, V, nPerm);
null_CYsupV = permutation_conditional_nmi(C, Ysup, V, nPerm);

row = table();
row.layer = string(layerName);
row.k = k;
row.pcaDim = pcaDim;
row.seed = seed;
row.H_C = HC;

row.NMI_Yobj = NMI_Yobj;
row.NMI_Ysup = NMI_Ysup;
row.NMI_A    = NMI_A;
row.NMI_V    = NMI_V;
row.CNMI_Yobj_given_V = CNMI_Yobj_given_V;
row.CNMI_Ysup_given_V = CNMI_Ysup_given_V;

row.null_Yobj_mean = mean(null_Yobj);
row.null_Yobj_lo95 = prctile(null_Yobj, 2.5);
row.null_Yobj_hi95 = prctile(null_Yobj, 97.5);
row.corr_NMI_Yobj  = NMI_Yobj - row.null_Yobj_mean;

row.null_Ysup_mean = mean(null_Ysup);
row.null_Ysup_lo95 = prctile(null_Ysup, 2.5);
row.null_Ysup_hi95 = prctile(null_Ysup, 97.5);
row.corr_NMI_Ysup  = NMI_Ysup - row.null_Ysup_mean;

row.null_A_mean = mean(null_A);
row.null_A_lo95 = prctile(null_A, 2.5);
row.null_A_hi95 = prctile(null_A, 97.5);
row.corr_NMI_A  = NMI_A - row.null_A_mean;

row.null_V_mean = mean(null_V);
row.null_V_lo95 = prctile(null_V, 2.5);
row.null_V_hi95 = prctile(null_V, 97.5);
row.corr_NMI_V  = NMI_V - row.null_V_mean;

row.null_CNMI_Yobj_given_V_mean = mean(null_CYobjV);
row.null_CNMI_Yobj_given_V_lo95 = prctile(null_CYobjV, 2.5);
row.null_CNMI_Yobj_given_V_hi95 = prctile(null_CYobjV, 97.5);
row.corr_CNMI_Yobj_given_V = CNMI_Yobj_given_V - row.null_CNMI_Yobj_given_V_mean;

row.null_CNMI_Ysup_given_V_mean = mean(null_CYsupV);
row.null_CNMI_Ysup_given_V_lo95 = prctile(null_CYsupV, 2.5);
row.null_CNMI_Ysup_given_V_hi95 = prctile(null_CYsupV, 97.5);
row.corr_CNMI_Ysup_given_V = CNMI_Ysup_given_V - row.null_CNMI_Ysup_given_V_mean;

end

function S = summarise_seed_rows(T, layerName, k, pcaDim)

metrics = T.Properties.VariableNames;
metrics = metrics(startsWith(metrics, {'H_C','NMI_','CNMI_','corr_','null_'}));

S = table();
S.layer = string(layerName);
S.k = k;
S.pcaDim = pcaDim;
S.nSeeds = height(T);

for i = 1:numel(metrics)
    m = metrics{i};
    vals = T.(m);
    S.([m '_mean']) = mean(vals, 'omitnan');
    S.([m '_sd'])   = std(vals, 0, 'omitnan');
end

end

function H = entropy_discrete(X)
X = categorical(X);
counts = countcats(removecats(X));
p = counts / sum(counts);
p = p(p > 0);
H = -sum(p .* log2(p));
end

function I = mutual_information_discrete(X, Y)
I = entropy_discrete(X) + entropy_discrete(Y) - joint_entropy_discrete(X, Y);
I = max(I, 0);
end

function H = joint_entropy_discrete(X, Y)
X = categorical(X);
Y = categorical(Y);
[~, ~, ix] = unique(X);
[~, ~, iy] = unique(Y);
J = [ix(:), iy(:)];
[~, ~, ij] = unique(J, 'rows');
counts = accumarray(ij, 1);
p = counts / sum(counts);
p = p(p > 0);
H = -sum(p .* log2(p));
end

function val = nmi_target(C, T)
HT = entropy_discrete(T);
if HT <= eps
    val = 0;
else
    val = mutual_information_discrete(C, T) / HT;
end
end

function val = conditional_nmi_target(C, Y, V)
Icond = conditional_mutual_information(C, Y, V);
HYV = conditional_entropy(Y, V);
if HYV <= eps
    val = 0;
else
    val = Icond / HYV;
end
end

function Icond = conditional_mutual_information(C, Y, V)
C = categorical(C);
Y = categorical(Y);
V = categorical(V);
levels = categories(removecats(V));
Icond = 0;
n = numel(V);
for i = 1:numel(levels)
    idx = V == levels{i};
    if any(idx)
        w = sum(idx) / n;
        Icond = Icond + w * mutual_information_discrete(C(idx), Y(idx));
    end
end
end

function Hcond = conditional_entropy(Y, V)
Y = categorical(Y);
V = categorical(V);
levels = categories(removecats(V));
Hcond = 0;
n = numel(V);
for i = 1:numel(levels)
    idx = V == levels{i};
    if any(idx)
        w = sum(idx) / n;
        Hcond = Hcond + w * entropy_discrete(Y(idx));
    end
end
end

function nullVals = permutation_nmi(C, T, nPerm)
nullVals = nan(nPerm,1);
T = categorical(T);
for p = 1:nPerm
    Tperm = T(randperm(numel(T)));
    nullVals(p) = nmi_target(C, Tperm);
end
end

function nullVals = permutation_conditional_nmi(C, Y, V, nPerm)
nullVals = nan(nPerm,1);
Y = categorical(Y);
V = categorical(V);
for p = 1:nPerm
    Yperm = shuffle_within_group(Y, V);
    nullVals(p) = conditional_nmi_target(C, Yperm, V);
end
end

function Yperm = shuffle_within_group(Y, G)
Y = categorical(Y);
G = categorical(G);
Yperm = Y;
levels = categories(removecats(G));
for i = 1:numel(levels)
    idx = find(G == levels{i});
    if numel(idx) > 1
        Yperm(idx) = Y(idx(randperm(numel(idx))));
    end
end
end

%% ========================================================================
% Output helpers
% ========================================================================
function make_control_figure(cfg, T)

x = 1:height(T);
xLabels = clean_layer_labels(T.layer);

fig = figure('Color','w','Position',[120 120 980 430]);
hold on;

plot(x, T.NMI_Yobj_mean, '-o', 'LineWidth', 1.2, 'MarkerSize', 5);
plot(x, T.CNMI_Yobj_given_V_mean, '-s', 'LineWidth', 1.2, 'MarkerSize', 5);
plot(x, T.NMI_V_mean, '-^', 'LineWidth', 1.2, 'MarkerSize', 5);
plot(x, T.corr_CNMI_Yobj_given_V_mean, '--s', 'LineWidth', 1.2, 'MarkerSize', 5);

set(gca, ...
    'XTick', x, ...
    'XTickLabel', xLabels, ...
    'XTickLabelRotation', 35, ...
    'TickLabelInterpreter', 'none', ...
    'Box','off', ...
    'TickDir','out', ...
    'TickLength',[.01 .01], ...
    'LineWidth',0.5, ...
    'FontName','Times New Roman', ...
    'FontSize',9);

ylabel('normalised information', 'Interpreter','none');
xlabel('ResNet-50 layer', 'Interpreter','none');
legend({'object NMI', 'object NMI conditioned on nuisance', ...
        'nuisance NMI', 'null-corrected conditional object NMI'}, ...
        'Location','northoutside', 'Orientation','horizontal', 'Box','off');

title(sprintf('Conditional network control (%s mode)', cfg.runMode), 'Interpreter','none');

mx = max([T.NMI_Yobj_mean; T.CNMI_Yobj_given_V_mean; T.NMI_V_mean; T.corr_CNMI_Yobj_given_V_mean]);
ylim([0, max(0.05, mx * 1.15)]);

outPng = fullfile(cfg.figDir, sprintf('figure_nn_resnet50_v4_conditional_controls_%s.png', cfg.runMode));
outPdf = fullfile(cfg.figDir, sprintf('figure_nn_resnet50_v4_conditional_controls_%s.pdf', cfg.runMode));
print(fig, outPng, '-dpng', '-r300');
print(fig, outPdf, '-dpdf', '-painters');
fprintf('  %s\n  %s\n', outPng, outPdf);

end

function labels = clean_layer_labels(layerNames)
labels = cellstr(layerNames);
for i = 1:numel(labels)
    labels{i} = strrep(labels{i}, 'activation_', 'act ');
    labels{i} = strrep(labels{i}, '_relu', '');
    labels{i} = strrep(labels{i}, 'avg_pool', 'avg pool');
end
end

function write_latex_control_table(T, outFile)

fid = fopen(outFile, 'w');
if fid < 0
    error('Could not write LaTeX table: %s', outFile);
end

fprintf(fid, '%% Generated by main_nn_entropic_compression_resnet50_controls_v4_SWITCHED.m\n');
fprintf(fid, '\\begin{table}[t]\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '\\begin{revthreeblock}\n');
fprintf(fid, '\\caption{Conditional and baseline-corrected control analysis for the ResNet-50 nuisance-control run. Values are means across k-means initialisations. \\(Y_{\\mathrm{obj}}\\) denotes CIFAR-10 object class and \\(V\\) denotes nuisance condition. Conditional object information reports \\(I(C_{l,k};Y_{\\mathrm{obj}}\\mid V)/H(Y_{\\mathrm{obj}}\\mid V)\\). Null-corrected conditional object information subtracts the mean within-nuisance permutation baseline.}\n');
fprintf(fid, '\\label{tab:network_conditional_control_summary}\n');
fprintf(fid, '\\footnotesize\n');
fprintf(fid, '\\setlength{\\tabcolsep}{5pt}\n');
fprintf(fid, '\\begin{tabular}{lccccc}\n');
fprintf(fid, '\\toprule\n');
fprintf(fid, 'Layer & \\(H(C)\\) & \\(Y_{\\mathrm{obj}}\\) & \\(V\\) & \\(Y_{\\mathrm{obj}}\\mid V\\) & corrected \\(Y_{\\mathrm{obj}}\\mid V\\) \\\\\n');
fprintf(fid, '\\midrule\n');

for i = 1:height(T)
    fprintf(fid, '\\texttt{%s} & %.3f & %.3f & %.3f & %.3f & %.3f \\\\\n', ...
        char(T.layer(i)), ...
        T.H_C_mean(i), ...
        T.NMI_Yobj_mean(i), ...
        T.NMI_V_mean(i), ...
        T.CNMI_Yobj_given_V_mean(i), ...
        T.corr_CNMI_Yobj_given_V_mean(i));
end

fprintf(fid, '\\bottomrule\n');
fprintf(fid, '\\end{tabular}\n');
fprintf(fid, '\\end{revthreeblock}\n');
fprintf(fid, '\\end{table}\n');
fclose(fid);

end
