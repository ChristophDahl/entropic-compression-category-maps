%% main_mild_balanced_supervised_readout_10class_v2.m
% Final MILD balanced supervised-readout analysis for target-specific ResNet-50 models.
%
% This is the confirmatory version, not the pilot version.
%
% It addresses the three pilot limitations:
%   1. Low PCA dimensionality:
%        PCA dimensionality is increased to 100.
%   2. Imbalanced nuisance target:
%        The dataset is rebuilt as a balanced 10-class object x clean/perturbed set.
%        For each object class:
%             n clean images
%             n perturbed images
%        The perturbed set is sampled evenly from the available perturbed variants.
%   3. Short training:
%        Target-specific models are trained for 10 epochs.
%
% Main target comparison:
%   object target   : Ysub       = 10 CIFAR object classes
%   nuisance target : Vbinary    = clean / perturbed
%
% Main confirmatory prediction:
%   object-trained model:
%       object readout > nuisance readout
%
%   nuisance-trained model:
%       nuisance readout > object readout
%
% Main output:
%   supervised readout means, SD, SEM, 95% bootstrap CI
%   double-dissociation index:
%       D = object readout - nuisance readout
%
% Expected source files:
%   I:\entropicCompression\results\nn_resnet50_v3_pooled_variants_labels.csv
%   I:\entropicCompression\data\nn_images_resnet50_v3\pooled_variants\
%
% Important:
%   Set cfg.rebuildBalancedLabels = false and cfg.retrainModels = false
%   after a successful final run, so the exact dataset and models remain fixed.

clear; clc; close all;

cfg = make_cfg();
initialise_project(cfg);

diary(cfg.logFile);
cleanupObj = onCleanup(@() diary('off')); %#ok<NASGU>

fprintf('\n=== EXTENDED MILD balanced supervised readout: 10-class object vs clean/perturbed nuisance ===\n');
fprintf('Project root : %s\n', cfg.projectRoot);
fprintf('Analysis ID  : %s\n', cfg.analysisID);
fprintf('Log file     : %s\n', cfg.logFile);
fprintf('Started      : %s\n', datestr(now));

%% ------------------------------------------------------------------------
% 1. Build or load final balanced label table
% -------------------------------------------------------------------------
if cfg.rebuildBalancedLabels || ~exist(cfg.balancedLabelFile, 'file')
    fprintf('\n[1/5] Building final balanced label table...\n');
    build_final_balanced_label_table(cfg);
else
    fprintf('\n[1/5] Using existing final balanced label table:\n  %s\n', cfg.balancedLabelFile);
end

L = readtable(cfg.balancedLabelFile, 'Delimiter', ',', 'PreserveVariableNames', true);
[L, imagePaths] = resolve_image_paths(L, cfg.sourceImageDir);
validate_image_paths(imagePaths);

Yobj = removecats(categorical(L.Ysub));
Vbin = removecats(categorical(L.Vbinary));

fprintf('\nFinal balanced dataset loaded: %d images.\n', height(L));
fprintf('\nObject counts:\n'); tabulate(Yobj);
fprintf('\nNuisance counts:\n'); tabulate(Vbin);
if ismember('V', L.Properties.VariableNames)
    fprintf('\nOriginal variant counts:\n'); tabulate(categorical(L.V));
end

%% ------------------------------------------------------------------------
% 2. Load pretrained ResNet-50
% -------------------------------------------------------------------------
fprintf('\n[2/5] Loading pretrained ResNet-50...\n');
try
    baseNet = resnet50;
catch ME
    fprintf(2, '\nCould not load pretrained ResNet-50.\n');
    fprintf(2, 'Install: Deep Learning Toolbox Model for ResNet-50 Network.\n');
    rethrow(ME);
end
inputSize = baseNet.Layers(1).InputSize(1:2);

%% ------------------------------------------------------------------------
% 3. Train or load target-specific models
% -------------------------------------------------------------------------
fprintf('\n[3/5] Training/loading final target-specific models...\n');

trainSpecs = struct([]);

trainSpecs(1).name = 'object_10class_mild_final';
trainSpecs(1).target = Yobj;
trainSpecs(1).targetName = 'Ysub';

trainSpecs(2).name = 'nuisance_binary_mild_final';
trainSpecs(2).target = Vbin;
trainSpecs(2).targetName = 'Vbinary';

models = struct([]);

for iSpec = 1:numel(trainSpecs)
    spec = trainSpecs(iSpec);
    modelFile = fullfile(cfg.modelDir, sprintf('resnet50_%s_%s.mat', cfg.analysisID, spec.name));

    fprintf('\n--- Model %d/%d: %s ---\n', iSpec, numel(trainSpecs), spec.name);

    if exist(modelFile, 'file') && ~cfg.retrainModels
        fprintf('  Loading existing model:\n  %s\n', modelFile);
        S = load(modelFile, 'netTrained', 'classes', 'accVal', 'spec');
        models(iSpec).name = spec.name; %#ok<AGROW>
        models(iSpec).net = S.netTrained;
        models(iSpec).classes = S.classes;
        models(iSpec).accVal = S.accVal;
        models(iSpec).modelFile = modelFile;
        fprintf('  Stored validation accuracy: %.3f\n', S.accVal);
    else
        [netTrained, classes, accVal] = train_target_resnet(cfg, baseNet, imagePaths, spec.target, inputSize, spec.name);
        save(modelFile, 'netTrained', 'classes', 'accVal', 'spec', '-v7.3');
        fprintf('  Saved model:\n  %s\n', modelFile);
        fprintf('  Validation accuracy: %.3f\n', accVal);

        models(iSpec).name = spec.name; %#ok<AGROW>
        models(iSpec).net = netTrained;
        models(iSpec).classes = classes;
        models(iSpec).accVal = accVal;
        models(iSpec).modelFile = modelFile;
    end
end

%% ------------------------------------------------------------------------
% 4. Extract activations and run final readout diagnostics
% -------------------------------------------------------------------------
fprintf('\n[4/5] Running final activation/readout diagnostics...\n');

summaryRows = table();
repeatRows = table();

for iM = 1:numel(models)
    modelName = models(iM).name;
    net = models(iM).net;

    fprintf('\n=== Model: %s ===\n', modelName);

    for iL = 1:numel(cfg.layers)
        layerName = cfg.layers{iL};
        fprintf('\n--- Layer %d/%d: %s ---\n', iL, numel(cfg.layers), layerName);

        Xraw = load_or_extract_activations(cfg, net, imagePaths, inputSize, modelName, layerName);
        X = prepare_X(cfg, Xraw, layerName);

        for iP = 1:numel(cfg.pcaDimList)
            pcaDim = cfg.pcaDimList(iP);
            pcaDimUse = min([pcaDim, size(X,1)-1, size(X,2)]);

            fprintf('  PCA sensitivity %d/%d: requested %d, using %d PCs.\n', ...
                iP, numel(cfg.pcaDimList), pcaDim, pcaDimUse);
            fprintf('  PCA: %d observations x %d features -> %d PCs\n', size(X,1), size(X,2), pcaDimUse);
            tPca = tic;
            [~, score] = pca(double(X), 'NumComponents', pcaDimUse);
            fprintf('  Finished PCA in %.1f s.\n', toc(tPca));

            % Supervised readouts.
            fprintf('  Supervised readout: object target...\n');
            readObj = repeated_readout(score, Yobj, cfg, 'object');

            fprintf('  Supervised readout: nuisance target...\n');
            readNui = repeated_readout(score, Vbin, cfg, 'nuisance');

            % Optional unsupervised reference, retained in table but not used as
            % the decisive double-dissociation result.
            if cfg.computeUnsupervisedReference
                fprintf('  Unsupervised k-means reference...\n');
                unsup = unsupervised_reference(score, Yobj, Vbin, cfg);
            else
                unsup.object_nmi = NaN;
                unsup.nuisance_nmi = NaN;
                unsup.object_nmi_sd = NaN;
                unsup.nuisance_nmi_sd = NaN;
            end

            % Repeated-run table.
            R = table();
            R.model = repmat(string(modelName), cfg.nReadoutRepeats, 1);
            R.layer = repmat(string(layerName), cfg.nReadoutRepeats, 1);
            R.pcaDim = repmat(pcaDimUse, cfg.nReadoutRepeats, 1);
            R.repeat = (1:cfg.nReadoutRepeats)';
            R.object_NMI = readObj.nmi(:);
            R.nuisance_NMI = readNui.nmi(:);
            R.object_balAcc = readObj.balacc(:);
            R.nuisance_balAcc = readNui.balacc(:);
            R.D_NMI = R.object_NMI - R.nuisance_NMI;
            R.D_balAcc = R.object_balAcc - R.nuisance_balAcc;
            repeatRows = [repeatRows; R]; %#ok<AGROW>

            % Summary table.
            row = summarise_final_readout(modelName, layerName, pcaDimUse, ...
                readObj, readNui, unsup, cfg);
            summaryRows = [summaryRows; row]; %#ok<AGROW>
        end
    end
end

%% ------------------------------------------------------------------------
% 5. Save outputs
% -------------------------------------------------------------------------
fprintf('\n[5/5] Saving final outputs...\n');

summaryFile = fullfile(cfg.resultsDir, sprintf('%s_summary.csv', cfg.analysisID));
repeatFile  = fullfile(cfg.resultsDir, sprintf('%s_repeats.csv', cfg.analysisID));

writetable(summaryRows, summaryFile);
writetable(repeatRows, repeatFile);

mainRows = summaryRows(summaryRows.pcaDim == cfg.mainPcaDim, :);
mainFile = fullfile(cfg.resultsDir, sprintf('%s_mainPCA%d_summary.csv', cfg.analysisID, cfg.mainPcaDim));
writetable(mainRows, mainFile);

make_final_readout_figure(cfg, mainRows);
make_final_D_figure(cfg, mainRows);
make_pca_sensitivity_figure(cfg, summaryRows);
write_final_latex_table(cfg, mainRows);

fprintf('Saved main-PCA summary:\n  %s\n', mainFile);

fprintf('Saved summary:\n  %s\n', summaryFile);
fprintf('Saved repeats:\n  %s\n', repeatFile);

fprintf('\nDone.\nFinished: %s\n', datestr(now));

%% ========================================================================
% Configuration
% ========================================================================
function cfg = make_cfg()

cfg = struct();

cfg.projectRoot = 'I:\entropicCompression';
cfg.dataDir     = fullfile(cfg.projectRoot, 'data');
cfg.resultsDir  = fullfile(cfg.projectRoot, 'results');
cfg.figDir      = fullfile(cfg.projectRoot, 'figures');

cfg.sourceImageDir  = fullfile(cfg.dataDir, 'nn_images_resnet50_v7_10class_mild');
cfg.sourceLabelFile = fullfile(cfg.resultsDir, 'nn_resnet50_v7_10class_mild_labels.csv');

cfg.analysisID = 'mild_balanced_supervised_readout_10class_v2_extended';

cfg.balancedLabelFile = fullfile(cfg.resultsDir, sprintf('%s_labels.csv', cfg.analysisID));
cfg.modelDir = fullfile(cfg.resultsDir, ['trained_models_' cfg.analysisID]);
cfg.cacheDir = fullfile(cfg.resultsDir, ['activation_cache_' cfg.analysisID]);
cfg.logFile  = fullfile(cfg.resultsDir, [cfg.analysisID '_runlog.txt']);

% First final run:
cfg.rebuildBalancedLabels = false;
cfg.retrainModels = false;

% After the final run has succeeded, set both to false for exact reruns:
% cfg.rebuildBalancedLabels = false;
% cfg.retrainModels = false;

% Final confirmatory settings.
cfg.maxCleanPerObjectClass = 120;
cfg.numEpochs = 10;
cfg.pcaDimList = [30 50 100 200];
cfg.mainPcaDim = 50;
cfg.nReadoutRepeats = 50;   % kept fixed for final bootstrap CIs
cfg.testFraction = 0.20;

% Layers for final version.
% Extended layer grid. The very early activation_1_relu layer is deliberately
% excluded because it is expensive and dominated by low-level image statistics.
% The grid below samples early, middle, late, and pooled stages densely enough
% for the final manuscript control.
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

% Training and activation settings.
cfg.miniBatchSize = 32;
cfg.initialLearnRate = 1e-4;
cfg.maxFeaturesForPca = 50000;

% Optional unsupervised reference.
cfg.computeUnsupervisedReference = true;
cfg.k = 10;
cfg.seedList = 1:10;
cfg.kmeansMaxIter = 500;

% Bootstrap CIs for reported means and differences.
cfg.nBootstrap = 2000;
cfg.ciAlpha = 0.05;

end

function initialise_project(cfg)
if ~exist(cfg.resultsDir, 'dir'), mkdir(cfg.resultsDir); end
if ~exist(cfg.figDir, 'dir'), mkdir(cfg.figDir); end
if ~exist(cfg.modelDir, 'dir'), mkdir(cfg.modelDir); end
if ~exist(cfg.cacheDir, 'dir'), mkdir(cfg.cacheDir); end
end

%% ========================================================================
% Balanced subset construction
% ========================================================================
function build_final_balanced_label_table(cfg)

if ~exist(cfg.sourceLabelFile, 'file')
    error('Source label file not found:\n  %s', cfg.sourceLabelFile);
end

L0 = readtable(cfg.sourceLabelFile, 'Delimiter', ',', 'PreserveVariableNames', true);
[L0, imagePaths0] = resolve_image_paths(L0, cfg.sourceImageDir);
validate_image_paths(imagePaths0);

if ~ismember('Ysub', L0.Properties.VariableNames)
    error('Source label file is missing required column: Ysub');
end

% Identify nuisance / variant column. Prefer Vbinary if present for the
% clean/perturbed target, and keep V/Vsource for balancing perturbed variants.
variantCol = find_first(L0.Properties.VariableNames, {'V','variant','nuisance','condition','imageCondition'});
if isempty(variantCol)
    variantCol = 'Vbinary';
end

Ysub = removecats(categorical(L0.Ysub));
Vraw = string(L0.(variantCol));

if ismember('Vbinary', L0.Properties.VariableNames)
    Vbin0 = removecats(categorical(L0.Vbinary));
    isClean = Vbin0 == 'clean';
else
    Vlower = lower(Vraw);
    isClean = contains(Vlower, 'clean');
end
isPerturbed = ~isClean;

classes = categories(Ysub);
if numel(classes) < 10
    error('Expected at least ten object classes, found %d.', numel(classes));
end
classes = classes(1:10);

selected = [];

for c = 1:numel(classes)
    className = classes{c};

    idxClean = find(Ysub == className & isClean);
    idxPertAll = find(Ysub == className & isPerturbed);

    if numel(idxClean) < cfg.maxCleanPerObjectClass
        error('Class %s has only %d clean images; need %d.', ...
            className, numel(idxClean), cfg.maxCleanPerObjectClass);
    end

    rng(11000 + c, 'twister');
    idxClean = idxClean(randperm(numel(idxClean), cfg.maxCleanPerObjectClass));

    % Sample an equal number of perturbed images. If multiple perturbed
    % variants exist, sample as evenly as possible across them.
    idxPert = sample_perturbed_evenly(idxPertAll, Vraw, cfg.maxCleanPerObjectClass, 12000 + c);

    selected = [selected; idxClean(:); idxPert(:)]; %#ok<AGROW>
end

L = L0(selected, :);
L.imagePath = imagePaths0(selected);
L.Vbinary = categorical(ifelse_string(isClean(selected), "clean", "perturbed"));
L.Vsource = categorical(Vraw(selected));

% Ensure canonical target columns.
L.Ysub = categorical(L.Ysub);
if ismember('Ybasic', L.Properties.VariableNames)
    L.Ybasic = categorical(L.Ybasic);
end

L = sortrows(L, {'Ysub','Vbinary'});

writetable(L, cfg.balancedLabelFile);

fprintf('Saved final balanced label table:\n  %s\n', cfg.balancedLabelFile);
fprintf('Balanced subset size: %d images.\n', height(L));
fprintf('\nFinal object counts:\n'); tabulate(categorical(L.Ysub));
fprintf('\nFinal clean/perturbed counts:\n'); tabulate(categorical(L.Vbinary));
fprintf('\nFinal source-variant counts:\n'); tabulate(categorical(L.Vsource));

end

function idxPert = sample_perturbed_evenly(idxPertAll, Vraw, nTotal, seed)

rng(seed, 'twister');

if isempty(idxPertAll)
    error('No perturbed images available for one class.');
end

Vpert = categorical(Vraw(idxPertAll));
vLevels = categories(removecats(Vpert));
nLevels = numel(vLevels);

baseN = floor(nTotal / nLevels);
remainder = nTotal - baseN * nLevels;

idxPert = [];

for v = 1:nLevels
    idxV = idxPertAll(Vpert == vLevels{v});
    nV = baseN + double(v <= remainder);

    if numel(idxV) < nV
        error('Perturbed variant %s has only %d images; need %d.', ...
            vLevels{v}, numel(idxV), nV);
    end

    idxV = idxV(randperm(numel(idxV), nV));
    idxPert = [idxPert; idxV(:)]; %#ok<AGROW>
end

idxPert = idxPert(randperm(numel(idxPert)));

end

function out = ifelse_string(cond, a, b)
out = strings(numel(cond),1);
out(cond) = a;
out(~cond) = b;
end

%% ========================================================================
% Training
% ========================================================================
function [netTrained, classes, accVal] = train_target_resnet(cfg, baseNet, imagePaths, target, inputSize, modelName)

target = removecats(categorical(target));
classes = categories(target);

fprintf('  Classes: %d\n', numel(classes));
disp(classes);

imds = imageDatastore(cellstr(imagePaths), ...
    'ReadFcn', @(f) read_for_resnet(f, inputSize));
imds.Labels = target;

[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

lgraph = layerGraph(baseNet);
numClasses = numel(classes);

newFc = fullyConnectedLayer(numClasses, ...
    'Name', 'fc_final_target', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);

newClass = classificationLayer('Name', 'classoutput_final_target');

lgraph = replaceLayer(lgraph, 'fc1000', newFc);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newClass);

options = trainingOptions('sgdm', ...
    'MiniBatchSize', cfg.miniBatchSize, ...
    'MaxEpochs', cfg.numEpochs, ...
    'InitialLearnRate', cfg.initialLearnRate, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsVal, ...
    'ValidationFrequency', max(10, floor(numel(imdsTrain.Files) / cfg.miniBatchSize)), ...
    'Verbose', true, ...
    'Plots', 'none', ...
    'ExecutionEnvironment', 'auto');

fprintf('  Training %s: %d train, %d validation images, %d epochs.\n', ...
    modelName, numel(imdsTrain.Files), numel(imdsVal.Files), cfg.numEpochs);

netTrained = trainNetwork(imdsTrain, lgraph, options);

YPred = classify(netTrained, imdsVal, 'MiniBatchSize', cfg.miniBatchSize);
accVal = mean(YPred == imdsVal.Labels);

end

%% ========================================================================
% Activation extraction
% ========================================================================
function X = load_or_extract_activations(cfg, net, imagePaths, inputSize, modelName, layerName)

safeLayer = regexprep(layerName, '[^A-Za-z0-9_]', '_');
cacheFile = fullfile(cfg.cacheDir, sprintf('%s_%s_activations.mat', modelName, safeLayer));

if exist(cacheFile, 'file') && ~cfg.retrainModels
    fprintf('  Loading cache:\n  %s\n', cacheFile);
    S = load(cacheFile, 'X');
    X = S.X;
    return;
end

fprintf('  Extracting activations:\n  %s\n', cacheFile);

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

X = single(X);
save(cacheFile, 'X', 'layerName', 'modelName', '-v7.3');
fprintf('  Saved cache:\n  %s\n', cacheFile);

end

function X = prepare_X(cfg, Xraw, layerName)

fprintf('  Raw class: %s; raw size:', class(Xraw));
fprintf(' %d', size(Xraw));
fprintf('\n');

X = single(gather(Xraw));
X = reshape(X, size(X,1), []);

if any(~isfinite(X(:)))
    error('Activation matrix for layer %s contains NaN or Inf.', layerName);
end

keep = std(X,0,1,'omitnan') > 0;
X = X(:,keep);

if size(X,2) > cfg.maxFeaturesForPca
    rng(5000 + sum(double(char(layerName))), 'twister');
    idx = randperm(size(X,2), cfg.maxFeaturesForPca);
    X = X(:,idx);
    fprintf('  Feature cap applied: %d features.\n', cfg.maxFeaturesForPca);
end

X = zscore(X);
X(isnan(X)) = 0;
X = single(X);

fprintf('  Prepared matrix: %d x %d\n', size(X,1), size(X,2));

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
% Supervised readout
% ========================================================================
function out = repeated_readout(X, Y, cfg, targetName)

Y = removecats(categorical(Y));
nRep = cfg.nReadoutRepeats;

acc = nan(nRep,1);
balacc = nan(nRep,1);
nmi = nan(nRep,1);

for r = 1:nRep
    [trainIdx, testIdx] = stratified_holdout(Y, cfg.testFraction, 8000+r);

    Xtrain = X(trainIdx,:);
    Xtest  = X(testIdx,:);
    Ytrain = Y(trainIdx);
    Ytest  = Y(testIdx);

    if numel(categories(Y)) == 2
        mdl = fitclinear(Xtrain, Ytrain, ...
            'Learner','svm', ...
            'Regularization','ridge', ...
            'Solver','lbfgs');
    else
        tpl = templateLinear('Learner','svm', ...
            'Regularization','ridge', ...
            'Solver','lbfgs');
        mdl = fitcecoc(Xtrain, Ytrain, ...
            'Learners',tpl, ...
            'Coding','onevsall');
    end

    Yhat = categorical(predict(mdl, Xtest));

    acc(r) = mean(Yhat == Ytest);
    balacc(r) = balanced_accuracy(Ytest, Yhat);
    nmi(r) = nmi_target(Yhat, Ytest);
end

out.targetName = targetName;
out.acc = acc;
out.balacc = balacc;
out.nmi = nmi;

end

function [trainIdx, testIdx] = stratified_holdout(Y, testFraction, seed)

rng(seed,'twister');
Y = removecats(categorical(Y));
levels = categories(Y);

trainIdx = false(numel(Y),1);
testIdx  = false(numel(Y),1);

for i = 1:numel(levels)
    idx = find(Y == levels{i});
    idx = idx(randperm(numel(idx)));
    nTest = max(1, round(testFraction * numel(idx)));
    testIdx(idx(1:nTest)) = true;
    trainIdx(idx(nTest+1:end)) = true;
end

end

function ba = balanced_accuracy(Ytrue, Ypred)

Ytrue = removecats(categorical(Ytrue));
Ypred = categorical(Ypred);
levels = categories(Ytrue);
rec = nan(numel(levels),1);

for i = 1:numel(levels)
    idx = Ytrue == levels{i};
    rec(i) = mean(Ypred(idx) == levels{i});
end

ba = mean(rec,'omitnan');

end

%% ========================================================================
% Unsupervised reference
% ========================================================================
function unsup = unsupervised_reference(score, Yobj, Vbin, cfg)

unsupObj = nan(numel(cfg.seedList),1);
unsupNui = nan(numel(cfg.seedList),1);

for s = 1:numel(cfg.seedList)
    rng(cfg.seedList(s), 'twister');
    C = kmeans(score, cfg.k, ...
        'Start','plus', ...
        'Replicates',1, ...
        'MaxIter',cfg.kmeansMaxIter, ...
        'Display','off', ...
        'Options',statset('UseParallel',false));
    C = categorical(C);
    unsupObj(s) = nmi_target(C, Yobj);
    unsupNui(s) = nmi_target(C, Vbin);
end

unsup.object_nmi = mean(unsupObj,'omitnan');
unsup.nuisance_nmi = mean(unsupNui,'omitnan');
unsup.object_nmi_sd = std(unsupObj,0,'omitnan');
unsup.nuisance_nmi_sd = std(unsupNui,0,'omitnan');

end

%% ========================================================================
% Summary and bootstrap CIs
% ========================================================================
function row = summarise_final_readout(modelName, layerName, pcaDimUse, readObj, readNui, unsup, cfg)

D_NMI = readObj.nmi(:) - readNui.nmi(:);
D_bal = readObj.balacc(:) - readNui.balacc(:);

ciObjNMI = bootstrap_ci_mean(readObj.nmi, cfg);
ciNuiNMI = bootstrap_ci_mean(readNui.nmi, cfg);
ciDNMI   = bootstrap_ci_mean(D_NMI, cfg);

ciObjBal = bootstrap_ci_mean(readObj.balacc, cfg);
ciNuiBal = bootstrap_ci_mean(readNui.balacc, cfg);
ciDBal   = bootstrap_ci_mean(D_bal, cfg);

row = table();
row.model = string(modelName);
row.layer = string(layerName);
row.pcaDim = pcaDimUse;
row.nRepeats = cfg.nReadoutRepeats;

row.unsup_object_NMI_mean = unsup.object_nmi;
row.unsup_object_NMI_sd = unsup.object_nmi_sd;
row.unsup_nuisance_NMI_mean = unsup.nuisance_nmi;
row.unsup_nuisance_NMI_sd = unsup.nuisance_nmi_sd;

row.readout_object_NMI_mean = mean(readObj.nmi,'omitnan');
row.readout_object_NMI_sd = std(readObj.nmi,0,'omitnan');
row.readout_object_NMI_sem = row.readout_object_NMI_sd / sqrt(numel(readObj.nmi));
row.readout_object_NMI_ci_low = ciObjNMI(1);
row.readout_object_NMI_ci_high = ciObjNMI(2);

row.readout_nuisance_NMI_mean = mean(readNui.nmi,'omitnan');
row.readout_nuisance_NMI_sd = std(readNui.nmi,0,'omitnan');
row.readout_nuisance_NMI_sem = row.readout_nuisance_NMI_sd / sqrt(numel(readNui.nmi));
row.readout_nuisance_NMI_ci_low = ciNuiNMI(1);
row.readout_nuisance_NMI_ci_high = ciNuiNMI(2);

row.D_NMI_mean = mean(D_NMI,'omitnan');
row.D_NMI_sd = std(D_NMI,0,'omitnan');
row.D_NMI_sem = row.D_NMI_sd / sqrt(numel(D_NMI));
row.D_NMI_ci_low = ciDNMI(1);
row.D_NMI_ci_high = ciDNMI(2);

row.readout_object_balAcc_mean = mean(readObj.balacc,'omitnan');
row.readout_object_balAcc_sd = std(readObj.balacc,0,'omitnan');
row.readout_object_balAcc_ci_low = ciObjBal(1);
row.readout_object_balAcc_ci_high = ciObjBal(2);

row.readout_nuisance_balAcc_mean = mean(readNui.balacc,'omitnan');
row.readout_nuisance_balAcc_sd = std(readNui.balacc,0,'omitnan');
row.readout_nuisance_balAcc_ci_low = ciNuiBal(1);
row.readout_nuisance_balAcc_ci_high = ciNuiBal(2);

row.D_balAcc_mean = mean(D_bal,'omitnan');
row.D_balAcc_sd = std(D_bal,0,'omitnan');
row.D_balAcc_ci_low = ciDBal(1);
row.D_balAcc_ci_high = ciDBal(2);

end

function ci = bootstrap_ci_mean(x, cfg)

x = x(:);
x = x(isfinite(x));
n = numel(x);

if n == 0
    ci = [NaN NaN];
    return;
end

bootMeans = nan(cfg.nBootstrap,1);
for b = 1:cfg.nBootstrap
    idx = randi(n, n, 1);
    bootMeans(b) = mean(x(idx), 'omitnan');
end

lo = 100 * cfg.ciAlpha/2;
hi = 100 * (1 - cfg.ciAlpha/2);
ci = prctile(bootMeans, [lo hi]);

end

%% ========================================================================
% Information functions
% ========================================================================
function H = entropy_discrete(X)

X = removecats(categorical(X));
counts = countcats(X);
p = counts / sum(counts);
p = p(p > 0);
H = -sum(p .* log2(p));

end

function H = joint_entropy_discrete(X,Y)

X = removecats(categorical(X));
Y = removecats(categorical(Y));

[~,~,ix] = unique(X);
[~,~,iy] = unique(Y);
[~,~,ij] = unique([ix(:),iy(:)], 'rows');

counts = accumarray(ij,1);
p = counts / sum(counts);
p = p(p > 0);
H = -sum(p .* log2(p));

end

function I = mutual_information_discrete(X,Y)

I = entropy_discrete(X) + entropy_discrete(Y) - joint_entropy_discrete(X,Y);
I = max(I,0);

end

function val = nmi_target(C,T)

HT = entropy_discrete(T);
if HT <= eps
    val = 0;
else
    val = mutual_information_discrete(C,T) / HT;
end

end

%% ========================================================================
% Labels and paths
% ========================================================================
function [L, imagePaths] = resolve_image_paths(L, imageDir)

candidateCols = {'imagePath','imagePathResolved','ImagePath','path','Path','filename','fileName', ...
    'Filename','file','File','imageFile','ImageFile'};

colNames = L.Properties.VariableNames;
pathCol = '';

for i = 1:numel(candidateCols)
    if any(strcmp(colNames, candidateCols{i}))
        pathCol = candidateCols{i};
        break;
    end
end

if isempty(pathCol)
    error('No image path column found.');
end

raw = string(L.(pathCol));
imagePaths = strings(height(L),1);

for i = 1:height(L)
    p = raw(i);
    if isfile(p)
        imagePaths(i) = p;
    else
        imagePaths(i) = string(fullfile(imageDir, char(p)));
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
% Figures and LaTeX
% ========================================================================
function make_final_readout_figure(cfg, T)

models = unique(string(T.model), 'stable');
layers = unique(string(T.layer), 'stable');

fig = figure('Color','w','Position',[120 120 1200 450]);

for iM = 1:numel(models)
    subplot(1,numel(models),iM);
    hold on;

    M = T(string(T.model)==models(iM),:);
    [~,ord] = ismember(string(M.layer), layers);
    [~,idx] = sort(ord);
    M = M(idx,:);

    x = 1:height(M);

    yObj = M.readout_object_NMI_mean;
    eObjLow = M.readout_object_NMI_mean - M.readout_object_NMI_ci_low;
    eObjHigh = M.readout_object_NMI_ci_high - M.readout_object_NMI_mean;

    yNui = M.readout_nuisance_NMI_mean;
    eNuiLow = M.readout_nuisance_NMI_mean - M.readout_nuisance_NMI_ci_low;
    eNuiHigh = M.readout_nuisance_NMI_ci_high - M.readout_nuisance_NMI_mean;

    errorbar(x, yObj, eObjLow, eObjHigh, '-o', 'LineWidth',1.5, 'MarkerSize',5);
    errorbar(x, yNui, eNuiLow, eNuiHigh, '-^', 'LineWidth',1.5, 'MarkerSize',5);

    set(gca, 'XTick', x, ...
        'XTickLabel', clean_layer_labels(M.layer), ...
        'XTickLabelRotation', 35, ...
        'TickLabelInterpreter','none', ...
        'Box','off', ...
        'TickDir','out', ...
        'FontName','Times New Roman', ...
        'FontSize',9);

    ylim([0 1]);
    ylabel('readout normalised information');
    xlabel('layer');
    title(clean_model_label(models(iM)), 'Interpreter','none');
    legend({'object readout','nuisance readout'}, ...
        'Location','northoutside', 'Orientation','horizontal', 'Box','off');
end

outPng = fullfile(cfg.figDir, [cfg.analysisID '_readout_CI.png']);
outPdf = fullfile(cfg.figDir, [cfg.analysisID '_readout_CI.pdf']);

print(fig, outPng, '-dpng', '-r300');
print(fig, outPdf, '-dpdf', '-bestfit');

fprintf('Saved final readout figure:\n  %s\n  %s\n', outPng, outPdf);

end

function make_final_D_figure(cfg, T)

models = unique(string(T.model), 'stable');
layers = unique(string(T.layer), 'stable');

fig = figure('Color','w','Position',[160 160 750 430]);
hold on;

xBase = 1:numel(layers);
offset = [-0.12 0.12];

for iM = 1:numel(models)
    M = T(string(T.model)==models(iM),:);
    [~,ord] = ismember(string(M.layer), layers);
    [~,idx] = sort(ord);
    M = M(idx,:);

    x = xBase + offset(iM);
    y = M.D_NMI_mean;
    eLow = M.D_NMI_mean - M.D_NMI_ci_low;
    eHigh = M.D_NMI_ci_high - M.D_NMI_mean;

    errorbar(x, y, eLow, eHigh, '-o', 'LineWidth',1.5, 'MarkerSize',5);
end

yline(0, '--', 'LineWidth', 1.0);

set(gca, 'XTick', xBase, ...
    'XTickLabel', clean_layer_labels(layers), ...
    'XTickLabelRotation', 35, ...
    'TickLabelInterpreter','none', ...
    'Box','off', ...
    'TickDir','out', ...
    'FontName','Times New Roman', ...
    'FontSize',9);

ylabel('object readout minus nuisance readout');
xlabel('layer');
legend(cellfun(@clean_model_label, cellstr(models), 'UniformOutput', false), ...
    'Location','northoutside', 'Orientation','horizontal', 'Box','off');

outPng = fullfile(cfg.figDir, [cfg.analysisID '_double_dissociation_index.png']);
outPdf = fullfile(cfg.figDir, [cfg.analysisID '_double_dissociation_index.pdf']);

print(fig, outPng, '-dpng', '-r300');
print(fig, outPdf, '-dpdf', '-bestfit');

fprintf('Saved final double-dissociation-index figure:\n  %s\n  %s\n', outPng, outPdf);

end


function make_pca_sensitivity_figure(cfg, T)

models = unique(string(T.model), 'stable');

fig = figure('Color','w','Position',[130 130 1150 450]);

for iM = 1:numel(models)
    subplot(1,numel(models),iM);
    hold on;

    M = T(string(T.model)==models(iM) & string(T.layer)=="avg_pool",:);
    [~,idx] = sort(M.pcaDim);
    M = M(idx,:);

    errorbar(M.pcaDim, M.readout_object_NMI_mean, ...
        M.readout_object_NMI_mean - M.readout_object_NMI_ci_low, ...
        M.readout_object_NMI_ci_high - M.readout_object_NMI_mean, ...
        '-o', 'LineWidth',1.5, 'MarkerSize',5);

    errorbar(M.pcaDim, M.readout_nuisance_NMI_mean, ...
        M.readout_nuisance_NMI_mean - M.readout_nuisance_NMI_ci_low, ...
        M.readout_nuisance_NMI_ci_high - M.readout_nuisance_NMI_mean, ...
        '-^', 'LineWidth',1.5, 'MarkerSize',5);

    set(gca, 'Box','off', ...
        'TickDir','out', ...
        'FontName','Times New Roman', ...
        'FontSize',9);

    ylim([0 1]);
    xlabel('PCA dimensions');
    ylabel('readout normalised information');
    title([clean_model_label(models(iM)) ' / avg pool'], 'Interpreter','none');
    legend({'object readout','nuisance readout'}, ...
        'Location','best', 'Box','off');
end

outPng = fullfile(cfg.figDir, [cfg.analysisID '_PCA_sensitivity_avg_pool.png']);
outPdf = fullfile(cfg.figDir, [cfg.analysisID '_PCA_sensitivity_avg_pool.pdf']);

print(fig, outPng, '-dpng', '-r300');
print(fig, outPdf, '-dpdf', '-bestfit');

fprintf('Saved PCA-sensitivity figure:\n  %s\n  %s\n', outPng, outPdf);

end


function write_final_latex_table(cfg, T)

outFile = fullfile(cfg.resultsDir, [cfg.analysisID '_table.tex']);
fid = fopen(outFile,'w');

if fid < 0
    error('Could not write table: %s', outFile);
end

fprintf(fid, '%% Generated by main_mild_balanced_supervised_readout_10class_v2_extended.m\n');
fprintf(fid, '\\begin{table}[t]\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '\\begin{revthreeblock}\n');
fprintf(fid, '\\caption{Final MILD balanced supervised-readout analysis. Networks were trained either on ten-class object identity or on clean--perturbed mild nuisance status. Values report normalised mutual information between decoder predictions and target labels. The double-dissociation index is object readout minus nuisance readout.}\n');
fprintf(fid, '\\label{tab:final_balanced_supervised_readout_10class}\n');
fprintf(fid, '\\footnotesize\n');
fprintf(fid, '\\begin{tabular}{llccc}\n');
fprintf(fid, '\\toprule\n');
fprintf(fid, 'Model & Layer & object readout & nuisance readout & difference \\\\\n');
fprintf(fid, '\\midrule\n');

for i = 1:height(T)
    fprintf(fid, '%s & \\texttt{%s} & %.3f [%.3f, %.3f] & %.3f [%.3f, %.3f] & %.3f [%.3f, %.3f] \\\\\n', ...
        clean_model_label(T.model(i)), char(T.layer(i)), ...
        T.readout_object_NMI_mean(i), T.readout_object_NMI_ci_low(i), T.readout_object_NMI_ci_high(i), ...
        T.readout_nuisance_NMI_mean(i), T.readout_nuisance_NMI_ci_low(i), T.readout_nuisance_NMI_ci_high(i), ...
        T.D_NMI_mean(i), T.D_NMI_ci_low(i), T.D_NMI_ci_high(i));
end

fprintf(fid, '\\bottomrule\n');
fprintf(fid, '\\end{tabular}\n');
fprintf(fid, '\\end{revthreeblock}\n');
fprintf(fid, '\\end{table}\n');

fclose(fid);
fprintf('Saved LaTeX table:\n  %s\n', outFile);

end

function labels = clean_layer_labels(layerNames)

labels = cellstr(layerNames);
for i = 1:numel(labels)
    labels{i} = strrep(labels{i}, 'activation_', 'act ');
    labels{i} = strrep(labels{i}, '_relu', '');
    labels{i} = strrep(labels{i}, 'avg_pool', 'avg pool');
end

end

function lab = clean_model_label(x)

x = string(x);
switch char(x)
    case 'object_10class_mild_final'
        lab = 'object-trained';
    case 'nuisance_binary_mild_final'
        lab = 'nuisance-trained';
    otherwise
        lab = char(x);
end

end
