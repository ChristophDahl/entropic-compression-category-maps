%% prepare_cifar10_variants.m
% Prepare CIFAR-10 for the neural-network entropic-compression analysis.
%
% Input:
%   I:\entropicCompression\data\cifar-10-batches-mat
%
% Output:
%   I:\entropicCompression\data\nn_images\
%   I:\entropicCompression\data\nn_image_labels.csv
%
% Labels generated:
%   Ysub   = CIFAR-10 fine class
%   Ybasic = animal / vehicle
%   A      = environment / affordance-like target: air / land / water
%   V      = image condition / nuisance target: clean / blur / pixelated / noise
%
% The nuisance variants make it possible to ask whether layer-derived
% categories preserve object-relevant information, nuisance information,
% or both.

clear; clc; close all;

%% ------------------------- SETTINGS ------------------------------------
P = struct();

thisFile = mfilename('fullpath');
if isempty(thisFile)
    scriptDir = pwd;
else
    scriptDir = fileparts(thisFile);
end
P.projectRoot = fileparts(fileparts(scriptDir));
P.dataDir     = fullfile(P.projectRoot, 'data');
P.cifarDir    = fullfile(P.dataDir, 'cifar-10-batches-mat');
P.imageDir    = fullfile(P.dataDir, 'nn_images');
P.labelFile   = fullfile(P.dataDir, 'nn_image_labels.csv');

% Use train and test batches.
P.useTrainBatches = true;
P.useTestBatch    = true;

% Number of ORIGINAL images per CIFAR-10 class before nuisance variants.
% Total exported images = maxPerClass x 10 x numel(P.variants).
% Suggested:
%   100 = quick
%   300 = moderate
%   500 = full-ish but still manageable
P.maxPerClass = 300;

% Nuisance variants. For a minimal run, use {'clean'}.
P.variants = {'clean','blur','pixelated','noise'};

% Variant parameters
P.blurSigma       = 1.2;
P.pixelBlock      = 4;      % 4 means 32x32 -> 8x8 -> 32x32
P.noiseSigma      = 18;     % pixel noise SD on 0..255 scale

% Output image format
P.imageExt = '.png';

% Clear old nn_images before export?
P.clearExistingImages = true;

rng(1);

fprintf('\nPreparing CIFAR-10 for neural-network entropic-compression analysis\n');
fprintf('CIFAR folder: %s\n', P.cifarDir);
fprintf('Output images: %s\n', P.imageDir);
fprintf('Label CSV: %s\n', P.labelFile);
fprintf('Original images per class: %d\n', P.maxPerClass);
fprintf('Variants: %s\n\n', strjoin(P.variants, ', '));

%% ------------------------- CHECKS --------------------------------------
if ~isfolder(P.cifarDir)
    error('CIFAR folder not found: %s', P.cifarDir);
end

if P.clearExistingImages && isfolder(P.imageDir)
    fprintf('Removing existing image folder: %s\n', P.imageDir);
    rmdir(P.imageDir, 's');
end
if ~isfolder(P.imageDir)
    mkdir(P.imageDir);
end

metaFile = fullfile(P.cifarDir, 'batches.meta.mat');
if ~isfile(metaFile)
    error('Missing CIFAR metadata file: %s', metaFile);
end

%% ------------------------- LOAD CIFAR ----------------------------------
M = load(metaFile);
labelNames = string(M.label_names(:));

fprintf('CIFAR-10 labels detected:\n');
disp(labelNames);

allData = [];
allLabels = [];

if P.useTrainBatches
    for b = 1:5
        f = fullfile(P.cifarDir, sprintf('data_batch_%d.mat', b));
        if ~isfile(f)
            error('Missing CIFAR batch: %s', f);
        end
        B = load(f);
        allData   = [allData; B.data]; %#ok<AGROW>
        allLabels = [allLabels; double(B.labels)]; %#ok<AGROW>
    end
end

if P.useTestBatch
    f = fullfile(P.cifarDir, 'test_batch.mat');
    if ~isfile(f)
        error('Missing CIFAR test batch: %s', f);
    end
    B = load(f);
    allData   = [allData; B.data];
    allLabels = [allLabels; double(B.labels)];
end

% CIFAR labels are 0..9. Convert to 1..10 for MATLAB indexing.
allLabels1 = allLabels + 1;
fprintf('Loaded %d CIFAR-10 images.\n', numel(allLabels1));

%% ------------------------- BALANCED SUBSET -----------------------------
keep = false(numel(allLabels1),1);
for c = 1:numel(labelNames)
    idx = find(allLabels1 == c);
    idx = idx(randperm(numel(idx)));
    nKeep = min(numel(idx), P.maxPerClass);
    keep(idx(1:nKeep)) = true;
end

data = allData(keep,:);
labels1 = allLabels1(keep);

fprintf('Selected %d original images.\n', numel(labels1));
fprintf('Will export %d images after variants.\n\n', numel(labels1) * numel(P.variants));

%% ------------------------- EXPORT IMAGES + TABLE ------------------------
nOut = numel(labels1) * numel(P.variants);

imagePath = strings(nOut,1);
Ysub      = strings(nOut,1);
Ybasic    = strings(nOut,1);
A         = strings(nOut,1);
V         = strings(nOut,1);
OriginalID = strings(nOut,1);

counterByClassVariant = containers.Map('KeyType','char','ValueType','double');

rowIdx = 0;
for i = 1:numel(labels1)

    className = labelNames(labels1(i));
    imgClean = cifar_row_to_image(data(i,:));

    for vi = 1:numel(P.variants)
        variant = string(P.variants{vi});
        img = apply_variant(imgClean, variant, P);

        rowIdx = rowIdx + 1;

        classFolder = fullfile(P.imageDir, char(variant), char(className));
        if ~isfolder(classFolder)
            mkdir(classFolder);
        end

        key = char(variant + "_" + className);
        if ~isKey(counterByClassVariant, key)
            counterByClassVariant(key) = 0;
        end
        counterByClassVariant(key) = counterByClassVariant(key) + 1;
        localIdx = counterByClassVariant(key);

        fname = sprintf('%s_%s_%05d%s', char(className), char(variant), localIdx, P.imageExt);
        absPath = fullfile(classFolder, fname);
        imwrite(img, absPath);

        relPath = string(fullfile(char(variant), char(className), fname));
        relPath = replace(relPath, '\', '/');

        imagePath(rowIdx) = relPath;
        Ysub(rowIdx)      = className;
        Ybasic(rowIdx)    = map_basic(className);
        A(rowIdx)         = map_affordance_like(className);
        V(rowIdx)         = variant;
        OriginalID(rowIdx)= sprintf('%s_%05d', char(className), i);
    end
end

T = table(imagePath, Ysub, Ybasic, A, V, OriginalID);
writetable(T, P.labelFile);

fprintf('\nExport complete.\n');
fprintf('Images written to: %s\n', P.imageDir);
fprintf('Label CSV written: %s\n\n', P.labelFile);

fprintf('Counts by Ybasic and Ysub:\n');
disp(groupsummary(T, {'Ybasic','Ysub'}));

fprintf('Counts by A:\n');
disp(groupsummary(T, 'A'));

fprintf('Counts by V:\n');
disp(groupsummary(T, 'V'));

fprintf('\nNext run:\n');
fprintf('cd(''%s'')\n', fullfile(P.projectRoot,'programs'));
fprintf('run_resnet50_two_scenario_analysis\n');

%% ========================= LOCAL FUNCTIONS =============================

function img = cifar_row_to_image(row)
    % CIFAR-10 MATLAB rows are 1 x 3072:
    % first 1024 red, next 1024 green, final 1024 blue.
    r = reshape(row(1:1024), 32, 32)';
    g = reshape(row(1025:2048), 32, 32)';
    b = reshape(row(2049:3072), 32, 32)';
    img = uint8(cat(3, r, g, b));
end

function imgOut = apply_variant(img, variant, P)
    variant = lower(string(variant));

    switch variant
        case "clean"
            imgOut = img;

        case "blur"
            imgOut = imgaussfilt(img, P.blurSigma);

        case "pixelated"
            small = imresize(img, 1/P.pixelBlock, 'nearest');
            imgOut = imresize(small, [size(img,1), size(img,2)], 'nearest');

        case "noise"
            x = double(img) + P.noiseSigma * randn(size(img));
            x = min(max(x,0),255);
            imgOut = uint8(x);

        otherwise
            error('Unknown variant: %s', variant);
    end
end

function y = map_basic(className)
    className = string(className);
    animalClasses  = ["bird","cat","deer","dog","frog","horse"];
    vehicleClasses = ["airplane","automobile","ship","truck"];

    if any(className == animalClasses)
        y = "animal";
    elseif any(className == vehicleClasses)
        y = "vehicle";
    else
        y = "other";
    end
end

function a = map_affordance_like(className)
    % This is not a literal action label.
    % It is a constructed environmental / affordance-like grouping.
    className = string(className);

    if any(className == ["airplane","bird"])
        a = "air";
    elseif any(className == ["ship"])
        a = "water";
    else
        a = "land";
    end
end
