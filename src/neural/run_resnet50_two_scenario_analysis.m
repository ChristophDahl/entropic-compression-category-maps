%% run_resnet50_two_scenario_analysis.m
% Full neural-network entropic-compression pipeline.
%
% Purpose
% -------
% Runs two complementary ResNet-50 analyses:
%
%   (1) CLEAN-ONLY OBJECT RUN
%       Images: clean CIFAR-10 only.
%       Question: do layer-derived category maps preserve object-relevant
%       information when nuisance transformations are absent?
%
%   (2) POOLED-NUISANCE RUN
%       Images: clean / blur / pixelated / noise variants.
%       Question: when nuisance variation is present, do unsupervised
%       layer-derived category maps preserve object information or nuisance
%       information?
%
% Input
% -----
%   I:\entropicCompression\data\cifar-10-batches-mat
%
% Output
% ------
%   I:\entropicCompression\data\nn_images_resnet50\clean_only\
%   I:\entropicCompression\data\nn_images_resnet50\pooled_variants\
%
%   I:\entropicCompression\results\resnet50_two_scenarios_clean_only_results.csv
%   I:\entropicCompression\results\resnet50_two_scenarios_pooled_variants_results.csv
%   I:\entropicCompression\results\resnet50_two_scenarios_combined_results.csv
%
%   I:\entropicCompression\figures\figure_resnet50_two_scenarios_*.png/pdf
%
% Targets
% -------
%   Ysub   = CIFAR-10 fine class
%   Ybasic = animal / vehicle
%   A      = environmental / affordance-like class: air / land / water
%   V      = nuisance condition: clean / blur / pixelated / noise
%
% Notes
% -----
% This script is intentionally self-contained. It does not depend on the
% previous prepare_cifar10... or main_nn... scripts. This avoids problems
% caused by scripts that clear the caller workspace.
%
% Recommended first test:
%   cfg.scenario(1).maxPerClass = 50;
%   cfg.scenario(2).maxPerClass = 50;
%   cfg.maxImagesAnalysis = 600;
%
% Recommended fuller run:
%   cfg.scenario(1).maxPerClass = 300;
%   cfg.scenario(2).maxPerClass = 300;
%   cfg.maxImagesAnalysis = 3000;

clear; clc; close all;

%% ------------------------- CONFIGURATION -------------------------------
cfg = struct();

thisFile = mfilename('fullpath');
if isempty(thisFile)
    scriptDir = pwd;
else
    scriptDir = fileparts(thisFile);
end
cfg.projectRoot = fileparts(fileparts(scriptDir));
cfg.programDir  = fullfile(cfg.projectRoot, 'programs');
cfg.dataDir     = fullfile(cfg.projectRoot, 'data');
cfg.cifarDir    = fullfile(cfg.dataDir, 'cifar-10-batches-mat');
cfg.imageRoot   = fullfile(cfg.dataDir, 'nn_images_resnet50');
cfg.resultsDir  = fullfile(cfg.projectRoot, 'results');
cfg.figDir      = fullfile(cfg.projectRoot, 'figures');

% Whether to rebuild image folders and label tables.
% Set false only if the corresponding label CSVs and image folders already
% exist and should be reused.
cfg.prepareImages = false;

% Whether to run the ResNet-50 activation/clustering analysis after
% preparing the image folders.
cfg.runAnalysis = true;

% Network
cfg.netName = 'resnet50';
cfg.layerNames = { ...
    'activation_1_relu', ...
    'activation_10_relu', ...
    'activation_22_relu', ...
    'activation_40_relu', ...
    'activation_49_relu', ...
    'avg_pool' ...
};

% Analysis size.
% maxImagesAnalysis is applied after image generation. The subset is
% stratified over Ysub when possible.
cfg.maxImagesAnalysis = 3000;

% Feature processing
cfg.usePCA = true;
cfg.nPC = 30;
cfg.standardizeFeatures = true;

% K-means category maps
cfg.kList = [2 3 4 10];
cfg.kmeansReplicates = 5;
cfg.kmeansMaxIter = 200;

% ResNet batch size. Reduce to 8 if memory becomes tight.
cfg.miniBatchSize = 16;

% Export
cfg.savePNG = true;
cfg.savePDF = true;
cfg.figRes = 300;

% Reproducibility
cfg.seed = 1;

% Scenario 1: clean-only object run
cfg.scenario(1).name = 'clean_only';
cfg.scenario(1).displayName = 'Clean-only object run';
cfg.scenario(1).variants = {'clean'};
cfg.scenario(1).maxPerClass = 300;
cfg.scenario(1).clearExistingImages = true;

% Scenario 2: pooled nuisance run
cfg.scenario(2).name = 'pooled_variants';
cfg.scenario(2).displayName = 'Pooled nuisance run';
cfg.scenario(2).variants = {'clean','blur','pixelated','noise'};
cfg.scenario(2).maxPerClass = 300;
cfg.scenario(2).clearExistingImages = true;

% Variant parameters
cfg.blurSigma  = 1.2;
cfg.pixelBlock = 4;
cfg.noiseSigma = 18;
cfg.imageExt   = '.png';

rng(cfg.seed);

%% ------------------------- INITIALISE ----------------------------------
fprintf('\n============================================================\n');
fprintf('RESNET-50 ENTROPIC-COMPRESSION TWO-RUN PIPELINE V3\n');
fprintf('============================================================\n\n');

fprintf('Project root : %s\n', cfg.projectRoot);
fprintf('CIFAR folder : %s\n', cfg.cifarDir);
fprintf('Image root   : %s\n', cfg.imageRoot);
fprintf('Results      : %s\n', cfg.resultsDir);
fprintf('Figures      : %s\n\n', cfg.figDir);

ensure_dir(cfg.projectRoot);
ensure_dir(cfg.dataDir);
ensure_dir(cfg.imageRoot);
ensure_dir(cfg.resultsDir);
ensure_dir(cfg.figDir);

if ~isfolder(cfg.cifarDir)
    error('CIFAR-10 folder not found: %s', cfg.cifarDir);
end

%% ------------------------- LOAD CIFAR ONCE -----------------------------
fprintf('Loading CIFAR-10 batches...\n');
[cifarData, cifarLabels1, labelNames] = load_cifar10(cfg.cifarDir);
fprintf('Loaded %d CIFAR-10 images.\n', numel(cifarLabels1));
fprintf('Classes:\n');
disp(labelNames);

%% ------------------------- PREPARE IMAGES ------------------------------
for si = 1:numel(cfg.scenario)
    S = cfg.scenario(si);

    scenarioImageDir = fullfile(cfg.imageRoot, S.name);
    scenarioLabelFile = fullfile(cfg.resultsDir, sprintf('resnet50_two_scenarios_%s_labels.csv', S.name));

    cfg.scenario(si).imageDir = scenarioImageDir;
    cfg.scenario(si).labelFile = scenarioLabelFile;

    if cfg.prepareImages
        fprintf('\n============================================================\n');
        fprintf('PREPARING SCENARIO %d/%d: %s\n', si, numel(cfg.scenario), S.displayName);
        fprintf('============================================================\n\n');

        prepare_scenario_images(cifarData, cifarLabels1, labelNames, cfg, S, ...
            scenarioImageDir, scenarioLabelFile);
    else
        fprintf('\nSkipping image preparation for scenario: %s\n', S.name);
        report_file_status('Label CSV', scenarioLabelFile);
        report_folder_status('Image folder', scenarioImageDir);
    end
end

%% ------------------------- LOAD NETWORK --------------------------------
if cfg.runAnalysis
    fprintf('\n============================================================\n');
    fprintf('LOADING NETWORK: %s\n', cfg.netName);
    fprintf('============================================================\n\n');

    net = resnet50;
    inputSize = net.Layers(1).InputSize;
    names = string({net.Layers.Name})';

    fprintf('Network input size: [%d %d %d]\n', inputSize(1), inputSize(2), inputSize(3));

    missingLayers = setdiff(string(cfg.layerNames), names);
    if ~isempty(missingLayers)
        fprintf('Available layer names:\n');
        disp(names);
        error('The following requested layers do not exist: %s', strjoin(missingLayers, ', '));
    end

    fprintf('Using layers:\n');
    disp(string(cfg.layerNames(:)));
end

%% ------------------------- RUN ANALYSES --------------------------------
combinedResults = table;

if cfg.runAnalysis
    for si = 1:numel(cfg.scenario)
        S = cfg.scenario(si);

        fprintf('\n============================================================\n');
        fprintf('RUNNING SCENARIO %d/%d: %s\n', si, numel(cfg.scenario), S.displayName);
        fprintf('============================================================\n\n');

        labelFile = fullfile(cfg.resultsDir, sprintf('resnet50_two_scenarios_%s_labels.csv', S.name));
        imageDir  = fullfile(cfg.imageRoot, S.name);

        Timg = load_image_label_table(labelFile, imageDir);

        fprintf('Images available: %d\n', height(Timg));
        if isfinite(cfg.maxImagesAnalysis) && height(Timg) > cfg.maxImagesAnalysis
            Timg = stratified_subset(Timg, 'Ysub', cfg.maxImagesAnalysis);
            fprintf('Images selected for analysis: %d\n', height(Timg));
        end

        [targets, targetNames, targetDisplayNames] = build_targets(Timg);

        fprintf('\nTargets detected:\n');
        for ti = 1:numel(targetNames)
            tn = targetNames{ti};
            fprintf('  %-8s %2d levels, H = %.3f bits\n', ...
                tn, numel(unique(targets.(tn))), entropy_discrete(targets.(tn)));
        end

        scenarioResults = run_nn_analysis_for_scenario(net, inputSize, Timg, ...
            targets, targetNames, targetDisplayNames, cfg, S);

        outCsv = fullfile(cfg.resultsDir, sprintf('resnet50_two_scenarios_%s_results.csv', S.name));
        outMat = fullfile(cfg.resultsDir, sprintf('resnet50_two_scenarios_%s_results.mat', S.name));

        writetable(scenarioResults, outCsv);
        save(outMat, 'scenarioResults', 'Timg', 'targets', 'targetNames', ...
            'targetDisplayNames', 'cfg', 'S');

        fprintf('\nSaved scenario results:\n');
        fprintf('  %s\n', outCsv);
        fprintf('  %s\n', outMat);

        make_nn_figures(scenarioResults, targetNames, targetDisplayNames, cfg, S);

        combinedResults = [combinedResults; scenarioResults]; %#ok<AGROW>
    end

    combinedCsv = fullfile(cfg.resultsDir, 'resnet50_two_scenarios_combined_results.csv');
    combinedMat = fullfile(cfg.resultsDir, 'resnet50_two_scenarios_combined_results.mat');
    writetable(combinedResults, combinedCsv);
    save(combinedMat, 'combinedResults', 'cfg');

    make_comparison_figures(combinedResults, cfg);

    fprintf('\nCombined results saved:\n');
    fprintf('  %s\n', combinedCsv);
    fprintf('  %s\n', combinedMat);
end

%% ------------------------- DONE ----------------------------------------
fprintf('\n============================================================\n');
fprintf('V3 TWO-RUN PIPELINE COMPLETED\n');
fprintf('============================================================\n\n');

fprintf('Results folder: %s\n', cfg.resultsDir);
fprintf('Figures folder: %s\n', cfg.figDir);

%% ========================= LOCAL FUNCTIONS =============================

function [allData, allLabels1, labelNames] = load_cifar10(cifarDir)
    metaFile = fullfile(cifarDir, 'batches.meta.mat');
    if ~isfile(metaFile)
        error('Missing CIFAR metadata file: %s', metaFile);
    end

    M = load(metaFile);
    labelNames = string(M.label_names(:));

    allData = [];
    allLabels = [];

    for b = 1:5
        f = fullfile(cifarDir, sprintf('data_batch_%d.mat', b));
        if ~isfile(f)
            error('Missing CIFAR batch: %s', f);
        end
        B = load(f);
        allData = [allData; B.data]; %#ok<AGROW>
        allLabels = [allLabels; double(B.labels)]; %#ok<AGROW>
    end

    f = fullfile(cifarDir, 'test_batch.mat');
    if ~isfile(f)
        error('Missing CIFAR test batch: %s', f);
    end
    B = load(f);
    allData = [allData; B.data];
    allLabels = [allLabels; double(B.labels)];

    % CIFAR labels are 0..9. Convert to 1..10.
    allLabels1 = allLabels + 1;
end

function prepare_scenario_images(cifarData, cifarLabels1, labelNames, cfg, S, imageDir, labelFile)

    fprintf('Scenario image folder: %s\n', imageDir);
    fprintf('Scenario label file  : %s\n', labelFile);
    fprintf('Variants             : %s\n', strjoin(S.variants, ', '));
    fprintf('Original per class   : %d\n\n', S.maxPerClass);

    if S.clearExistingImages && isfolder(imageDir)
        fprintf('Removing existing scenario image folder: %s\n', imageDir);
        rmdir(imageDir, 's');
    end
    ensure_dir(imageDir);

    % Balanced subset of originals per CIFAR class.
    keep = false(numel(cifarLabels1),1);
    for c = 1:numel(labelNames)
        idx = find(cifarLabels1 == c);
        idx = idx(randperm(numel(idx)));
        nKeep = min(numel(idx), S.maxPerClass);
        keep(idx(1:nKeep)) = true;
    end

    data = cifarData(keep,:);
    labels1 = cifarLabels1(keep);

    nOut = numel(labels1) * numel(S.variants);
    fprintf('Selected %d original images.\n', numel(labels1));
    fprintf('Will export %d images after variants.\n\n', nOut);

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

        for vi = 1:numel(S.variants)
            variant = string(S.variants{vi});
            img = apply_variant(imgClean, variant, cfg);

            rowIdx = rowIdx + 1;

            classFolder = fullfile(imageDir, char(variant), char(className));
            if ~isfolder(classFolder)
                mkdir(classFolder);
            end

            key = char(variant + "_" + className);
            if ~isKey(counterByClassVariant, key)
                counterByClassVariant(key) = 0;
            end
            counterByClassVariant(key) = counterByClassVariant(key) + 1;
            localIdx = counterByClassVariant(key);

            fname = sprintf('%s_%s_%05d%s', char(className), char(variant), localIdx, cfg.imageExt);
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
    writetable(T, labelFile);

    fprintf('Export complete.\n');
    fprintf('Counts by Ybasic and Ysub:\n');
    disp(groupsummary(T, {'Ybasic','Ysub'}));

    fprintf('Counts by A:\n');
    disp(groupsummary(T, 'A'));

    fprintf('Counts by V:\n');
    disp(groupsummary(T, 'V'));
end

function img = cifar_row_to_image(row)
    r = reshape(row(1:1024), 32, 32)';
    g = reshape(row(1025:2048), 32, 32)';
    b = reshape(row(2049:3072), 32, 32)';
    img = uint8(cat(3, r, g, b));
end

function imgOut = apply_variant(img, variant, cfg)
    variant = lower(string(variant));

    switch variant
        case "clean"
            imgOut = img;

        case "blur"
            imgOut = imgaussfilt(img, cfg.blurSigma);

        case "pixelated"
            small = imresize(img, 1/cfg.pixelBlock, 'nearest');
            imgOut = imresize(small, [size(img,1), size(img,2)], 'nearest');

        case "noise"
            x = double(img) + cfg.noiseSigma * randn(size(img));
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
    className = string(className);

    if any(className == ["airplane","bird"])
        a = "air";
    elseif any(className == ["ship"])
        a = "water";
    else
        a = "land";
    end
end

function T = load_image_label_table(labelFile, imageDir)
    if ~isfile(labelFile)
        error('Label file not found: %s', labelFile);
    end

    T = readtable(labelFile, 'TextType','string');

    if ~ismember('imagePath', T.Properties.VariableNames)
        error('Label table must contain imagePath.');
    end

    imagePath = strings(height(T),1);
    for i = 1:height(T)
        p = string(T.imagePath(i));
        if isfile(p)
            imagePath(i) = p;
        else
            imagePath(i) = string(fullfile(imageDir, p));
        end
    end
    T.imagePath = imagePath;

    missing = ~isfile(T.imagePath);
    if any(missing)
        disp(T(missing,:));
        error('Some image paths do not exist.');
    end
end

function [targets, targetNames, targetDisplayNames] = build_targets(Timg)
    targets = struct();
    targetNames = {};
    targetDisplayNames = struct();

    if ismember('Ysub', Timg.Properties.VariableNames)
        targets.Ysub = relabel_consecutive(categorical(Timg.Ysub));
        targetNames{end+1} = 'Ysub';
        targetDisplayNames.Ysub = 'fine class';
    end
    if ismember('Ybasic', Timg.Properties.VariableNames)
        targets.Ybasic = relabel_consecutive(categorical(Timg.Ybasic));
        targetNames{end+1} = 'Ybasic';
        targetDisplayNames.Ybasic = 'basic class';
    end
    if ismember('A', Timg.Properties.VariableNames)
        targets.A = relabel_consecutive(categorical(Timg.A));
        targetNames{end+1} = 'A';
        targetDisplayNames.A = 'environment / affordance-like';
    end
    if ismember('V', Timg.Properties.VariableNames)
        targets.V = relabel_consecutive(categorical(Timg.V));
        targetNames{end+1} = 'V';
        targetDisplayNames.V = 'nuisance condition';
    end

    if isempty(targetNames)
        error('No target variables found.');
    end
end

function Tout = run_nn_analysis_for_scenario(net, inputSize, Timg, targets, targetNames, targetDisplayNames, cfg, S) %#ok<INUSD>

    imds = imageDatastore(Timg.imagePath);
    aug = augmentedImageDatastore(inputSize(1:2), imds, 'ColorPreprocessing', 'gray2rgb');

    Tout = table;

    for li = 1:numel(cfg.layerNames)
        layerName = cfg.layerNames{li};

        fprintf('\n[%d/%d] Extracting activations: %s\n', li, numel(cfg.layerNames), layerName);

        Z = activations(net, aug, layerName, ...
            'OutputAs', 'rows', ...
            'MiniBatchSize', cfg.miniBatchSize);

        Z = double(Z);
        fprintf('  Raw activation matrix: %d images x %d features\n', size(Z,1), size(Z,2));

        if cfg.standardizeFeatures
            Z = zscore_safe(Z);
        end

        if cfg.usePCA
            warnState = warning('off','stats:pca:ColRankDefX');
            [~, score, ~, ~, explained] = pca(Z, ...
                'Centered', false, ...
                'Economy', true, ...
                'Algorithm', 'svd');
            warning(warnState);

            nKeep = min([cfg.nPC, size(score,2), size(score,1)-1]);
            Zred = score(:,1:nKeep);
            fprintf('  PCA dimensions retained: %d; variance explained: %.2f%%\n', ...
                nKeep, sum(explained(1:nKeep)));
        else
            Zred = Z;
        end

        for ki = 1:numel(cfg.kList)
            k = cfg.kList(ki);
            fprintf('  Clustering layer %s with k=%d\n', layerName, k);

            C = kmeans(Zred, k, ...
                'Replicates', cfg.kmeansReplicates, ...
                'MaxIter', cfg.kmeansMaxIter, ...
                'Display', 'off');

            C = relabel_consecutive(C);
            H_C = entropy_discrete(C);

            row = table;
            row.Scenario = string(S.name);
            row.ScenarioDisplay = string(S.displayName);
            row.Network = string(cfg.netName);
            row.Layer = string(layerName);
            row.LayerIndex = li;
            row.K = k;
            row.H_C = H_C;
            row.NImages = height(Timg);
            row.NPC = size(Zred,2);

            for ti = 1:numel(targetNames)
                tn = targetNames{ti};
                y = targets.(tn);
                Ht = entropy_discrete(y);

                if Ht <= eps
                    Ival = 0;
                    Hcond = 0;
                    NMI = 0;
                else
                    Ival = mutual_information_discrete(C, y);
                    Hcond = Ht - Ival;
                    NMI = Ival / Ht;
                end

                row.("H_" + tn) = Ht;
                row.("I_C_" + tn) = Ival;
                row.("H_" + tn + "_given_C") = Hcond;
                row.("NMI_" + tn) = NMI;
            end

            Tout = [Tout; row]; %#ok<AGROW>
        end
    end

    disp(Tout);
end

function Tsub = stratified_subset(T, labelVar, nTotal)
    labels = unique(T.(labelVar));
    nPer = floor(nTotal / numel(labels));

    idxKeep = [];
    for i = 1:numel(labels)
        idx = find(T.(labelVar) == labels(i));
        idx = idx(randperm(numel(idx)));
        idxKeep = [idxKeep; idx(1:min(nPer,numel(idx)))]; %#ok<AGROW>
    end

    remaining = nTotal - numel(idxKeep);
    if remaining > 0
        pool = setdiff((1:height(T))', idxKeep);
        pool = pool(randperm(numel(pool)));
        idxKeep = [idxKeep; pool(1:min(remaining,numel(pool)))];
    end

    idxKeep = idxKeep(randperm(numel(idxKeep)));
    Tsub = T(idxKeep,:);
end

function make_nn_figures(Tout, targetNames, targetDisplayNames, cfg, S)

    kVals = unique(Tout.K);

    for ki = 1:numel(kVals)
        k = kVals(ki);
        Tk = Tout(Tout.K == k,:);
        Tk = sortrows(Tk, 'LayerIndex');

        fig = figure('Color','w','Position',[100 100 1050 560]);
        hold on;

        for ti = 1:numel(targetNames)
            tn = targetNames{ti};
            varName = "NMI_" + tn;

            displayName = tn;
            if isfield(targetDisplayNames, tn)
                displayName = targetDisplayNames.(tn);
            end

            plot(1:height(Tk), Tk.(varName), '-o', ...
                'LineWidth', 1.6, ...
                'DisplayName', char(displayName));
        end

        ylim([0 1.05]);
        xlim([0.75 height(Tk)+0.25]);
        set(gca, 'XTick', 1:height(Tk), ...
                 'XTickLabel', clean_layer_labels(Tk.Layer), ...
                 'XTickLabelRotation', 35);
        ylabel('normalised mutual information I(C_l;T)/H(T)');
        xlabel('network layer');
        title(sprintf('%s: learned category maps, k=%d', S.displayName, k));
        legend('Location','best');
        box off;

        baseName = sprintf('figure_resnet50_two_scenarios_%s_nmi_k%d', S.name, k);
        save_figure(fig, cfg.figDir, baseName, cfg);
    end

    % Heatmap for largest k.
    k = max(kVals);
    Tk = Tout(Tout.K == k,:);
    Tk = sortrows(Tk, 'LayerIndex');

    M = zeros(height(Tk), numel(targetNames));
    xlabels = cell(1,numel(targetNames));

    for ti = 1:numel(targetNames)
        tn = targetNames{ti};
        M(:,ti) = Tk.("NMI_" + tn);
        if isfield(targetDisplayNames, tn)
            xlabels{ti} = targetDisplayNames.(tn);
        else
            xlabels{ti} = tn;
        end
    end

    fig = figure('Color','w','Position',[160 160 900 580]);
    imagesc(M);
    caxis([0 1]);
    colorbar;
    set(gca, 'XTick', 1:numel(targetNames), 'XTickLabel', xlabels, ...
             'YTick', 1:height(Tk), 'YTickLabel', clean_layer_labels(Tk.Layer), ...
             'TickLength', [0 0]);
    xlabel('target variable');
    ylabel('network layer');
    title(sprintf('%s: target information preserved, k=%d', S.displayName, k));

    for i = 1:size(M,1)
        for j = 1:size(M,2)
            text(j,i,sprintf('%.2f',M(i,j)), ...
                'HorizontalAlignment','center', ...
                'FontSize',9);
        end
    end

    baseName = sprintf('figure_resnet50_two_scenarios_%s_heatmap_k%d', S.name, k);
    save_figure(fig, cfg.figDir, baseName, cfg);

    % Complexity figure.
    fig = figure('Color','w','Position',[180 180 950 520]);
    hold on;
    for ki = 1:numel(kVals)
        k = kVals(ki);
        Tk = Tout(Tout.K == k,:);
        Tk = sortrows(Tk, 'LayerIndex');
        plot(1:height(Tk), Tk.H_C, '-o', ...
            'LineWidth', 1.6, ...
            'DisplayName', sprintf('k=%d', k));
    end

    set(gca, 'XTick', 1:height(Tk), ...
             'XTickLabel', clean_layer_labels(Tk.Layer), ...
             'XTickLabelRotation', 35);
    ylabel('category entropy H(C_l) [bits]');
    xlabel('network layer');
    title(sprintf('%s: complexity of learned layer category maps', S.displayName));
    legend('Location','best');
    box off;

    baseName = sprintf('figure_resnet50_two_scenarios_%s_category_entropy', S.name);
    save_figure(fig, cfg.figDir, baseName, cfg);
end

function make_comparison_figures(Tout, cfg)

    if isempty(Tout)
        return
    end

    k = max(unique(Tout.K));
    Tk = Tout(Tout.K == k,:);
    Tk = sortrows(Tk, {'Scenario','LayerIndex'});

    needed = ["NMI_Ysub","NMI_Ybasic","NMI_A","NMI_V"];
    for i = 1:numel(needed)
        if ~ismember(needed(i), string(Tk.Properties.VariableNames))
            return
        end
    end

    scenarios = unique(Tk.Scenario, 'stable');

    for si = 1:numel(scenarios)
        Ts = Tk(Tk.Scenario == scenarios(si),:);
        Ts = sortrows(Ts,'LayerIndex');

        fig = figure('Color','w','Position',[160 160 1050 560]);
        hold on;
        plot(1:height(Ts), Ts.NMI_Ysub, '-o', 'LineWidth', 1.6, 'DisplayName','fine class');
        plot(1:height(Ts), Ts.NMI_Ybasic, '-o', 'LineWidth', 1.6, 'DisplayName','basic class');
        plot(1:height(Ts), Ts.NMI_A, '-o', 'LineWidth', 1.6, 'DisplayName','environment / affordance-like');
        plot(1:height(Ts), Ts.NMI_V, '-o', 'LineWidth', 1.6, 'DisplayName','nuisance condition');

        ylim([0 1.05]);
        xlim([0.75 height(Ts)+0.25]);
        set(gca, 'XTick', 1:height(Ts), ...
                 'XTickLabel', clean_layer_labels(Ts.Layer), ...
                 'XTickLabelRotation', 35);
        ylabel('normalised mutual information I(C_l;T)/H(T)');
        xlabel('network layer');
        title(sprintf('Scenario comparison-ready plot: %s, k=%d', scenarios(si), k));
        legend('Location','best');
        box off;

        baseName = sprintf('figure_resnet50_two_scenarios_comparison_ready_%s_k%d', scenarios(si), k);
        save_figure(fig, cfg.figDir, baseName, cfg);
    end

    % Final-layer comparison bar plot.
    finalLayer = "avg_pool";
    Tf = Tk(Tk.Layer == finalLayer,:);
    if height(Tf) == numel(scenarios)

        M = [Tf.NMI_Ysub, Tf.NMI_Ybasic, Tf.NMI_A, Tf.NMI_V];

        fig = figure('Color','w','Position',[220 220 850 520]);
        bar(M);
        set(gca, 'XTick', 1:height(Tf), 'XTickLabel', Tf.ScenarioDisplay, ...
            'XTickLabelRotation', 15);
        ylabel('normalised mutual information at avg\_pool');
        title(sprintf('Final-layer target preservation across scenarios, k=%d', k));
        legend({'fine class','basic class','environment / affordance-like','nuisance condition'}, ...
            'Location','best');
        box off;

        save_figure(fig, cfg.figDir, sprintf('figure_resnet50_two_scenarios_final_layer_scenario_comparison_k%d', k), cfg);
    end
end

function labels = clean_layer_labels(layerStrings)
    labels = cellstr(string(layerStrings));
    for i = 1:numel(labels)
        labels{i} = strrep(labels{i}, '_', '\_');
    end
end

function X = zscore_safe(X)
    mu = mean(X,1,'omitnan');
    sd = std(X,0,1,'omitnan');
    sd(sd == 0 | isnan(sd)) = 1;
    X = (X - mu) ./ sd;
    X(isnan(X)) = 0;
end

function h = entropy_discrete(x)
    x = relabel_consecutive(x);
    n = numel(x);
    vals = unique(x);
    p = zeros(numel(vals),1);
    for i = 1:numel(vals)
        p(i) = sum(x == vals(i)) / n;
    end
    p = p(p > 0);
    h = -sum(p .* log2(p));
end

function mi = mutual_information_discrete(x,y)
    x = relabel_consecutive(x);
    y = relabel_consecutive(y);

    n = numel(x);
    ux = unique(x);
    uy = unique(y);

    px = zeros(numel(ux),1);
    py = zeros(numel(uy),1);
    pxy = zeros(numel(ux), numel(uy));

    for i = 1:numel(ux)
        px(i) = sum(x == ux(i)) / n;
    end
    for j = 1:numel(uy)
        py(j) = sum(y == uy(j)) / n;
    end
    for i = 1:numel(ux)
        for j = 1:numel(uy)
            pxy(i,j) = sum(x == ux(i) & y == uy(j)) / n;
        end
    end

    mi = 0;
    for i = 1:numel(ux)
        for j = 1:numel(uy)
            if pxy(i,j) > 0
                mi = mi + pxy(i,j) * log2(pxy(i,j) / (px(i)*py(j)));
            end
        end
    end
end

function y = relabel_consecutive(x)
    if iscategorical(x) || isstring(x) || iscellstr(x) || ischar(x)
        [~,~,y] = unique(categorical(x), 'stable');
    else
        [~,~,y] = unique(x, 'stable');
    end
    y = double(y(:));
end

function ensure_dir(d)
    if ~isfolder(d)
        mkdir(d);
    end
end

function report_file_status(label, filePath)
    if isfile(filePath)
        fprintf('  %-24s FOUND    %s\n', label, filePath);
    else
        fprintf('  %-24s MISSING  %s\n', label, filePath);
    end
end

function report_folder_status(label, folderPath)
    if isfolder(folderPath)
        fprintf('  %-24s FOUND    %s\n', label, folderPath);
    else
        fprintf('  %-24s MISSING  %s\n', label, folderPath);
    end
end

function save_figure(figHandle, figDir, baseName, cfg)
    ax = findall(figHandle, 'Type', 'axes');
    for a = reshape(ax,1,[])
        try
            disableDefaultInteractivity(a);
        catch
        end
        try
            if isprop(a, 'Toolbar') && ~isempty(a.Toolbar)
                a.Toolbar.Visible = 'off';
            end
        catch
        end
    end
    drawnow;

    if cfg.savePNG
        exportgraphics(figHandle, fullfile(figDir, [baseName '.png']), ...
            'Resolution', cfg.figRes);
    end
    if cfg.savePDF
        exportgraphics(figHandle, fullfile(figDir, [baseName '.pdf']), ...
            'ContentType','vector');
    end
end
