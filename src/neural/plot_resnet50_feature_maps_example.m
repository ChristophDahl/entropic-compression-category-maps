%% plot_resnet50_feature_maps_example.m
% Plot ResNet-50 feature maps for one CIFAR-10 example across four variants.
%
% Example target:
%   horse_clean_00046.png
%
% Figure layout:
%   6 rows  = ResNet-50 layers
%   4 cols  = clean, blurred, pixelated, noise-perturbed
%
% Output:
%   I:\entropicCompression\figures\figure_resnet50_feature_maps_example.png
%   I:\entropicCompression\figures\figure_resnet50_feature_maps_example.pdf

clear; clc; close all;

%% ------------------------------------------------------------------------
% Configuration
% -------------------------------------------------------------------------

thisFile = mfilename('fullpath');
if isempty(thisFile)
    scriptDir = pwd;
else
    scriptDir = fileparts(thisFile);
end
projectRoot = fileparts(fileparts(scriptDir));

imageRoot = fullfile(projectRoot, 'data', 'nn_images_resnet50');
figDir    = fullfile(projectRoot, 'figures');

if ~isfolder(figDir)
    mkdir(figDir);
end

className = 'horse';
imageID   = '00036';

variantLabels = { ...
    'Clean', ...
    'Blurred', ...
    'Pixelated', ...
    'Noise-perturbed'};

variantTokens = { ...
    {'clean'}, ...
    {'blurred', 'blur'}, ...
    {'pixelated', 'pixel'}, ...
    {'noise', 'noisy', 'noise-perturbed', 'perturbed'}};

layersToPlot = { ...
    'activation_1_relu', ...
    'activation_10_relu', ...
    'activation_22_relu', ...
    'activation_40_relu', ...
    'activation_49_relu'};

layerLabels = { ...
    'activation\_1\_relu', ...
    'activation\_10\_relu', ...
    'activation\_22\_relu', ...
    'activation\_40\_relu', ...
    'activation\_49\_relu'};

nVariants = numel(variantLabels);
nLayers   = numel(layersToPlot);

%% ------------------------------------------------------------------------
% Locate image files
% -------------------------------------------------------------------------

imageFiles = cell(1, nVariants);

fprintf('\nSearching image root:\n  %s\n\n', imageRoot);

for v = 1:nVariants
    imageFiles{v} = find_variant_image(imageRoot, className, imageID, variantTokens{v});

    fprintf('%-18s: %s\n', variantLabels{v}, imageFiles{v});
end

%% ------------------------------------------------------------------------
% Load network
% -------------------------------------------------------------------------

fprintf('\nLoading ResNet-50...\n');
net = resnet50;
inputSize = net.Layers(1).InputSize(1:2);

%% ------------------------------------------------------------------------
% Extract feature-map summaries
% -------------------------------------------------------------------------

featureMaps = cell(nLayers, nVariants);

for v = 1:nVariants

    img = imread(imageFiles{v});

    if size(img,3) == 1
        img = repmat(img, 1, 1, 3);
    end

    img = imresize(img, inputSize);

    for l = 1:nLayers

        layerName = layersToPlot{l};

        fprintf('Extracting %-24s | %s\n', layerName, variantLabels{v});

        act = activations(net, img, layerName, ...
            'OutputAs', 'channels');

        featureMaps{l,v} = feature_map_summary(act);
    end
end

%% ------------------------------------------------------------------------
% Normalise feature maps within each layer across variants
% -------------------------------------------------------------------------

for l = 1:nLayers

    allVals = [];

    for v = 1:nVariants
        fmap = featureMaps{l,v};
        allVals = [allVals; fmap(:)]; %#ok<AGROW>
    end

    mn = min(allVals, [], 'omitnan');
    mx = max(allVals, [], 'omitnan');

    for v = 1:nVariants
        fmap = featureMaps{l,v};

        if mx > mn
            fmap = (fmap - mn) ./ (mx - mn);
        else
            fmap = zeros(size(fmap));
        end

        featureMaps{l,v} = fmap;
    end
end

%% ------------------------------------------------------------------------
% Plot 6 x 4 figure
% -------------------------------------------------------------------------

fig = figure('Color','w', ...
    'Position',[100 100 620 780]);

t = tiledlayout(fig, nLayers, nVariants, ...
    'TileSpacing','compact', ...
    'Padding','compact');

for l = 1:nLayers
    for v = 1:nVariants

        ax = nexttile(t);
        imagesc(ax, featureMaps{l,v});
        axis(ax, 'image');
        axis(ax, 'off');
        colormap(ax, gray);

        if l == 1
            title(ax, variantLabels{v}, ...
                'Interpreter','none', ...
                'FontName','Times New Roman', ...
                'FontSize',10, ...
                'FontWeight','normal');
        end

        if v == 1
            text(ax, -0.18, 0.5, layerLabels{l}, ...
                'Units','normalized', ...
                'Rotation',90, ...
                'HorizontalAlignment','center', ...
                'VerticalAlignment','middle', ...
                'Interpreter','latex', ...
                'FontName','Times New Roman', ...
                'FontSize',10);
        end
    end
end
% 
% annotation(fig, 'textbox', [0.02 0.965 1 0.03], ...
%     'String', sprintf('ResNet-50 feature-map summaries'), ...
%     'Interpreter','none', ...
%     'FontName','Times New Roman', ...
%     'FontSize',11, ...
%     'FontWeight','normal', ...
%     'HorizontalAlignment','center', ...
%     'VerticalAlignment','middle', ...
%     'LineStyle','none');

drawnow;

%% ------------------------------------------------------------------------
% Export
% -------------------------------------------------------------------------

outPNG = fullfile(figDir, 'figure08.png');
outPDF = fullfile(figDir, 'figure08.pdf');

exportgraphics(fig, outPNG, 'Resolution', 600);
exportgraphics(fig, outPDF, 'ContentType','vector');

fprintf('\nSaved feature-map figure:\n  %s\n  %s\n\n', outPNG, outPDF);

%% ========================= LOCAL FUNCTIONS ==============================

function filePath = find_variant_image(imageRoot, className, imageID, tokens)

    allMatches = [];

    for t = 1:numel(tokens)

        token = tokens{t};

        patterns = { ...
            sprintf('*%s*%s*%s*.png', className, token, imageID), ...
            sprintf('*%s*%s*%s*.jpg', className, token, imageID), ...
            sprintf('*%s*%s*%s*.jpeg', className, token, imageID), ...
            sprintf('*%s*%s*%s*.bmp', className, token, imageID), ...
            sprintf('*%s*%s*%s*.png', className, imageID, token), ...
            sprintf('*%s*%s*%s*.jpg', className, imageID, token), ...
            sprintf('*%s*%s*%s*.jpeg', className, imageID, token), ...
            sprintf('*%s*%s*%s*.bmp', className, imageID, token)};

        for p = 1:numel(patterns)
            d = dir(fullfile(imageRoot, '**', patterns{p}));
            allMatches = [allMatches; d(:)]; %#ok<AGROW>
        end
    end

    if isempty(allMatches)
        error('Could not find image for class=%s, imageID=%s, token(s)=%s under:\n%s', ...
            className, imageID, strjoin(tokens, ', '), imageRoot);
    end

    % Prefer shortest full path if duplicates exist.
    fullPaths = strings(numel(allMatches),1);
    for i = 1:numel(allMatches)
        fullPaths(i) = string(fullfile(allMatches(i).folder, allMatches(i).name));
    end

    [~, idx] = min(strlength(fullPaths));
    filePath = char(fullPaths(idx));
end

function fmap = feature_map_summary(act)
% Convert a layer activation tensor into one 2D feature-map summary.
%
% Convolutional layers:
%   act = H x W x C.
%   The plotted map is the channel-wise mean activation.
%
% avg_pool:
%   no spatial map remains. The vector is reshaped into a square summary.

    act = squeeze(act);

    if ndims(act) == 3

        fmap = mean(double(act), 3);

    elseif isvector(act)

        v = double(act(:));
        side = ceil(sqrt(numel(v)));

        padded = nan(side^2, 1);
        padded(1:numel(v)) = v;

        fmap = reshape(padded, side, side);

    elseif ismatrix(act)

        fmap = double(act);

    else

        error('Unsupported activation size.');

    end
end