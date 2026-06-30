%% redraw_mild_balanced_readout_5panel_v4_greyscale_manual_errors.m
% Five-panel manuscript figure for the mild balanced supervised-readout
% analysis.
%
% Layout:
%   Row 1: two wide readout-across-layer panels
%       A  object-trained
%       B  nuisance-trained
%
%   Row 2: three panels
%       C  target-balance index across layers
%       D  PCA sensitivity, object-trained / avg pool
%       E  PCA sensitivity, nuisance-trained / avg pool
%
% Figure-only script:
%   - no retraining
%   - no activation extraction
%   - no PCA
%   - no readout reruns
%
% Visual convention:
%   Points denote means across repeated stratified readout splits.
%   Error bars denote bootstrap 95% confidence intervals.
%   Lines connect discrete layer-wise estimates only as visual guidance.

clear; clc; close all;

cfg = struct();
cfg.projectRoot = 'I:\entropicCompression';
cfg.resultsDir  = fullfile(cfg.projectRoot, 'results');
cfg.figDir      = fullfile(cfg.projectRoot, 'figures');
cfg.analysisID  = 'mild_balanced_supervised_readout_10class_v2_extended';
cfg.mainPcaDim  = 50;

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

cfg.colObject   = [0.10 0.10 0.10];   % dark grey
cfg.colNuisance = [0.45 0.45 0.45];   % medium grey

cfg.colObjModel = cfg.colObject;
cfg.colNuiModel = cfg.colNuisance;

cfg.lineObject   = '-';
cfg.lineNuisance = '-';

cfg.markerObject   = 'o';
cfg.markerNuisance = 'o';

cfg.lineWidth  = 1;
cfg.errWidth   = 0.85;
cfg.markerSize = 1.5;

cfg.capWidth    = 0.16;  % layer-axis cap width
cfg.capWidthPCA = 5.0;   % PCA-axis cap width

summaryFile = fullfile(cfg.resultsDir, [cfg.analysisID '_summary.csv']);
mainFile    = fullfile(cfg.resultsDir, [cfg.analysisID '_mainPCA50_summary.csv']);

if ~exist(summaryFile, 'file')
    error('Summary file not found:\n  %s', summaryFile);
end
if ~exist(mainFile, 'file')
    error('Main-PCA summary file not found:\n  %s', mainFile);
end

T     = readtable(summaryFile, 'Delimiter', ',', 'PreserveVariableNames', true);
Tmain = readtable(mainFile,    'Delimiter', ',', 'PreserveVariableNames', true);

make_5panel_figure(cfg, Tmain, T);

fprintf('\nFive-panel greyscale manually errored figure redrawn only. No analysis rerun.\n');

%% ========================================================================
% Main five-panel figure
% ========================================================================
function make_5panel_figure(cfg, Tmain, Tall)

models = unique(string(Tmain.model), 'stable');
layers = string(cfg.layers);
xBase  = 1:numel(layers);

fig = figure('Color','w', 'Position',[70 70 900 650]);
tl = tiledlayout(fig, 2, 8, 'TileSpacing','compact', 'Padding','compact');

% Row 1: two wide panels, each spanning three columns.
axA = nexttile(tl, 1, [1 4]);
plot_readout_panel(axA, Tmain, models(1), cfg, 'Object-trained');

axB = nexttile(tl, 5, [1 4]);
plot_readout_panel(axB, Tmain, models(2), cfg, 'Nuisance-trained');

% Row 2: three panels, each spanning two columns.
axC = nexttile(tl, 9, [1 4]);
plot_difference_panel(axC, Tmain, models, layers, xBase, cfg, 'Target-balance index');

axD = nexttile(tl, 13, [1 2]);
plot_pca_raw_panel(axD, Tall, models(1), cfg, 'Object-trained / avg pool');
axE = nexttile(tl, 15, [1 2]);
plot_pca_raw_panel(axE, Tall, models(2), cfg, 'Nuisance-trained / avg pool');

ax = [axA axB axC axD axE];

% Global formatting.
set(findall(fig, '-property', 'FontName'), 'FontName', 'Times New Roman');
set(findall(fig, '-property', 'FontSize'), 'FontSize', 9);

% Add external subplot labels. A--C use the usual labelSubplots
% placement. D and E are positioned manually because the smaller bottom-row
% PCA panels otherwise place their labels too close to the panel titles.
if exist('labelSubplots', 'file') == 2
    labelSubplots([axA axB axC], {'A','B','C'}, [0.1 0.03], false, ...
        'FontSize', 14, ...
        'FontName', 'Times New Roman', ...
        'FontWeight', 'normal');
else
    local_label_subplots([axA axB axC], {'A','B','C'}, [0.1 0.03], ...
        'FontSize', 14, ...
        'FontName', 'Times New Roman', ...
        'FontWeight', 'normal');
end

text(axD, -0.24, 1.075, 'D', ...
    'Units', 'normalized', ...
    'FontSize', 14, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal', ...
    'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'top', ...
    'Clipping', 'off');

text(axE, -0.24, 1.075, 'E', ...
    'Units', 'normalized', ...
    'FontSize', 14, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal', ...
    'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'top', ...
    'Clipping', 'off');

outPng = fullfile(cfg.figDir, [cfg.analysisID '_5PANEL_GREYSCALE_MANUALERR.png']);
outPdf = fullfile(cfg.figDir, [cfg.analysisID '_5PANEL_GREYSCALE_MANUALERR.pdf']);

print(fig, outPng, '-dpng', '-r300');
print(fig, outPdf, '-dpdf', '-bestfit');

fprintf('Saved five-panel labelled figure:\n  %s\n  %s\n', outPng, outPdf);

end

%% ========================================================================
% Panel helpers
% ========================================================================
function plot_readout_panel(ax, Tmain, modelName, cfg, panelTitle)

hold(ax, 'on');

layers = string(cfg.layers);
M = Tmain(string(Tmain.model)==modelName,:);
M = sort_by_layer(M, layers);
x = 1:height(M);

yObj = M.readout_object_NMI_mean;
eObjLow = M.readout_object_NMI_mean - M.readout_object_NMI_ci_low;
eObjHigh = M.readout_object_NMI_ci_high - M.readout_object_NMI_mean;

yNui = M.readout_nuisance_NMI_mean;
eNuiLow = M.readout_nuisance_NMI_mean - M.readout_nuisance_NMI_ci_low;
eNuiHigh = M.readout_nuisance_NMI_ci_high - M.readout_nuisance_NMI_mean;

h1 = plot(ax, x, yObj, ...
    [cfg.lineObject cfg.markerObject], ...
    'Color', cfg.colObject, ...
    'MarkerFaceColor', 'w', ...
    'MarkerEdgeColor', cfg.colObject, ...
    'LineWidth', cfg.lineWidth, ...
    'MarkerSize', cfg.markerSize);

h2 = plot(ax, x, yNui, ...
    [cfg.lineNuisance cfg.markerNuisance], ...
    'Color', cfg.colNuisance, ...
    'MarkerFaceColor', 'w', ...
    'MarkerEdgeColor', cfg.colNuisance, ...
    'LineWidth', cfg.lineWidth, ...
    'MarkerSize', cfg.markerSize);

draw_manual_errorbars(ax, x, ...
    M.readout_object_NMI_ci_low, ...
    M.readout_object_NMI_ci_high, ...
    cfg.colObject, cfg.capWidth, cfg.errWidth);

draw_manual_errorbars(ax, x, ...
    M.readout_nuisance_NMI_ci_low, ...
    M.readout_nuisance_NMI_ci_high, ...
    cfg.colNuisance, cfg.capWidth, cfg.errWidth);

format_layer_axis(ax, cfg.layers);
ylim(ax, [0.25 0.90]);
ylabel(ax, 'Readout normalised information');
xlabel(ax, 'Network layer');
title(ax, panelTitle, ...
    'FontWeight','normal', ...
    'HorizontalAlignment','center', ...
    'Interpreter','none');

legend(ax, [h1 h2], {'Object readout','Nuisance readout'}, ...
    'Location','northwest', ...
    'Orientation','vertical', ...
    'Box','off');

end

function plot_difference_panel(ax, Tmain, models, layers, xBase, cfg, panelTitle)

hold(ax, 'on');
offset = [-0.10 0.10];
modelCols    = [cfg.colObjModel; cfg.colNuiModel];
modelLines   = {cfg.lineObject, cfg.lineNuisance};
modelMarkers = {cfg.markerObject, cfg.markerNuisance};

h = gobjects(numel(models),1);

for iM = 1:numel(models)
    M = Tmain(string(Tmain.model)==models(iM),:);
    M = sort_by_layer(M, layers);

    x = xBase + offset(iM);
    y = M.D_NMI_mean;

    h(iM) = plot(ax, x, y, ...
        [modelLines{iM} modelMarkers{iM}], ...
        'Color', modelCols(iM,:), ...
        'MarkerFaceColor', 'w', ...
        'MarkerEdgeColor', modelCols(iM,:), ...
        'LineWidth', cfg.lineWidth, ...
        'MarkerSize', cfg.markerSize);

    draw_manual_errorbars(ax, x, ...
        M.D_NMI_ci_low, ...
        M.D_NMI_ci_high, ...
        modelCols(iM,:), cfg.capWidth, cfg.errWidth);
end

yline(ax, 0, '--', 'LineWidth', 1.0, 'Color', [0.35 0.35 0.35]);

format_layer_axis(ax, layers);
ylabel(ax, 'Object readout minus nuisance readout');
xlabel(ax, 'Network layer');
title(ax, panelTitle, ...
    'FontWeight','normal', ...
    'HorizontalAlignment','center', ...
    'Interpreter','none');

legend(ax, h, cellfun(@clean_model_label, cellstr(models), 'UniformOutput', false), ...
    'Location','northwest', ...
    'Orientation','vertical', ...
    'Box','off');

ylim(ax, [-0.45 0.38]);

end

function plot_pca_raw_panel(ax, Tall, modelName, cfg, panelTitle)

hold(ax, 'on');

M = Tall(string(Tall.model)==modelName & string(Tall.layer)=="avg_pool",:);
M = sortrows(M, 'pcaDim');

h1 = plot(ax, M.pcaDim, M.readout_object_NMI_mean, '-o', ...
    'Color', cfg.colObject, ...
    'MarkerFaceColor', 'w', ...
    'MarkerEdgeColor', cfg.colObject, ...
    'LineWidth', cfg.lineWidth, ...
    'MarkerSize', cfg.markerSize);

h2 = plot(ax, M.pcaDim, M.readout_nuisance_NMI_mean, '-^', ...
    'Color', cfg.colNuisance, ...
    'MarkerFaceColor', 'w', ...
    'MarkerEdgeColor', cfg.colNuisance, ...
    'LineWidth', cfg.lineWidth, ...
    'MarkerSize', cfg.markerSize);

draw_manual_errorbars(ax, M.pcaDim, ...
    M.readout_object_NMI_ci_low, ...
    M.readout_object_NMI_ci_high, ...
    cfg.colObject, cfg.capWidthPCA, cfg.errWidth);

draw_manual_errorbars(ax, M.pcaDim, ...
    M.readout_nuisance_NMI_ci_low, ...
    M.readout_nuisance_NMI_ci_high, ...
    cfg.colNuisance, cfg.capWidthPCA, cfg.errWidth);

set(ax, ...
    'Box','off', ...
    'TickDir','out', ...
    'FontName','Times New Roman', ...
    'FontSize',9, ...
    'LineWidth',0.75);

ylim(ax, [0 1]);
xlim(ax, [20 210]);
xticks(ax, [30 50 100 200]);

xlabel(ax, 'PCA dimensions');
ylabel(ax, 'Readout normalised information');
title(ax, panelTitle, ...
    'FontWeight','normal', ...
    'HorizontalAlignment','center', ...
    'Interpreter','none');

legend(ax, [h1 h2], {'Object readout','Nuisance readout'}, ...
    'Location','southwest', ...
    'Orientation','vertical', ...
    'Box','off');

end

function M = sort_by_layer(M, layers)

[tf,ord] = ismember(string(M.layer), string(layers));
if any(~tf)
    warning('Some layers in table were not found in cfg.layers.');
end
[~,idx] = sort(ord);
M = M(idx,:);

end

function format_layer_axis(ax, layers)

set(ax, ...
    'XTick', 1:numel(layers), ...
    'XTickLabel', clean_layer_labels(layers), ...
    'XTickLabelRotation', 45, ...
    'TickLabelInterpreter','none', ...
    'Box','off', ...
    'TickDir','out', ...
    'FontName','Times New Roman', ...
    'FontSize',9, ...
    'LineWidth',0.75);

end

function labels = clean_layer_labels(layerNames)

labels = cellstr(string(layerNames));
for i = 1:numel(labels)
    labels{i} = strrep(labels{i}, 'activation_', 'act ');
    labels{i} = strrep(labels{i}, '_relu', ' relu');
    labels{i} = strrep(labels{i}, 'avg_pool', 'avg pool');
end

end

function lab = clean_model_label(x)

x = string(x);
switch char(x)
    case 'object_10class_mild_final'
        lab = 'Object-trained';
    case 'nuisance_binary_mild_final'
        lab = 'Nuisance-trained';
    otherwise
        lab = char(x);
end

end

function local_label_subplots(ax, labels, offset, varargin)
% Minimal fallback for labelSubplots. Uses normalised axes positions.
%
% offset(1) moves the label left from the axes left boundary.
% offset(2) moves the label up from the axes top boundary.

for i = 1:numel(ax)
    if ~isgraphics(ax(i), 'axes')
        continue;
    end

    pos = get(ax(i), 'Position');
    x = pos(1) - offset(1) * pos(3);
    y = pos(2) + pos(4) + offset(2) * pos(4);

    annotation(gcf, 'textbox', [x y 0.03 0.03], ...
        'String', labels{i}, ...
        'LineStyle', 'none', ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'bottom', ...
        varargin{:});
end

end

function draw_manual_errorbars(ax, x, yLow, yHigh, col, capWidth, lineWidth)

x     = x(:);
yLow  = yLow(:);
yHigh = yHigh(:);

for i = 1:numel(x)
    line(ax, [x(i) x(i)], [yLow(i) yHigh(i)], ...
        'Color', col, ...
        'LineWidth', lineWidth, ...
        'HandleVisibility', 'off');

    line(ax, [x(i)-capWidth/2 x(i)+capWidth/2], [yLow(i) yLow(i)], ...
        'Color', col, ...
        'LineWidth', lineWidth, ...
        'HandleVisibility', 'off');

    line(ax, [x(i)-capWidth/2 x(i)+capWidth/2], [yHigh(i) yHigh(i)], ...
        'Color', col, ...
        'LineWidth', lineWidth, ...
        'HandleVisibility', 'off');
end

end

