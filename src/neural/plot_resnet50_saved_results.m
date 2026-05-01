%% plot_resnet50_saved_results.m
% Plot saved ResNet-50 entropic-compression results.
%
% Purpose
% -------
% This script DOES NOT prepare images and DOES NOT run ResNet-50.
% It reloads previously saved result CSV files and regenerates the
% manuscript-ready network figures:
%
%   figure07  Layer-wise target preservation for clean-only and pooled-nuisance runs
%   figure08  Cluster-target heatmaps if per-image cluster-target files exist;
%             otherwise a final-layer target-preservation heatmap from summary results
%
% Expected existing result files for Figure 7
% ------------------------------------------
%   I:\entropicCompression\results\resnet50_two_scenarios_combined_results.csv
%
% If the combined file does not exist, the script tries to reconstruct it from:
%
%   I:\entropicCompression\results\resnet50_two_scenarios_clean_only_results.csv
%   I:\entropicCompression\results\resnet50_two_scenarios_pooled_variants_results.csv
%
% Optional per-image files for the true cluster-target Figure 8
% ------------------------------------------------------------
%   I:\entropicCompression\results\resnet50_two_scenarios_clean_only_avg_pool_k10_cluster_targets.csv
%   I:\entropicCompression\results\resnet50_two_scenarios_pooled_variants_avg_pool_k10_cluster_targets.csv
%
% Required columns in those optional files:
%   cluster, Yobj, Ysup, A, V
%
% If those files are not present, Figure 8 is still generated from summary
% final-layer NMI values. That fallback is not a cluster-composition plot;
% it is a compact target-preservation heatmap for avg_pool.
%
% Output
% ------
%   I:\entropicCompression\figures\figure07.png/pdf
%   I:\entropicCompression\figures\figure08.png/pdf
%
% Author: Christoph D. Dahl / revised plotting-only V5

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
cfg.resultsDir  = fullfile(cfg.projectRoot, 'results');
cfg.figDir      = fullfile(cfg.projectRoot, 'figures');

% Explicitly keep the expensive parts off.
cfg.prepareImages = false;
cfg.runAnalysis   = false;

% Plotting choices.
cfg.kForPlot   = 10;
cfg.finalLayer = "avg_pool";

% Figure 8 mode:
%   "auto"           = use cluster-target files if present; otherwise fallback to summary heatmap
%   "cluster_target" = require per-image cluster-target files
%   "summary"        = always use final-layer summary heatmap
cfg.figure08Mode = "auto";

% Export.
cfg.savePNG = true;
cfg.savePDF = true;
cfg.figRes  = 300;

% Expected result files.
cfg.combinedCsv = fullfile(cfg.resultsDir, 'resnet50_two_scenarios_combined_results.csv');
cfg.cleanCsv    = fullfile(cfg.resultsDir, 'resnet50_two_scenarios_clean_only_results.csv');
cfg.pooledCsv   = fullfile(cfg.resultsDir, 'resnet50_two_scenarios_pooled_variants_results.csv');

% Optional per-image cluster-target files for Figure 8.
cfg.cleanClusterTargetCsv = fullfile(cfg.resultsDir, ...
    'resnet50_two_scenarios_clean_only_avg_pool_k10_cluster_targets.csv');
cfg.pooledClusterTargetCsv = fullfile(cfg.resultsDir, ...
    'resnet50_two_scenarios_pooled_variants_avg_pool_k10_cluster_targets.csv');

ensure_dir(cfg.resultsDir);
ensure_dir(cfg.figDir);

fprintf('\n============================================================\n');
fprintf('RESNET-50 ENTROPIC-COMPRESSION PLOT-ONLY SCRIPT\n');
fprintf('============================================================\n\n');

fprintf('Project root : %s\n', cfg.projectRoot);
fprintf('Results      : %s\n', cfg.resultsDir);
fprintf('Figures      : %s\n', cfg.figDir);
fprintf('Run analysis : %d\n', cfg.runAnalysis);
fprintf('Prepare imgs : %d\n', cfg.prepareImages);
fprintf('Figure 8 mode: %s\n\n', cfg.figure08Mode);

%% ------------------------- LOAD SAVED RESULTS --------------------------
combinedResults = load_saved_nn_results(cfg);
validate_results_table(combinedResults);

fprintf('\nLoaded %d result rows.\n', height(combinedResults));
fprintf('Available scenarios:\n');
disp(unique(string(combinedResults.Scenario), 'stable'));

fprintf('Available k values:\n');
disp(unique(combinedResults.K)');

fprintf('Available layers:\n');
disp(unique(string(combinedResults.Layer), 'stable'));

%% ------------------------- MAKE MANUSCRIPT FIGURES ---------------------
make_network_manuscript_figures(combinedResults, cfg);

fprintf('\n============================================================\n');
fprintf('PLOT-ONLY SCRIPT COMPLETED\n');
fprintf('============================================================\n\n');

fprintf('Figures written to: %s\n\n', cfg.figDir);

%% ========================= LOCAL FUNCTIONS =============================

function T = load_saved_nn_results(cfg)
% Load existing combined results, or reconstruct them from scenario CSVs.

    if isfile(cfg.combinedCsv)
        fprintf('Loading combined results:\n  %s\n', cfg.combinedCsv);
        T = readtable(cfg.combinedCsv, 'TextType','string');
        return;
    end

    fprintf('Combined results file not found:\n  %s\n', cfg.combinedCsv);
    fprintf('Trying to reconstruct combined table from scenario result CSVs...\n\n');

    missing = {};
    if ~isfile(cfg.cleanCsv)
        missing{end+1} = cfg.cleanCsv; %#ok<AGROW>
    end
    if ~isfile(cfg.pooledCsv)
        missing{end+1} = cfg.pooledCsv; %#ok<AGROW>
    end

    if ~isempty(missing)
        fprintf('Missing required result files:\n');
        for i = 1:numel(missing)
            fprintf('  %s\n', missing{i});
        end
        error(['No saved network results found. Do not set cfg.runAnalysis=true unless ', ...
               'you really want to recompute ResNet activations. First locate or regenerate ', ...
               'the result CSV files.']);
    end

    Tclean  = readtable(cfg.cleanCsv,  'TextType','string');
    Tpooled = readtable(cfg.pooledCsv, 'TextType','string');

    T = [Tclean; Tpooled];

    fprintf('Writing reconstructed combined results:\n  %s\n', cfg.combinedCsv);
    writetable(T, cfg.combinedCsv);
end

function validate_results_table(T)
% Check that the saved results contain the columns needed for plotting.

    requiredVars = [ ...
        "Scenario", ...
        "ScenarioDisplay", ...
        "Layer", ...
        "LayerIndex", ...
        "K", ...
        "NMI_Ysub", ...
        "NMI_Ybasic", ...
        "NMI_A", ...
        "NMI_V" ...
    ];

    varNames = string(T.Properties.VariableNames);
    missing = requiredVars(~ismember(requiredVars, varNames));

    if ~isempty(missing)
        fprintf('Available table variables:\n');
        disp(varNames');
        error('Missing required variables in result table: %s', strjoin(missing, ', '));
    end
end

function make_network_manuscript_figures(Tout, cfg)
% Create the network figures used in the manuscript.

    Tout.Scenario = string(Tout.Scenario);
    Tout.ScenarioDisplay = string(Tout.ScenarioDisplay);
    Tout.Layer = string(Tout.Layer);

    if ~any(Tout.K == cfg.kForPlot)
        error('Requested cfg.kForPlot=%d, but available K values are: %s', ...
            cfg.kForPlot, strjoin(string(unique(Tout.K)'), ', '));
    end

    Tk = Tout(Tout.K == cfg.kForPlot,:);
    Tk = sortrows(Tk, {'Scenario','LayerIndex'});

    make_figure07_network_summary(Tk, cfg);
    make_figure08(Tk, cfg);
end

function make_figure07_network_summary(Tk, cfg)
% Figure 7: network extension summary.
% A: clean-only layer-wise target preservation, k = 10.
% B: pooled-nuisance layer-wise target preservation, k = 10.
% C: final-layer comparison at avg_pool.

    scenariosWanted = ["clean_only", "pooled_variants"];
    panelLetters = {'A','B'};
    panelTitles = {'Clean-only object run', 'Pooled nuisance run'};

    targetVars = ["NMI_Ysub", "NMI_Ybasic", "NMI_A", "NMI_V"];
    targetLabels = get_target_labels();
    grayCols = get_gray_cols();

    lineStyles = {'-', '-', '-', '--'};
    markers    = {'o', 's', '^', 'd'};

    fig7 = figure('Color','w', 'Position',[100 100 1200 370]);

    t7 = tiledlayout(fig7, 1, 3, ...
        'TileSpacing','normal', ...
        'Padding','normal');

    axList = gobjects(1,3);

    % ---------------------------------------------------------------------
    % A-B. Layer-wise target preservation
    % ---------------------------------------------------------------------
    for si = 1:numel(scenariosWanted)

        scenarioName = scenariosWanted(si);
        Ts = Tk(Tk.Scenario == scenarioName,:);
        Ts = sortrows(Ts, 'LayerIndex');

        if isempty(Ts)
            error('Scenario "%s" not found in result table.', scenarioName);
        end

        ax = nexttile(t7);
        axList(si) = ax;
        hold(ax, 'on');

        x = 1:height(Ts);

        for tv = 1:numel(targetVars)
            plot(ax, x, Ts.(targetVars(tv)), ...
                'LineStyle', lineStyles{tv}, ...
                'Marker', markers{tv}, ...
                'Color', grayCols(tv,:), ...
                'LineWidth', 1, ...
                'MarkerSize', 4, ...
                'MarkerFaceColor', grayCols(tv,:), ...
                'MarkerEdgeColor', grayCols(tv,:));
        end

        set(ax, ...
            'XTick', x, ...
            'XTickLabel', clean_layer_labels(Ts.Layer), ...
            'XTickLabelRotation', 35, ...
            'TickLabelInterpreter','latex', ...
            'Box','off', ...
            'TickDir','out', ...
            'TickLength',[.01 .01], ...
            'LineWidth',0.5, ...
            'FontName','Times New Roman', ...
            'FontSize',10);

        xlabel(ax, 'Network layer', 'Interpreter','latex');
        ylabel(ax, 'Mutual information [normalised]', 'Interpreter','latex');

        ylim(ax, [0 1.05]);
        xlim(ax, [0.75 height(Ts)+0.25]);
        axis(ax, 'square');

        text(ax, 0.02, 1.05, panelTitles{si}, ...
            'Units','normalized', ...
            'HorizontalAlignment','left', ...
            'VerticalAlignment','top', ...
            'FontName','Times New Roman', ...
            'FontSize',10, ...
            'Interpreter','none');

        labelSubplots(ax, panelLetters{si}, [-0.1 1], false, ...
            'FontSize', 14, ...
            'FontName', 'Times New Roman', ...
            'FontWeight', 'normal');
    end

    % Manual line legend attached to panel B. Applies to A and B.
    drawManualLineLegend(axList(2), grayCols, lineStyles, markers, targetLabels, ...
        .3, 0.68, ...   % xAnchor, yAnchor in axes-relative coordinates
        0.13, ...       % line length
        0.090, ...      % vertical spacing
        9);             % font size

    % ---------------------------------------------------------------------
    % C. Final-layer comparison
    % ---------------------------------------------------------------------
    finalLayer = string(cfg.finalLayer);
    Tf = Tk(Tk.Layer == finalLayer,:);

    if isempty(Tf)
        fprintf('Available layers:\n');
        disp(unique(Tk.Layer, 'stable'));
        error('Final layer "%s" not found.', finalLayer);
    end

    scenarioOrder = ["clean_only", "pooled_variants"];
    rowIdx = zeros(numel(scenarioOrder),1);

    for i = 1:numel(scenarioOrder)
        idx = find(Tf.Scenario == scenarioOrder(i), 1);

        if isempty(idx)
            error('Scenario "%s" not found for final layer "%s".', ...
                scenarioOrder(i), finalLayer);
        end

        rowIdx(i) = idx;
    end

    Tf = Tf(rowIdx,:);
    M = [Tf.NMI_Ysub, Tf.NMI_Ybasic, Tf.NMI_A, Tf.NMI_V];
    xLabels = {'Clean-only', 'Pooled nuisance'};

    axC = nexttile(t7, 3);
    axList(3) = axC; %#ok<NASGU>
    hold(axC, 'on');

    hb = bar(axC, M, 'BarWidth', 0.72);

    for b = 1:numel(hb)
        hb(b).FaceColor = grayCols(b,:);
        hb(b).EdgeColor = 'none';
    end

    set(axC, ...
        'XTick', 1:numel(xLabels), ...
        'XTickLabel', xLabels, ...
        'TickLabelInterpreter','none', ...
        'Box','off', ...
        'TickDir','out', ...
        'TickLength',[.01 .01], ...
        'LineWidth',0.5, ...
        'FontName','Times New Roman', ...
        'FontSize',10);

    ylabel(axC, 'Mutual information [normalised]', 'Interpreter','latex');
    ylim(axC, [0, max(M(:))*1.22]);
    axis(axC, 'square');

    text(axC, 0.02, 1.05, 'Final-layer comparison', ...
        'Units','normalized', ...
        'HorizontalAlignment','left', ...
        'VerticalAlignment','top', ...
        'FontName','Times New Roman', ...
        'FontSize',10, ...
        'Interpreter','none');

    drawManualBarLegend(axC, grayCols, targetLabels, ...
        0.15, -.18, ...   % xAnchor, yAnchor in axes-relative coordinates
        0.12, 0.020, ...  % swatch width, swatch height
        0.075, ...        % vertical spacing
        9);               % font size

    labelSubplots(axC, 'C', [-0.1 1], false, ...
        'FontSize', 14, ...
        'FontName', 'Times New Roman', ...
        'FontWeight', 'normal');

    save_figure(fig7, cfg.figDir, 'figure07', cfg);
end

function make_figure08(Tk, cfg)
% Figure 8 dispatcher.
%
% In auto mode, this uses the true cluster-target heatmaps if the optional
% per-image files are present. If they are absent, it creates a fallback
% final-layer target-preservation heatmap from summary NMI values.

    mode = string(cfg.figure08Mode);

    haveClusterFiles = isfile(cfg.cleanClusterTargetCsv) && ...
                       isfile(cfg.pooledClusterTargetCsv);

    if mode == "cluster_target"
        if ~haveClusterFiles
            fprintf('\nMissing cluster-target files required for Figure 8:\n');
            fprintf('  %s\n', cfg.cleanClusterTargetCsv);
            fprintf('  %s\n', cfg.pooledClusterTargetCsv);
            error(['Figure 8 in cluster_target mode requires per-image cluster assignments. ', ...
                   'Create these CSV files during the neural-network analysis.']);
        end
        make_figure08_cluster_target_heatmaps(cfg);
        return;
    end

    if mode == "summary"
        make_figure08_summary_heatmap(Tk, cfg);
        return;
    end

    if mode ~= "auto"
        error('Unknown cfg.figure08Mode: %s', mode);
    end

    if haveClusterFiles
        fprintf('\nFigure 8: using per-image cluster-target files.\n');
        make_figure08_cluster_target_heatmaps(cfg);
    else
        fprintf('\nFigure 8: per-image cluster-target files not found.\n');
        fprintf('Using summary final-layer target-preservation heatmap instead.\n');
        make_figure08_summary_heatmap(Tk, cfg);
    end
end

function make_figure08_cluster_target_heatmaps(cfg)
% Figure 8 true cluster-target heatmaps.
%
% Rows are k-means clusters C_{l,10}. Columns are target values. Cell values
% are P(target value | cluster).

    Tclean  = readtable(cfg.cleanClusterTargetCsv,  'TextType','string');
    Tpooled = readtable(cfg.pooledClusterTargetCsv, 'TextType','string');

    requiredVars = {'cluster','Yobj','Ysup','A','V'};

    for r = 1:numel(requiredVars)
        if ~ismember(requiredVars{r}, Tclean.Properties.VariableNames)
            error('Clean-only cluster-target table is missing required column: %s', requiredVars{r});
        end

        if ~ismember(requiredVars{r}, Tpooled.Properties.VariableNames)
            error('Pooled-nuisance cluster-target table is missing required column: %s', requiredVars{r});
        end
    end

    scenarioTables = {Tclean, Tpooled};
    scenarioNames  = {'Clean-only', 'Pooled nuisance'};

    targetVars = {'Yobj', 'Ysup', 'A', 'V'};
    targetLabs = get_target_labels();

    fig8 = figure('Color','w', ...
        'Units','centimeters', ...
        'Position',[4 4 25 12]);

    t8 = tiledlayout(fig8, 2, 4, ...
        'TileSpacing','compact', ...
        'Padding','compact');

    for s = 1:2

        T = scenarioTables{s};

        for q = 1:4

            ax = nexttile(t8);
            hold(ax, 'on');

            [M, rowLabs, colLabs] = clusterTargetMatrix_local( ...
                T.cluster, T.(targetVars{q}));

            imagesc(ax, M);
            caxis(ax, [0 1]);
            colormap(ax, parula);

            set(ax, ...
                'XTick', 1:numel(colLabs), ...
                'XTickLabel', colLabs, ...
                'YTick', 1:numel(rowLabs), ...
                'YTickLabel', rowLabs, ...
                'TickLength', [0 0], ...
                'TickDir', 'out', ...
                'Box', 'off', ...
                'LineWidth', 0.5, ...
                'FontName', 'Times New Roman', ...
                'FontSize', 8, ...
                'TickLabelInterpreter', 'none');

            xtickangle(ax, 45);

            if q == 1
                ylabel(ax, sprintf('%s\nCluster $C_{l,10}$', scenarioNames{s}), ...
                    'Interpreter','latex', ...
                    'FontName','Times New Roman', ...
                    'FontSize',9);
            else
                ylabel(ax, '');
            end

            title(ax, targetLabs{q}, ...
                'Interpreter','none', ...
                'FontName','Times New Roman', ...
                'FontSize',9, ...
                'FontWeight','normal');

            write_heatmap_values(ax, M, 7);
        end
    end

    cb = colorbar;
    cb.Layout.Tile = 'east';
    cb.Box = 'off';
    cb.TickDirection = 'out';
    cb.LineWidth = 0.5;
    cb.FontName = 'Times New Roman';
    cb.FontSize = 9;
    cb.Label.String = 'P(target value | cluster)';
    cb.Label.Interpreter = 'none';

    save_figure(fig8, cfg.figDir, 'figure08', cfg);
end

function make_figure08_summary_heatmap(Tk, cfg)
% Figure 8 fallback summary heatmap.
%
% This does not show cluster composition. It shows final-layer normalised
% mutual information values by scenario and target, using the summary table
% already needed for Figure 7.

    finalLayer = string(cfg.finalLayer);
    Tf = Tk(Tk.Layer == finalLayer,:);

    if isempty(Tf)
        fprintf('Available layers:\n');
        disp(unique(Tk.Layer, 'stable'));
        error('Final layer "%s" not found.', finalLayer);
    end

    scenarioOrder = ["clean_only", "pooled_variants"];
    scenarioLabels = {'Clean-only', 'Pooled nuisance'};
    rowIdx = zeros(numel(scenarioOrder),1);

    for i = 1:numel(scenarioOrder)
        idx = find(Tf.Scenario == scenarioOrder(i), 1);
        if isempty(idx)
            error('Scenario "%s" not found for final layer "%s".', ...
                scenarioOrder(i), finalLayer);
        end
        rowIdx(i) = idx;
    end

    Tf = Tf(rowIdx,:);
    M = [Tf.NMI_Ysub, Tf.NMI_Ybasic, Tf.NMI_A, Tf.NMI_V];
    targetLabels = get_target_labels();

    fig8 = figure('Color','w', ...
        'Units','centimeters', ...
        'Position',[6 6 15 8]);

    ax = axes(fig8);
    hold(ax, 'on');

    imagesc(ax, M);
    caxis(ax, [0 1]);
    colormap(ax, parula);

    set(ax, ...
        'XTick', 1:numel(targetLabels), ...
        'XTickLabel', targetLabels, ...
        'YTick', 1:numel(scenarioLabels), ...
        'YTickLabel', scenarioLabels, ...
        'TickLength', [0 0], ...
        'TickDir', 'out', ...
        'Box', 'off', ...
        'LineWidth', 0.5, ...
        'FontName', 'Times New Roman', ...
        'FontSize', 9, ...
        'TickLabelInterpreter', 'none');

    xtickangle(ax, 25);

    xlabel(ax, 'Target variable', 'Interpreter','none');
    ylabel(ax, 'Scenario', 'Interpreter','none');

    title(ax, 'Final-layer target preservation at avg\_pool', ...
        'Interpreter','none', ...
        'FontName','Times New Roman', ...
        'FontSize',10, ...
        'FontWeight','normal');

    write_heatmap_values(ax, M, 9);

    cb = colorbar(ax);
    cb.Box = 'off';
    cb.TickDirection = 'out';
    cb.LineWidth = 0.5;
    cb.FontName = 'Times New Roman';
    cb.FontSize = 9;
    cb.Label.String = 'Mutual information [normalised]';
    cb.Label.Interpreter = 'none';

    save_figure(fig8, cfg.figDir, 'figure08', cfg);
end

function targetLabels = get_target_labels()
    targetLabels = { ...
        'Object class', ...
        ['Superordinate animal' char(8211) 'vehicle'], ...
        'Environment / affordance-like', ...
        'Nuisance condition'};
end

function grayCols = get_gray_cols()
    grayCols = [ ...
        0.10 0.10 0.10; ...
        0.35 0.35 0.35; ...
        0.60 0.60 0.60; ...
        0.82 0.82 0.82];
end

function labels = clean_layer_labels(layerStrings)
% Clean layer labels for LaTeX-safe tick labels.

    labels = cellstr(string(layerStrings));
    for i = 1:numel(labels)
        labels{i} = strrep(labels{i}, '_', '\_');
    end
end

function drawManualLineLegend(ax, cols, lineStyles, markers, labels, ...
    xAnchor, yAnchor, lineW, dy, fs)
% Draw compact manual legend for line plots.
%
% Coordinates xAnchor/yAnchor are in axes-relative coordinates.
% Annotation objects are attached to the ancestor figure.

    fig = ancestor(ax, 'figure');

    if isempty(fig) || ~isvalid(fig)
        error('drawManualLineLegend:InvalidFigure', ...
            'Could not find a valid ancestor figure for the supplied axes.');
    end

    oldUnits = ax.Units;
    ax.Units = 'normalized';
    axPos = ax.Position;
    ax.Units = oldUnits;

    for i = 1:numel(labels)

        x0 = axPos(1) + xAnchor * axPos(3);
        y0 = axPos(2) + (yAnchor - (i-1)*dy) * axPos(4);
        x1 = x0 + lineW * axPos(3);

        annotation(fig, 'line', ...
            [x0 x1], [y0 y0], ...
            'Color', cols(i,:), ...
            'LineStyle', lineStyles{i}, ...
            'LineWidth', 1);

        annotation(fig, 'textbox', ...
            [x1 + 0.035*axPos(3), ...
             y0 - 0.025*axPos(4), ...
             0.80*axPos(3), ...
             0.055*axPos(4)], ...
            'String', labels{i}, ...
            'Interpreter', 'none', ...
            'FontName', 'Times New Roman', ...
            'FontSize', fs, ...
            'LineStyle', 'none', ...
            'VerticalAlignment', 'middle', ...
            'HorizontalAlignment', 'left');

        % Marker style is visible in the data. The manual legend uses line
        % style and grayscale coding because annotation markers export less
        % robustly across MATLAB backends.
        %#ok<NASGU>
        markers;
    end
end

function drawManualBarLegend(ax, cols, labels, ...
    xAnchor, yAnchor, swatchW, swatchH, dy, fs)
% Draw compact manual legend for bar plots.
%
% Coordinates xAnchor/yAnchor are in axes-relative coordinates.
% Annotation objects are attached to the ancestor figure.

    fig = ancestor(ax, 'figure');

    if isempty(fig) || ~isvalid(fig)
        error('drawManualBarLegend:InvalidFigure', ...
            'Could not find a valid ancestor figure for the supplied axes.');
    end

    oldUnits = ax.Units;
    ax.Units = 'normalized';
    axPos = ax.Position;
    ax.Units = oldUnits;

    for i = 1:numel(labels)

        x0 = axPos(1) + xAnchor * axPos(3);
        y0 = axPos(2) + (yAnchor - (i-1)*dy) * axPos(4);

        annotation(fig, 'rectangle', ...
            [x0, y0, swatchW*axPos(3), swatchH*axPos(4)], ...
            'FaceColor', cols(i,:), ...
            'Color', 'none');

        annotation(fig, 'textbox', ...
            [x0 + (swatchW + 0.035)*axPos(3), ...
             y0 - 0.020*axPos(4), ...
             0.90*axPos(3), ...
             0.055*axPos(4)], ...
            'String', labels{i}, ...
            'Interpreter', 'none', ...
            'FontName', 'Times New Roman', ...
            'FontSize', fs, ...
            'LineStyle', 'none', ...
            'VerticalAlignment', 'middle', ...
            'HorizontalAlignment', 'left');
    end
end

function labelSubplots(ax, labelText, xy, useDataUnits, varargin)
% Simple subplot label helper.
%
% xy is either normalized axes coordinates or data coordinates.

    if nargin < 4
        useDataUnits = false;
    end

    if useDataUnits
        text(ax, xy(1), xy(2), labelText, ...
            'Units','data', ...
            'HorizontalAlignment','left', ...
            'VerticalAlignment','bottom', ...
            varargin{:});
    else
        text(ax, xy(1), xy(2), labelText, ...
            'Units','normalized', ...
            'HorizontalAlignment','left', ...
            'VerticalAlignment','bottom', ...
            varargin{:});
    end
end

function write_heatmap_values(ax, M, fs)
% Add numerical values to a heatmap.

    for ii = 1:size(M,1)
        for jj = 1:size(M,2)

            val = M(ii,jj);

            if val >= 0.5
                txtCol = 'w';
            else
                txtCol = 'k';
            end

            text(ax, jj, ii, sprintf('%.2f', val), ...
                'HorizontalAlignment','center', ...
                'VerticalAlignment','middle', ...
                'FontName','Times New Roman', ...
                'FontSize',fs, ...
                'Color', txtCol, ...
                'Interpreter','none');
        end
    end
end

function [M, rowLabs, colLabs] = clusterTargetMatrix_local(C, Y)
% Conditional target composition by cluster: P(target value | cluster).

    C = categorical(C);
    Y = categorical(Y);

    rowCats = categories(C);
    colCats = categories(Y);

    rowID = double(C);
    colID = double(Y);

    counts = accumarray([rowID, colID], 1, ...
        [numel(rowCats), numel(colCats)], @sum, 0);

    rowSums = sum(counts, 2);
    rowSums(rowSums == 0) = 1;

    M = counts ./ rowSums;

    rowLabs = rowCats;
    colLabs = colCats;
end

function ensure_dir(d)
    if ~isfolder(d)
        mkdir(d);
    end
end

function save_figure(figHandle, figDir, baseName, cfg)
% Export PNG/PDF without axes toolbar artifacts.

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
