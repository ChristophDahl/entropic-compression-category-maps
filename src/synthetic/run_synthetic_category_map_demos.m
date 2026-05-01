%% run_synthetic_category_map_demos.m
% Entropic compression and category usefulness.
%
% Folder layout expected:
%   I:\entropicCompression\
%       manuscript\
%       programs\
%       figures\
%       results\
%       data\
%
% Recommended location:
%   I:\entropicCompression\programs\main_entropic_compression.m
%
% Demonstration 1: Compression alone is insufficient.
% Demonstration 2: Category usefulness depends on the preserved target.
% Demonstration 3: Category objectivity can be operationalised as robustness
%                  of preserved information under perturbation.
% Demonstration 4: Category level is target-dependent / category-level independent framework.
%
% Outputs:
%   data\synthetic_category_world.mat
%   results\demo01_metrics.mat / .csv
%   results\demo02_target_dependence.mat / .csv
%   results\demo03_robustness.mat / .csv
%   results\demo04_hierarchical_levels.mat / .csv
%   figures\figure01-figure11 as PNG and PDF
%
% Author: Christoph D. Dahl

clear; close all; clc;
rng(7);

%% --------------------------- PROJECT PATHS ------------------------------
thisFile = mfilename('fullpath');
if isempty(thisFile)
    scriptDir = pwd;
else
    scriptDir = fileparts(thisFile);
end

[~, currentFolderName] = fileparts(scriptDir);
if strcmpi(currentFolderName, 'programs')
    projectRoot = fileparts(scriptDir);
else
    projectRoot = scriptDir;
end

P.projectRoot   = projectRoot;
P.manuscriptDir = fullfile(projectRoot, 'manuscript');
P.programsDir   = fullfile(projectRoot, 'programs');
P.figDir        = fullfile(projectRoot, 'figures');
P.resultsDir    = fullfile(projectRoot, 'results');
P.dataDir       = fullfile(projectRoot, 'data');

ensure_dir(P.manuscriptDir);
ensure_dir(P.programsDir);
ensure_dir(P.figDir);
ensure_dir(P.resultsDir);
ensure_dir(P.dataDir);

fprintf('\nProject root: %s\n', P.projectRoot);
fprintf('Figures     : %s\n', P.figDir);
fprintf('Results     : %s\n', P.resultsDir);
fprintf('Data        : %s\n\n', P.dataDir);

%% ----------------------------- SETTINGS --------------------------------
P.N        = 3000;
P.K        = 3;
P.noiseStd = 0.55;
P.viewGain = [0.85, -0.25];
P.nGrid    = 14;
P.lambda   = 0.20;
P.savePNG  = true;
P.savePDF  = true;
P.figRes   = 300;
P.saveSeparateRobustness = false;  % set true to also save separate robustness panels

%% ---------------------- SYNTHETIC STIMULUS WORLD ------------------------
% Y is the latent identity/kind of the stimulus.
Y = sample_categorical([0.34 0.33 0.33], P.N);

% A is the action/affordance target. Here, identities 1 and 3 imply the
% same action, whereas identity 2 implies a different action.
A = ones(P.N,1);
A(Y == 2) = 2;

% Three latent means in a 2-D sensory or feature space.
mu = [-2.2,  0.0;
       2.2,  0.0;
       0.0,  2.8];

% V is a nuisance/context variable, e.g., viewpoint or illumination.
V = 2*rand(P.N,1) - 1;

% Observed sensory state S.
S = mu(Y,:) + V.*P.viewGain + P.noiseStd*randn(P.N,2);
Sgrid = discretize_2d_state(S, P.nGrid);
H_Sgrid = entropy_discrete(Sgrid);

% Extra targets for Demonstration 2.
Vbin = discretize_quantile(V, P.K);             % context/viewpoint target
Gtarget = simple_kmeans(S, P.K, 80);            % sensory-geometric target

save(fullfile(P.dataDir, 'synthetic_category_world.mat'), ...
    'S','Sgrid','Y','A','V','Vbin','Gtarget','mu','P');

%% -------------------------- CATEGORY MAPS -------------------------------
partitions = struct([]);

partitions(end+1).name = 'Identity-preserving';
partitions(end).shortName = 'Identity';
partitions(end).C = Y;

partitions(end+1).name = 'Action-preserving';
partitions(end).shortName = 'Action';
partitions(end).C = A;

partitions(end+1).name = 'Geometric clustering';
partitions(end).shortName = 'Geometric';
partitions(end).C = simple_kmeans(S, P.K, 60);

partitions(end+1).name = 'Nuisance-based';
partitions(end).shortName = 'Nuisance';
partitions(end).C = Vbin;

partitions(end+1).name = 'Random partition';
partitions(end).shortName = 'Random';
partitions(end).C = randi(P.K, P.N, 1);

partitions(end+1).name = 'One-category compression';
partitions(end).shortName = 'One category';
partitions(end).C = ones(P.N,1);

for i = 1:numel(partitions)
    partitions(i).C = relabel_consecutive(partitions(i).C);
end

partitionLabels = string({partitions.shortName})';
partitionFullLabels = string({partitions.name})';

%% ======================= DEMONSTRATION 1 ================================
% Compression alone is insufficient: compare H(C), I(C;Y), I(C;A), and
% remaining target uncertainty.

nP = numel(partitions);
metrics1 = table('Size', [nP 12], ...
    'VariableTypes', {'string','double','double','double','double','double','double','double','double','double','double','double'}, ...
    'VariableNames', {'Partition','nCategories','H_C','H_Sgrid','I_C_Sgrid','I_C_Y','I_C_A', ...
                      'H_Y_given_C','H_A_given_C','NMI_C_Y','Q_Y','Q_A'});

for i = 1:nP
    C = partitions(i).C;
    H_C = entropy_discrete(C);
    I_C_Sgrid = mutual_information_discrete(C, Sgrid);
    I_C_Y = mutual_information_discrete(C, Y);
    I_C_A = mutual_information_discrete(C, A);

    H_Y = entropy_discrete(Y);
    H_A = entropy_discrete(A);
    H_Y_given_C = H_Y - I_C_Y;
    H_A_given_C = H_A - I_C_A;

    metrics1.Partition(i)    = partitionLabels(i);
    metrics1.nCategories(i)  = numel(unique(C));
    metrics1.H_C(i)          = H_C;
    metrics1.H_Sgrid(i)      = H_Sgrid;
    metrics1.I_C_Sgrid(i)    = I_C_Sgrid;
    metrics1.I_C_Y(i)        = I_C_Y;
    metrics1.I_C_A(i)        = I_C_A;
    metrics1.H_Y_given_C(i)  = H_Y_given_C;
    metrics1.H_A_given_C(i)  = H_A_given_C;
    metrics1.NMI_C_Y(i)      = I_C_Y / max(eps, H_Y);
    metrics1.Q_Y(i)          = I_C_Y - P.lambda * H_C;
    metrics1.Q_A(i)          = I_C_A - P.lambda * H_C;
end

fprintf('\nDEMONSTRATION 1\n');
fprintf('Reference entropy of discretized stimulus space H(S_grid): %.3f bits\n\n', H_Sgrid);
disp(metrics1);

save(fullfile(P.resultsDir, 'demo01_metrics.mat'), 'metrics1','partitions','H_Sgrid','P');
writetable(metrics1, fullfile(P.resultsDir, 'demo01_metrics.csv'));

%% Demo 1 figures
fig1 = figure('Color','w','Position',[80 80 900 600]);
t1 = tiledlayout(2, 3, 'TileSpacing', 'normal', 'Padding', 'normal');

axList = gobjects(nP,1);

for i = 1:nP
    axList(i) = nexttile;

    scatter(S(:,1), S(:,2), 8, partitions(i).C, ...
        'filled', ...
        'MarkerFaceAlpha', 0.45);

    axis([-4.5 4.5 -2 4.5]);
    axis square;

    xlabel('$x_1$', 'Interpreter','latex');
    ylabel('$x_2$', 'Interpreter','latex');

    title(partitions(i).name, ...
        'Interpreter','none', ...
        'FontSize',10, ...
        'FontWeight','normal');

    set(gca, ...
        'Box','off', ...
        'TickDir','out', ...
        'TickLength',[.01 .01], ...
        'LineWidth',0.5, ...
        'FontName','Times New Roman', ...
        'FontSize',10, ...
        'TickLabelInterpreter','latex');
end

% Add subplot labels
panelLabels = {'A','B','C','D','E','F'};
labelSubplots(axList, panelLabels, [0.1 0.03], false, ...
    'FontSize', 14, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal');

save_figure(fig1, P.figDir, 'figure01', P);

% sgtitle('Demonstration 1: same stimulus space, different candidate category maps');
% save_figure(fig1, P.figDir, 'figure01_demo01_candidate_partitions', P);

fig2 = figure('Color','w','Position',[80 80 650 650]);
t2 = tiledlayout(2, 2, 'TileSpacing', 'normal', 'Padding', 'normal');

%% Panel G: category complexity and preserved target information
ax = nexttile;
hold(ax, 'on');

M = [metrics1.H_C, metrics1.I_C_Y, metrics1.I_C_A];

hb = bar(ax, M, 'BarWidth', 0.62);

grayCols_G = [ ...
    0.25 0.25 0.25; ...
    0.55 0.55 0.55; ...
    0.78 0.78 0.78];

for b = 1:numel(hb)
    hb(b).FaceColor = grayCols_G(b,:);
    hb(b).EdgeColor = 'none';
end

set(ax, ...
    'XTick', 1:nP, ...
    'XTickLabel', partitionLabels, ...
    'XTickLabelRotation', 35, ...
    'TickLabelInterpreter', 'none', ...
    'Box', 'off', ...
    'TickDir', 'out', ...
    'TickLength', [.01 .01], ...
    'LineWidth', 0.5, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 10);

ylabel(ax, 'Bits', 'Interpreter', 'latex');
ylim(ax, [0, max(M(:))*1.18]);
axis(ax, 'square');

% Mark the all-zero one-category case
zeroGroup = find(strcmpi(string(partitionLabels), "one category"));

if ~isempty(zeroGroup)
    yZero = 0.10;

    plot(ax, [zeroGroup-0.28, zeroGroup+0.28], [yZero yZero], ...
        'k-', 'LineWidth', 0.6, 'HandleVisibility', 'off');

    plot(ax, [zeroGroup-0.28, zeroGroup-0.28], ...
        [yZero-0.018 yZero+0.018], ...
        'k-', 'LineWidth', 0.6, 'HandleVisibility', 'off');

    plot(ax, [zeroGroup+0.28, zeroGroup+0.28], ...
        [yZero-0.018 yZero+0.018], ...
        'k-', 'LineWidth', 0.6, 'HandleVisibility', 'off');

    text(ax, zeroGroup, yZero+0.05, 'all zero', ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'Interpreter', 'latex', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 10);
end

% Manual legend with matched thin rectangle swatches
drawManualBarLegend(ax, grayCols_G, ...
    {'$H(C)$', '$I(C;Y)$', '$I(C;A)$'}, ...
    0.08, 1.10, ...   % xAnchor, yAnchor in axes-relative coordinates
    0.18, 0.020, ...  % swatch width, swatch height
    0.080, ...        % vertical spacing
    10);               % font size

labelSubplots(ax, 'A', [0.1 0.03], false, ...
    'FontSize', 14, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal');


%% Panel H: quality scores
ax3 = nexttile;
hold(ax3, 'on');

Mqual = [metrics1.Q_Y, metrics1.Q_A];

hb3 = bar(ax3, Mqual, 'BarWidth', 0.72);

grayCols_H = [ ...
    0.35 0.35 0.35; ...
    0.72 0.72 0.72];

for b = 1:numel(hb3)
    hb3(b).FaceColor = grayCols_H(b,:);
    hb3(b).EdgeColor = 'none';
end

yline(ax3, 0, ':k', ...
    'LineWidth', 0.8, ...
    'HandleVisibility', 'off');

set(ax3, ...
    'XTick', 1:nP, ...
    'XTickLabel', partitionLabels, ...
    'XTickLabelRotation', 35, ...
    'TickLabelInterpreter', 'none', ...
    'Box', 'off', ...
    'TickDir', 'out', ...
    'TickLength', [.01 .01], ...
    'LineWidth', 0.5, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 10);

ylabel(ax3, 'Quality score', 'Interpreter', 'latex');
axis(ax3, 'square');

% Extra headroom so the legend sits inside the axes range
yLow  = min([Mqual(:); 0]) - 0.10;
yHigh = max(Mqual(:)) * 1.55;
ylim(ax3, [yLow yHigh]);

% Manual legend with the same swatch thickness logic
drawManualBarLegend(ax3, grayCols_H, ...
    {'$Q_Y = I(C;Y) - \lambda H(C)$', ...
     '$Q_A = I(C;A) - \lambda H(C)$'}, ...
    0.10, 1.05, ...   % xAnchor, yAnchor in axes-relative coordinates
    0.13, 0.020, ...  % swatch width, swatch height
    0.090, ...        % vertical spacing
    10);              % font size

labelSubplots(ax3, 'B', [0.1 0.03], false, ...
    'FontSize', 14, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal');

%% Figure 4: complexity--relevance trade-off
ax4 = nexttile;
hold(ax4, 'on');

colY = [0.30 0.30 0.30];
colA = [0.65 0.65 0.65];

% Marker-only plotted data
plot(ax4, metrics1.H_C, metrics1.I_C_Y, 'o', ...
    'LineStyle','none', ...
    'MarkerSize',7, ...
    'LineWidth',1.0, ...
    'Color',colY, ...
    'MarkerFaceColor','none', ...
    'MarkerEdgeColor',colY);

plot(ax4, metrics1.H_C, metrics1.I_C_A, 's', ...
    'LineStyle','none', ...
    'MarkerSize',7, ...
    'LineWidth',1.0, ...
    'Color',colA, ...
    'MarkerFaceColor','none', ...
    'MarkerEdgeColor',colA);

% Manual labels for candidate maps
labelXY = [1.68 1.57;   % identity
           0.94 0.86;   % action
           1.36 1.47;   % geometric
           1.66 0.10;   % nuisance
           1.66 -0.01;  % random
           0.05 0.07];  % one category

for i = 1:nP
    text(ax4, labelXY(i,1), labelXY(i,2), partitionLabels{i}, ...
        'FontSize',9, ...
        'FontName','Times New Roman', ...
        'HorizontalAlignment','left', ...
        'Interpreter','none');
end

xlabel(ax4, '$H(C)$ [bits]', 'Interpreter','latex');
ylabel(ax4, 'Preserved information [bits]', 'Interpreter','latex');

xlim(ax4, [-0.02 1.88]);

% Add headroom for the manual legend
ylim(ax4, [-0.08, max(metrics1.I_C_Y) + 0.38]);

set(ax4, ...
    'Box','off', ...
    'TickDir','out', ...
    'TickLength',[.01 .01], ...
    'LineWidth',0.5, ...
    'FontName','Times New Roman', ...
    'FontSize',10, ...
    'TickLabelInterpreter','latex');

axis(ax4, 'square');

%% Manual marker-only legend as axes objects
xl = xlim(ax4);
yl = ylim(ax4);
xRange = diff(xl);
yRange = diff(yl);

% Legend position in data coordinates
xLeg = xl(1) + 0.1*xRange;     % increase = move right
yLeg = yl(2) + 0.05*yRange;     % decrease subtraction = move up

dy = 0.09*yRange;
xText = xLeg + 0.10*xRange;

plot(ax4, xLeg, yLeg, 'o', ...
    'LineStyle','none', ...
    'MarkerSize',7, ...
    'LineWidth',1.0, ...
    'MarkerFaceColor','none', ...
    'MarkerEdgeColor',colY, ...
    'Color',colY, ...
    'Clipping','off', ...
    'HandleVisibility','off');

text(ax4, xText, yLeg, 'Identity target: $I(C;Y)$', ...
    'Interpreter','latex', ...
    'FontName','Times New Roman', ...
    'FontSize',10, ...
    'HorizontalAlignment','left', ...
    'VerticalAlignment','middle', ...
    'Clipping','off');

plot(ax4, xLeg, yLeg-dy, 's', ...
    'LineStyle','none', ...
    'MarkerSize',7, ...
    'LineWidth',1.0, ...
    'MarkerFaceColor','none', ...
    'MarkerEdgeColor',colA, ...
    'Color',colA, ...
    'Clipping','off', ...
    'HandleVisibility','off');

text(ax4, xText, yLeg-dy, 'Action target: $I(C;A)$', ...
    'Interpreter','latex', ...
    'FontName','Times New Roman', ...
    'FontSize',10, ...
    'HorizontalAlignment','left', ...
    'VerticalAlignment','middle', ...
    'Clipping','off');

labelSubplots(ax4, 'C', [0.1 0.03], false, ...
    'FontSize',14, ...
    'FontName','Times New Roman', ...
    'FontWeight','normal');


%% Figure 5: remaining uncertainty after categorisation

ax5 = nexttile;
hold(ax5, 'on');

M2 = [metrics1.H_Y_given_C, metrics1.H_A_given_C];

hb = bar(ax5, M2, 'BarWidth', 0.72);

% Grayscale bar colours
grayCols = [ ...
    0.35 0.35 0.35; ...
    0.72 0.72 0.72];

for b = 1:numel(hb)
    hb(b).FaceColor = grayCols(b,:);
    hb(b).EdgeColor = 'none';
end

set(ax5, ...
    'XTick', 1:nP, ...
    'XTickLabel', partitionLabels, ...
    'XTickLabelRotation', 35, ...
    'TickLabelInterpreter','none', ...
    'Box','off', ...
    'TickDir','out', ...
    'TickLength',[.01 .01], ...
    'LineWidth',0.5, ...
    'FontName','Times New Roman', ...
    'FontSize',10);

ylabel(ax5, 'conditional entropy [bits]', ...
    'Interpreter','latex');

ylim(ax5, [0, max(M2(:))*1.18]);

axis(ax5, 'square');

drawManualBarLegend(ax5, grayCols, ...
    {'$H(Y\mid C)$', '$H(A\mid C)$'}, ...
    0.08, 1.05, ...   % xAnchor, yAnchor in axes-relative coordinates
    0.18, 0.020, ...  % swatch width, swatch height
    0.080, ...        % vertical spacing
    10);               % font size
axis square

labelSubplots(ax5, 'D', [0.1 0.03], false, ...
    'FontSize',14, ...
    'FontName','Times New Roman', ...
    'FontWeight','normal');

save_figure(fig2, P.figDir, 'figure02', P);

%% ======================= DEMONSTRATION 2 ================================
% Target-dependence: the same category map can be useful or useless
% depending on which consequence variable must be preserved.

targets = struct([]);
targets(end+1).name = 'identity Y';
targets(end).shortName = 'Y';
targets(end).T = Y;
targets(end+1).name = 'action A';
targets(end).shortName = 'A';
targets(end).T = A;
targets(end+1).name = 'nuisance/context V';
targets(end).shortName = 'V';
targets(end).T = Vbin;
targets(end+1).name = 'sensory geometry G';
targets(end).shortName = 'G';
targets(end).T = Gtarget;

nT = numel(targets);
I_mat = zeros(nP,nT);
NMI_target = zeros(nP,nT);
Eff_mat = zeros(nP,nT);
H_targets = zeros(1,nT);

for j = 1:nT
    Tj = targets(j).T;
    H_targets(j) = entropy_discrete(Tj);
    for i = 1:nP
        C = partitions(i).C;
        I_mat(i,j) = mutual_information_discrete(C, Tj);
        NMI_target(i,j) = I_mat(i,j) / max(eps, H_targets(j));
        Eff_mat(i,j) = I_mat(i,j) / max(eps, entropy_discrete(C));
    end
end

targetLabels = string({targets.shortName});
metrics2 = table(partitionLabels, I_mat(:,1), I_mat(:,2), I_mat(:,3), I_mat(:,4), ...
                 NMI_target(:,1), NMI_target(:,2), NMI_target(:,3), NMI_target(:,4), ...
    'VariableNames', {'Partition','I_C_Y','I_C_A','I_C_V','I_C_G', ...
                     'NMI_Y','NMI_A','NMI_V','NMI_G'});

fprintf('\nDEMONSTRATION 2\n');
disp(metrics2);

save(fullfile(P.resultsDir, 'demo02_target_dependence.mat'), ...
    'metrics2','I_mat','NMI_target','Eff_mat','targets','partitions','P');
writetable(metrics2, fullfile(P.resultsDir, 'demo02_target_dependence.csv'));


fig3 = figure('Color','w','Position',[80 80 850 350]);
t3 = tiledlayout(1, 2, 'TileSpacing', 'normal', 'Padding', 'normal');

ax = nexttile;
hold(ax, 'on');

imagesc(ax, NMI_target);
axis(ax, 'tight');

colormap(ax, parula);
caxis(ax, [0 1]);

set(ax, ...
    'XTick', 1:nT, ...
    'XTickLabel', targetLabels, ...
    'YTick', 1:nP, ...
    'YTickLabel', partitionLabels, ...
    'TickLength', [0 0], ...
    'TickDir','out', ...
    'Box','off', ...
    'LineWidth',0.5, ...
    'FontName','Times New Roman', ...
    'FontSize',10, ...
    'TickLabelInterpreter','latex');

xlabel(ax, 'Target variable to be preserved', ...
    'Interpreter','latex');

ylabel(ax, 'Candidate category map', ...
    'Interpreter','latex');

cb = colorbar(ax);
cb.Box = 'off';
cb.TickDirection = 'out';
cb.LineWidth = 0.5;
cb.FontName = 'Times New Roman';
cb.FontSize = 10;
cb.Label.String = 'Mutual information [normalised]';
cb.Label.Interpreter = 'latex';

for i = 1:nP
    for j = 1:nT
        val = NMI_target(i,j);

        if val < 0.35
            txtCol = 'w';
        else
            txtCol = 'k';
        end

        text(ax, j, i, sprintf('%.2f', val), ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'FontName','Times New Roman', ...
            'FontSize',9, ...
            'Color', txtCol, ...
            'Interpreter','none');
    end
end

labelSubplots(ax, 'A', [0.05 0.03], false, ...
    'FontSize', 14, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal');

axis(ax, 'square');

ax2 = nexttile;
hold(ax2, 'on');

hb = bar(ax2, NMI_target, 'BarWidth', 0.72);

% Grayscale bar colours
grayCols_target = [ ...
    0.20 0.20 0.20; ...
    0.45 0.45 0.45; ...
    0.68 0.68 0.68; ...
    0.82 0.82 0.82];

for b = 1:numel(hb)
    hb(b).FaceColor = grayCols_target(b,:);
    hb(b).EdgeColor = 'none';
end

% Make a little bracket and label for the zero-only conditions
yMax = max(NMI_target(:));
yAnn = 0.05 * yMax;      % height of the bracket above zero
tickH = 0.015 * yMax;    % small vertical tick height

x1 = 5 - 0.22;   % Random
x2 = 6 + 0.22;   % One category

plot(ax2, [x1 x2], [yAnn yAnn], 'k-', 'LineWidth', 0.8);
plot(ax2, [x1 x1], [yAnn-tickH yAnn+tickH], 'k-', 'LineWidth', 0.8);
plot(ax2, [x2 x2], [yAnn-tickH yAnn+tickH], 'k-', 'LineWidth', 0.8);

text(ax2, mean([x1 x2]), yAnn + 0.04*yMax, 'all zero', ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','bottom', ...
    'FontName','Times New Roman', ...
    'FontSize',10);

set(ax2, ...
    'XTick', 1:nP, ...
    'XTickLabel', partitionLabels, ...
    'XTickLabelRotation', 35, ...
    'TickLabelInterpreter','none', ...
    'Box','off', ...
    'TickDir','out', ...
    'TickLength',[.01 .01], ...
    'LineWidth',0.5, ...
    'FontName','Times New Roman', ...
    'FontSize',10);

ylabel(ax2, 'Mutual information [normalised]', ...
    'Interpreter','latex');

ylim(ax2, [0, max(NMI_target(:))*1.18]);

legLabels2 = { ...
    '$Y$: identity target', ...
    '$A$: action target', ...
    '$V$: nuisance target', ...
    '$G$: geometry target'};

drawManualBarLegend(ax2, grayCols_target, ...
    legLabels2, ...
    0.675, 1.0, ...   % xAnchor, yAnchor in axes-relative coordinates
    0.12, 0.020, ...  % swatch width, swatch height
    0.075, ...        % vertical spacing
    10);              % font size

labelSubplots(ax2, 'B', [0.1 0.03], false, ...
    'FontSize', 14, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal');

save_figure(fig3, P.figDir, 'figure03', P);

%% ======================= DEMONSTRATION 3 ================================
% Robustness/invariance: a category is more objective if it preserves the
% relevant information across perturbations. Here we perturb the sensory
% coordinates and apply fixed category readouts learned at baseline.

noiseLevels = linspace(0, 2.2, 12);
nRep = 30;

readoutNames = { ...
    'identity readout', ...
    'action readout', ...
    'geometric readout', ...
    'nuisance category', ...
    'random partition', ...
    'one category'};

nR = numel(readoutNames);

IY_noise = zeros(numel(noiseLevels), nR);
IA_noise = zeros(numel(noiseLevels), nR);

% Fixed baseline readouts.
classMeans = zeros(P.K, 2);
for k = 1:P.K
    classMeans(k,:) = mean(S(Y == k,:), 1);
end

% --- Find geometric partition robustly ---
partitionLabelsStr = string(partitionLabels);

idxGeom = find(contains(partitionLabelsStr, "geometric", "IgnoreCase", true), 1);

if isempty(idxGeom)
    error('Could not find geometric partition. Available partitionLabels are: %s', ...
        strjoin(partitionLabelsStr, ', '));
end

baseGeomC = partitions(idxGeom).C;

geomCenters = zeros(P.K, 2);
for k = 1:P.K
    geomCenters(k,:) = mean(S(baseGeomC == k,:), 1);
end

% --- Nuisance readout based on fixed projection edges ---
projBase = S * P.viewGain(:);

nuisanceEdges = quantile(projBase, linspace(0, 1, P.K + 1));
nuisanceEdges(1) = -inf;
nuisanceEdges(end) = inf;

for e = 2:numel(nuisanceEdges)
    if nuisanceEdges(e) <= nuisanceEdges(e-1)
        nuisanceEdges(e) = nuisanceEdges(e-1) + eps;
    end
end

% --- Find random partition robustly ---
idxRandom = find(contains(partitionLabelsStr, "random", "IgnoreCase", true), 1);

if isempty(idxRandom)
    error('Could not find random partition. Available partitionLabels are: %s', ...
        strjoin(partitionLabelsStr, ', '));
end

randomFixed = partitions(idxRandom).C;

for n = 1:numel(noiseLevels)
    sig = noiseLevels(n);
    tmpIY = zeros(nRep,nR);
    tmpIA = zeros(nRep,nR);
    for r = 1:nRep
        Sper = S + sig*randn(size(S));

        C_id_readout = nearest_center_labels(Sper, classMeans);
        C_action_readout = ones(P.N,1);
        C_action_readout(C_id_readout == 2) = 2;   % identities 1 and 3 imply action 1
        C_geom_readout = nearest_center_labels(Sper, geomCenters);
        % The nuisance category is intentionally defined by the nuisance/context
        % variable, not by identity or action. It should therefore remain
        % largely uninformative about Y and A.
        C_nuisance_readout = Vbin;
        C_one = ones(P.N,1);

        Ccell = {C_id_readout, C_action_readout, C_geom_readout, C_nuisance_readout, randomFixed, C_one};
        for q = 1:nR
            tmpIY(r,q) = mutual_information_discrete(Ccell{q}, Y);
            tmpIA(r,q) = mutual_information_discrete(Ccell{q}, A);
        end
    end
    IY_noise(n,:) = mean(tmpIY,1);
    IA_noise(n,:) = mean(tmpIA,1);
end

HY = entropy_discrete(Y);
HA = entropy_discrete(A);
NMIY_noise = IY_noise / max(eps,HY);
NMIA_noise = IA_noise / max(eps,HA);

rows = [];
for n = 1:numel(noiseLevels)
    for q = 1:nR
        rows = [rows; {noiseLevels(n), string(readoutNames{q}), IY_noise(n,q), IA_noise(n,q), NMIY_noise(n,q), NMIA_noise(n,q)}]; %#ok<AGROW>
    end
end
metrics3 = cell2table(rows, 'VariableNames', {'NoiseLevel','Readout','I_C_Y','I_C_A','NMI_Y','NMI_A'});

fprintf('\nDEMONSTRATION 3\n');
disp(metrics3(1:min(12,height(metrics3)),:));

save(fullfile(P.resultsDir, 'demo03_robustness.mat'), ...
    'metrics3','noiseLevels','readoutNames','IY_noise','IA_noise','NMIY_noise','NMIA_noise','P');
writetable(metrics3, fullfile(P.resultsDir, 'demo03_robustness.csv'));

fig4 = figure('Color','w', ...
    'Position',[100 100 850 350]);

t4 = tiledlayout(fig4, 1, 2, ...
    'TileSpacing','normal', ...
    'Padding','normal');

% Shared grayscale line colours
grayCols_robust = [ ...
    0.10 0.10 0.10; ...
    0.28 0.28 0.28; ...
    0.45 0.45 0.45; ...
    0.60 0.60 0.60; ...
    0.75 0.75 0.75; ...
    0.88 0.88 0.88];

lineStyles_robust  = {'-', '-', '-', '--', ':', '-'};
markerStyles_robust = {'o', 's', '^', 'd', 'v', 'none'};

% -------------------------------------------------------------------------
% A. Identity-relevant information
% -------------------------------------------------------------------------
axA = nexttile(t4);
hold(axA, 'on');

hA = gobjects(numel(readoutNames),1);

for r = 1:numel(readoutNames)
    hA(r) = plot(axA, noiseLevels, NMIY_noise(:,r), ...
        'LineStyle', lineStyles_robust{r}, ...
        'Marker', markerStyles_robust{r}, ...
        'Color', grayCols_robust(r,:), ...
        'LineWidth', 1.2, ...
        'MarkerSize', 4, ...
        'MarkerFaceColor', grayCols_robust(r,:), ...
        'MarkerEdgeColor', grayCols_robust(r,:));
end

set(axA, ...
    'Box','off', ...
    'TickDir','out', ...
    'TickLength',[.01 .01], ...
    'LineWidth',0.5, ...
    'FontName','Times New Roman', ...
    'FontSize',10);
axis square

xlabel(axA, 'Perturbation strength', ...
    'Interpreter','latex');

ylabel(axA, 'Identity information [normalised]', ...
    'Interpreter','latex');

ylim(axA, [0 1.05]);
xlim(axA, [min(noiseLevels) max(noiseLevels)]);

labelSubplots(axA, 'A', [0.1 0.01], false, ...
    'FontSize', 14, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal');

% -------------------------------------------------------------------------
% B. Action-relevant information
% -------------------------------------------------------------------------
axB = nexttile(t4);
hold(axB, 'on');

hB = gobjects(numel(readoutNames),1);

for r = 1:numel(readoutNames)
    hB(r) = plot(axB, noiseLevels, NMIA_noise(:,r), ...
        'LineStyle', lineStyles_robust{r}, ...
        'Marker', markerStyles_robust{r}, ...
        'Color', grayCols_robust(r,:), ...
        'LineWidth', 1.2, ...
        'MarkerSize', 4, ...
        'MarkerFaceColor', grayCols_robust(r,:), ...
        'MarkerEdgeColor', grayCols_robust(r,:));
end

set(axB, ...
    'Box','off', ...
    'TickDir','out', ...
    'TickLength',[.01 .01], ...
    'LineWidth',0.5, ...
    'FontName','Times New Roman', ...
    'FontSize',10);
axis square

xlabel(axB, 'Perturbation strength', ...
    'Interpreter','latex');

ylabel(axB, 'Action information [normalised]', ...
    'Interpreter','latex');

ylim(axB, [0 1.05]);
xlim(axB, [min(noiseLevels) max(noiseLevels)]);

labelSubplots(axB, 'B', [0.1 0.01], false, ...
    'FontSize', 14, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal');


% -------------------------------------------------------------------------
% Legend
% -------------------------------------------------------------------------
legendLabels_robust = { ...
    'Identity readout', ...
    'Action readout', ...
    'Geometric readout', ...
    'Nuisance category', ...
    'Random partition', ...
    'One category'};

legend(axB, hB, legendLabels_robust, ...
    'Location','eastoutside', ...
    'Interpreter','none', ...
    'Box','off', ...
    'FontName','Times New Roman', ...
    'FontSize',9);

% -------------------------------------------------------------------------
% Compact hardcoded overlap annotations
% -------------------------------------------------------------------------

% Panel A: identity + geometric overlap
[xTipA, yTipA]   = data2norm(axA, 1.10, 0.65);
[xTextA, yTextA] = data2norm(axA, 1.30, 0.7);

annotation(fig4, 'textarrow', [xTextA xTipA], [yTextA yTipA], ...
    'String', sprintf('Identity + Geometric\noverlap'), ...
    'FontName', 'Times New Roman', ...
    'FontSize', 9, ...
    'Interpreter', 'none', ...
    'LineWidth', 0.8, ...
    'HeadStyle', 'vback2');

% Panel A: nuisance/random/one-category overlap at zero
[xTipA0, yTipA0]   = data2norm(axA, 1.4, 0.125);
[xTextA0, yTextA0] = data2norm(axA, 1.2, 0.2);

annotation(fig4, 'textarrow', [xTextA0 xTipA0], [yTextA0 yTipA0], ...
    'String', sprintf('Nuisance + Random +\nOne-category at zero'), ...
    'FontName', 'Times New Roman', ...
    'FontSize', 9, ...
    'Interpreter', 'none', ...
    'LineWidth', 0.8, ...
    'HeadStyle', 'vback2');

% Panel B: identity + action + geometric overlap
[xTipB, yTipB]   = data2norm(axB, 1.10, 0.65);
[xTextB, yTextB] = data2norm(axB, 1.30, 0.7);

annotation(fig4, 'textarrow', [xTextB xTipB], [yTextB yTipB], ...
    'String', sprintf('Identity + Action +\nGeometric overlap'), ...
    'FontName', 'Times New Roman', ...
    'FontSize', 9, ...
    'Interpreter', 'none', ...
    'LineWidth', 0.8, ...
    'HeadStyle', 'vback2');

% Panel B: nuisance/random/one-category overlap at zero
[xTipB0, yTipB0]   = data2norm(axB, 1.4, 0.125);
[xTextB0, yTextB0] = data2norm(axB, 1.2, 0.2);

annotation(fig4, 'textarrow', [xTextB0 xTipB0], [yTextB0 yTipB0], ...
    'String', sprintf('Nuisance + Random +\nOne-category at zero'), ...
    'FontName', 'Times New Roman', ...
    'FontSize', 9, ...
    'Interpreter', 'none', ...
    'LineWidth', 0.8, ...
    'HeadStyle', 'vback2');

save_figure(fig4, P.figDir, 'figure04', P);


% if P.saveSeparateRobustness
%     fig9 = figure('Color','w','Position',[260 260 980 560]);
%     plot_robustness_panel(noiseLevels, NMIY_noise, readoutNames, ...
%         'normalised identity information I(C;Y)/H(Y)', ...
%         'Demonstration 3: robustness of identity-relevant information');
%     legend(readoutNames, 'Location','eastoutside', 'Interpreter','none');
%     save_figure(fig9, P.figDir, 'figure09_demo03_identity_robustness_supplement', P);
% 
%     fig10 = figure('Color','w','Position',[300 300 980 560]);
%     plot_robustness_panel(noiseLevels, NMIA_noise, readoutNames, ...
%         'normalised action information I(C;A)/H(A)', ...
%         'Demonstration 3: robustness of action-relevant information');
%     legend(readoutNames, 'Location','eastoutside', 'Interpreter','none');
%     save_figure(fig10, P.figDir, 'figure10_demo03_action_robustness_supplement', P);
% end


%% ======================= DEMONSTRATION 4 ================================
% Category-level independence: the framework is not tied to subordinate,
% basic, or superordinate classification. Instead, the preferred level of
% categorisation depends on which target variable must be preserved.

P4 = struct();
P4.N = 3600;
P4.Kfine = 6;     % subordinate identities
P4.Kbasic = 3;    % basic-level kinds
P4.Kaction = 2;   % superordinate/action classes
P4.noiseStd = 0.38;

Ysub = sample_categorical(ones(1,P4.Kfine)/P4.Kfine, P4.N);  % subordinate identity: 1..6
Ybasic = ceil(Ysub/2);                                      % two subordinate identities per basic kind
Alevel = ones(P4.N,1);                                      % action/superordinate target
Alevel(Ybasic == 2) = 2;                                    % basic kinds 1 and 3 share action 1

% Three basic-level centres, each containing two subordinate identities.
basicCenters = [-3.0,  0.0;
                 0.0,  2.8;
                 3.0,  0.0];
subOffsets = [-0.45, -0.25;
               0.45,  0.25;
              -0.45,  0.25;
               0.45, -0.25;
              -0.45, -0.25;
               0.45,  0.25];

muSub = zeros(P4.Kfine,2);
for k = 1:P4.Kfine
    b = ceil(k/2);
    muSub(k,:) = basicCenters(b,:) + subOffsets(k,:);
end

S4 = muSub(Ysub,:) + P4.noiseStd*randn(P4.N,2);

levelMaps = struct([]);
levelMaps(end+1).name = 'Subordinate identity';
levelMaps(end).shortName = 'Subordinate';
levelMaps(end).C = Ysub;

levelMaps(end+1).name = 'Basic kind';
levelMaps(end).shortName = 'Basic';
levelMaps(end).C = Ybasic;

levelMaps(end+1).name = 'Superordinate/action';
levelMaps(end).shortName = 'Action';
levelMaps(end).C = Alevel;

levelMaps(end+1).name = 'Geometric basic clustering';
levelMaps(end).shortName = 'Geometry';
levelMaps(end).C = simple_kmeans(S4, P4.Kbasic, 80);

levelMaps(end+1).name = 'Random fine partition';
levelMaps(end).shortName = 'Random';
levelMaps(end).C = randi(P4.Kfine, P4.N, 1);

levelMaps(end+1).name = 'One-category compression';
levelMaps(end).shortName = 'One category';
levelMaps(end).C = ones(P4.N,1);

for i = 1:numel(levelMaps)
    levelMaps(i).C = relabel_consecutive(levelMaps(i).C);
end

levelLabels = string({levelMaps.shortName})';
levelFullLabels = string({levelMaps.name})';

levelTargets = struct([]);
levelTargets(end+1).name = 'Subordinate identity';
levelTargets(end).shortName = 'Y_{sub}';
levelTargets(end).T = Ysub;
levelTargets(end+1).name = 'Basic kind';
levelTargets(end).shortName = 'Y_{basic}';
levelTargets(end).T = Ybasic;
levelTargets(end+1).name = 'Action/superordinate';
levelTargets(end).shortName = 'A';
levelTargets(end).T = Alevel;

nL = numel(levelMaps);
nLT = numel(levelTargets);
I_level = zeros(nL,nLT);
NMI_level = zeros(nL,nLT);
H_levelC = zeros(nL,1);
nCats_level = zeros(nL,1);

for i = 1:nL
    C = levelMaps(i).C;
    H_levelC(i) = entropy_discrete(C);
    nCats_level(i) = numel(unique(C));
    for j = 1:nLT
        Tj = levelTargets(j).T;
        I_level(i,j) = mutual_information_discrete(C,Tj);
        NMI_level(i,j) = I_level(i,j) / max(eps, entropy_discrete(Tj));
    end
end

metrics4 = table(levelLabels, nCats_level, H_levelC, ...
    I_level(:,1), I_level(:,2), I_level(:,3), ...
    NMI_level(:,1), NMI_level(:,2), NMI_level(:,3), ...
    'VariableNames', {'Partition','nCategories','H_C', ...
                      'I_C_Ysub','I_C_Ybasic','I_C_A', ...
                      'NMI_Ysub','NMI_Ybasic','NMI_A'});

fprintf('\nDEMONSTRATION 4\n');
disp(metrics4);

save(fullfile(P.resultsDir, 'demo04_hierarchical_levels.mat'), ...
    'metrics4','S4','Ysub','Ybasic','Alevel','levelMaps','levelTargets', ...
    'I_level','NMI_level','P4','P');
writetable(metrics4, fullfile(P.resultsDir, 'demo04_hierarchical_levels.csv'));

fig5 = figure('Color','w','Position',[100 100 900 600]);
tiledlayout(2,3,'TileSpacing','normal','Padding','normal');
axList = gobjects(nL,1);
panelTitles4 = { ...
    'Subordinate identity', ...
    'Basic-level categories', ...
    'Action/superordinate-like', ...
    'Geometric clustering', ...
    'Random partition', ...
    'One-category compression'};

for i = 1:nL
    axList(i) = nexttile;
    scatter(S4(:,1), S4(:,2), 8, levelMaps(i).C, ...
        'filled', ...
        'MarkerFaceAlpha', 0.45);
   
    axis([-4.5 4.5 -2 4.5]);
    axis square;

    xlabel('$x_1$', 'Interpreter','latex');
    ylabel('$x_2$', 'Interpreter','latex');

    title(panelTitles4{i}, ...
        'Interpreter','none', ...
        'FontSize',10, ...
        'FontWeight','normal');

    set(gca, ...
        'Box','off', ...
        'TickDir','out', ...
        'TickLength',[.01 .01], ...
        'LineWidth',0.5, ...
        'FontName','Times New Roman', ...
        'FontSize',10, ...
        'TickLabelInterpreter','latex');
end

% Add subplot labels
panelLabels = {'A','B','C','D','E','F'};
labelSubplots(axList, panelLabels, [0.1 0.03], false, ...
    'FontSize', 14, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal');

save_figure(fig5, P.figDir, 'figure05', P);


fig6 = figure('Color','w','Position',[80 80 850 350]);
t6 = tiledlayout(1, 2, 'TileSpacing', 'normal', 'Padding', 'normal');

% -------------------------------------------------------------------------
% A. Heatmap: target dependence across hierarchical category levels
% -------------------------------------------------------------------------
ax6 = nexttile(t6);
hold(ax6, 'on');

imagesc(ax6, NMI_level);
axis(ax6, 'tight');

colormap(ax6, parula);
caxis(ax6, [0 1]);
targetTickLabels6 = {'$Y_{\mathrm{sub}}$', '$Y_{\mathrm{basic}}$', '$A$'};
set(ax6, ...
    'XTick', 1:nLT, ...
    'XTickLabel', targetTickLabels6, ...
    'YTick', 1:nL, ...
    'YTickLabel', levelLabels, ...
    'TickLength', [0 0], ...
    'TickDir', 'out', ...
    'Box', 'off', ...
    'LineWidth', 0.5, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 10, ...
    'TickLabelInterpreter', 'latex');

xlabel(ax6, 'Target variable to be preserved', ...
    'Interpreter', 'latex');

ylabel(ax6, 'Candidate category map', ...
    'Interpreter', 'latex');

cb = colorbar(ax6);
cb.Box = 'off';
cb.TickDirection = 'out';
cb.LineWidth = 0.5;
cb.FontName = 'Times New Roman';
cb.FontSize = 10;
cb.Label.String = 'Mutual information [normalised]';
cb.Label.Interpreter = 'latex';

for i = 1:nL
    for j = 1:nLT
        val = NMI_level(i,j);

        if val < 0.35
            txtCol = 'w';
        else
            txtCol = 'k';
        end

        text(ax6, j, i, sprintf('%.2f', val), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'FontName', 'Times New Roman', ...
            'FontSize', 9, ...
            'Color', txtCol, ...
            'Interpreter', 'none');
    end
end

labelSubplots(ax6, 'A', [0.05 0.03], false, ...
    'FontSize', 14, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal');

axis(ax6, 'square');

% -------------------------------------------------------------------------
% B. Bar plot: complexity and preserved target information
% -------------------------------------------------------------------------
ax7 = nexttile(t6);
hold(ax7, 'on');

M7 = [metrics4.H_C, metrics4.I_C_Ysub, metrics4.I_C_Ybasic, metrics4.I_C_A];

hb7 = bar(ax7, M7, 'BarWidth', 0.72);

% Grayscale bar colours
grayCols_7 = [ ...
    0.20 0.20 0.20; ...
    0.42 0.42 0.42; ...
    0.65 0.65 0.65; ...
    0.82 0.82 0.82];

for b = 1:numel(hb7)
    hb7(b).FaceColor = grayCols_7(b,:);
    hb7(b).EdgeColor = 'none';
end

set(ax7, ...
    'XTick', 1:nL, ...
    'XTickLabel', levelLabels, ...
    'XTickLabelRotation', 35, ...
    'TickLabelInterpreter', 'none', ...
    'Box', 'off', ...
    'TickDir', 'out', ...
    'TickLength', [.01 .01], ...
    'LineWidth', 0.5, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 10);

ylabel(ax7, 'Bits', ...
    'Interpreter', 'latex');

ylim(ax7, [0, max(M7(:))*1.18]);
axis(ax7, 'square');

% Mark the all-zero one-category case
zeroGroup7 = find(strcmpi(string(levelLabels), "one category"), 1);

if ~isempty(zeroGroup7)
    yZero7 = 0.10;
    tickH7 = 0.025;

    plot(ax7, [zeroGroup7-0.30, zeroGroup7+0.30], ...
        [yZero7 yZero7], ...
        'k-', 'LineWidth', 0.6, ...
        'HandleVisibility', 'off');

    plot(ax7, [zeroGroup7-0.30, zeroGroup7-0.30], ...
        [yZero7-tickH7 yZero7+tickH7], ...
        'k-', 'LineWidth', 0.6, ...
        'HandleVisibility', 'off');

    plot(ax7, [zeroGroup7+0.30, zeroGroup7+0.30], ...
        [yZero7-tickH7 yZero7+tickH7], ...
        'k-', 'LineWidth', 0.6, ...
        'HandleVisibility', 'off');

    text(ax7, zeroGroup7, yZero7+0.07, 'all zero', ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'Interpreter', 'latex', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 10);
end

drawManualBarLegend(ax7, grayCols_7, ...
    {'$H(C)$', '$I(C;Y_{\mathrm{sub}})$', '$I(C;Y_{\mathrm{basic}})$', '$I(C;A)$'}, ...
    0.8, 1.0, ...   % xAnchor, yAnchor in axes-relative coordinates
    0.15, 0.020, ...  % swatch width, swatch height
    0.075, ...        % vertical spacing
    10);              % font size

labelSubplots(ax7, 'B', [0.1 0.03], false, ...
    'FontSize', 14, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal');

save_figure(fig6, P.figDir, 'figure06', P);

fprintf('\nSaved all data, results, and figures.\n');
fprintf('Figures folder: %s\n', P.figDir);
fprintf('Results folder: %s\n\n', P.resultsDir);

%% ---------------------------- LOCAL FUNCTIONS ---------------------------
function ensure_dir(d)
    if ~exist(d, 'dir')
        mkdir(d);
    end
end

function save_figure(figHandle, figDir, baseName, P)
    % Disable interactive axes toolbars before export. Otherwise MATLAB may
    % export the small pan/zoom toolbar if the mouse happens to hover over
    % an axis during saving.
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

    if P.savePNG
        exportgraphics(figHandle, fullfile(figDir, [baseName '.png']), 'Resolution', P.figRes);
    end
    if P.savePDF
        exportgraphics(figHandle, fullfile(figDir, [baseName '.pdf']), 'ContentType', 'vector');
    end
end

function labels = sample_categorical(p, N)
    p = p(:) / sum(p);
    edges = [0; cumsum(p)];
    r = rand(N,1);
    labels = zeros(N,1);
    for k = 1:numel(p)
        labels(r > edges(k) & r <= edges(k+1)) = k;
    end
end

function H = entropy_discrete(x)
    x = relabel_consecutive(x(:));
    counts = accumarray(x, 1);
    p = counts / sum(counts);
    p = p(p > 0);
    H = -sum(p .* log2(p));
end

function I = mutual_information_discrete(x, y)
    x = relabel_consecutive(x(:));
    y = relabel_consecutive(y(:));
    assert(numel(x) == numel(y), 'x and y must have the same length.');

    nx = max(x); ny = max(y);
    joint = accumarray([x y], 1, [nx ny]);
    Pxy = joint / sum(joint(:));
    Px = sum(Pxy, 2);
    Py = sum(Pxy, 1);

    I = 0;
    for ix = 1:nx
        for iy = 1:ny
            if Pxy(ix,iy) > 0
                I = I + Pxy(ix,iy) * log2(Pxy(ix,iy) / (Px(ix)*Py(iy)));
            end
        end
    end
end

function z = relabel_consecutive(x)
    [~,~,z] = unique(x(:), 'stable');
end

function state = discretize_2d_state(X, nBins)
    x1 = discretize_edges(X(:,1), nBins);
    x2 = discretize_edges(X(:,2), nBins);
    state = sub2ind([nBins nBins], x1, x2);
    state = relabel_consecutive(state);
end

function b = discretize_edges(x, nBins)
    edges = linspace(min(x), max(x), nBins+1);
    edges(1) = -inf; edges(end) = inf;
    b = discretize(x, edges);
    b(isnan(b)) = 1;
end

function b = discretize_quantile(x, nBins)
    q = quantile(x, linspace(0,1,nBins+1));
    q(1) = -inf; q(end) = inf;
    for i = 2:numel(q)
        if q(i) <= q(i-1)
            q(i) = q(i-1) + eps;
        end
    end
    b = discretize(x, q);
    b(isnan(b)) = 1;
    b = relabel_consecutive(b);
end

function idx = simple_kmeans(X, K, nIter)
    N = size(X,1);
    init = randperm(N, K);
    centers = X(init,:);
    idx = ones(N,1);

    for it = 1:nIter
        D = zeros(N,K);
        for k = 1:K
            dif = X - centers(k,:);
            D(:,k) = sum(dif.^2, 2);
        end
        [~, newIdx] = min(D, [], 2);

        if all(newIdx == idx) && it > 1
            break;
        end
        idx = newIdx;

        for k = 1:K
            if any(idx == k)
                centers(k,:) = mean(X(idx == k,:), 1);
            else
                centers(k,:) = X(randi(N),:);
            end
        end
    end
    idx = relabel_consecutive(idx);
end


function plot_robustness_panel(noiseLevels, M, readoutNames, ylab, ttl)
    plot(noiseLevels, M(:,1), '-o', 'LineWidth', 1.2); hold on;
    plot(noiseLevels, M(:,2), '-s', 'LineWidth', 1.2);
    plot(noiseLevels, M(:,3), '-^', 'LineWidth', 1.2);
    plot(noiseLevels, M(:,4), '-d', 'LineWidth', 1.2);
    plot(noiseLevels, M(:,5), ':o', 'LineWidth', 1.2);
    plot(noiseLevels, M(:,6), ':s', 'LineWidth', 1.2);
    xlabel('sensory perturbation strength');
    ylabel(ylab);
    title(ttl);
    ylim([-0.02 1.05]);
    xlim([min(noiseLevels) max(noiseLevels)]);
    box off;
end

function idx = nearest_center_labels(X, centers)
    % Return the index of the nearest centre.
    % Do NOT relabel here: the centre index carries semantic meaning
    % for identity readouts, e.g. centre 2 means latent identity 2.
    N = size(X,1);
    K = size(centers,1);
    D = zeros(N,K);
    for k = 1:K
        dif = X - centers(k,:);
        D(:,k) = sum(dif.^2, 2);
    end
    [~, idx] = min(D, [], 2);
end

function [xn, yn] = data2norm(ax, x, y)
% Convert data coordinates to normalized figure coordinates.

    oldUnits = ax.Units;
    ax.Units = 'normalized';
    axPos = ax.Position;
    ax.Units = oldUnits;

    xl = xlim(ax);
    yl = ylim(ax);

    xn = axPos(1) + (x - xl(1)) / (xl(2) - xl(1)) * axPos(3);
    yn = axPos(2) + (y - yl(1)) / (yl(2) - yl(1)) * axPos(4);
end


