function print_demo03_robustness_latex_table()
% Prints a LaTeX table for Demonstration 3 using exact values from
% results/demo03_robustness.csv.

resultsDir = 'I:\entropicCompression\results';
inFile = fullfile(resultsDir, 'demo03_robustness.csv');

T = readtable(inFile);

% Select representative perturbation levels.
wantedNoise = [0.0 0.4 1.0 2.2];

readoutOrder = { ...
    'identity readout', ...
    'geometric readout', ...
    'action readout', ...
    'nuisance category', ...
    'random partition', ...
    'one category'};

readoutLabel = containers.Map( ...
    readoutOrder, ...
    {'Identity','Geometric','Action','Nuisance','Random','One category'});

fprintf('\\begin{table}[t]\n');
fprintf('\\centering\n');
fprintf('\\begin{revtwoblock}\n');
fprintf('\\caption{Representative robustness values for Demonstration 3. Values report normalised mutual information between the category assignment $C$ and the relevant target variable under selected perturbation strengths. Identity-relevant preservation is reported as $I(C;Y)/H(Y)$, and action-relevant preservation is reported as $I(C;A)/H(A)$.}\n');
fprintf('\\label{tab:robustness_quantitative_summary}\n');
fprintf('\\footnotesize\n');
fprintf('\\setlength{\\tabcolsep}{5pt}\n');
fprintf('\\begin{tabular}{llcccc}\n');
fprintf('\\toprule\n');
fprintf('Target & Rule & Noise 0.0 & Noise 0.4 & Noise 1.0 & Noise 2.2 \\\\\n');
fprintf('\\midrule\n');

% Identity rows: use NMI_Y
for i = 1:numel(readoutOrder)
    rname = readoutOrder{i};
    vals = get_values(T, rname, wantedNoise, 'NMI_Y');

    fprintf('Identity $Y$ & %s & %.3f & %.3f & %.3f & %.3f \\\\\n', ...
        readoutLabel(rname), vals(1), vals(2), vals(3), vals(4));
end

fprintf('\\midrule\n');

% Action rows: use NMI_A
for i = 1:numel(readoutOrder)
    rname = readoutOrder{i};
    vals = get_values(T, rname, wantedNoise, 'NMI_A');

    fprintf('Action $A$ & %s & %.3f & %.3f & %.3f & %.3f \\\\\n', ...
        readoutLabel(rname), vals(1), vals(2), vals(3), vals(4));
end

fprintf('\\bottomrule\n');
fprintf('\\end{tabular}\n');
fprintf('\\end{revtwoblock}\n');
fprintf('\\end{table}\n');

end

function vals = get_values(T, readoutName, wantedNoise, valueVar)
vals = nan(size(wantedNoise));

for j = 1:numel(wantedNoise)
    dNoise = abs(T.NoiseLevel - wantedNoise(j));
    isReadout = strcmp(string(T.Readout), string(readoutName));
    idx = find(isReadout & dNoise < 1e-9, 1);

    if isempty(idx)
        % Fall back to nearest available noise level.
        idxAll = find(isReadout);
        [~, kBest] = min(abs(T.NoiseLevel(idxAll) - wantedNoise(j)));
        idx = idxAll(kBest);
    end

    vals(j) = T.(valueVar)(idx);
end
end