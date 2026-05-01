function labelSubplots(axList, labelText, offset, useLatex, varargin)
% labelSubplots  Add panel labels (e.g. 'A', 'B', 'C') to axes.
%
%   labelSubplots(ax, 'A')
%       Places label 'A' at the upper-left outside corner of axes ax,
%       using the default offset and no LaTeX interpreter.
%
%   labelSubplots([ax1 ax2], {'A','B'})
%       Uses the given labels for each axes handle.
%
%   labelSubplots(axList, labelText, offset)
%       offset = [dx dy] in normalised axes units, where the label is
%       placed at (x,y) = (-dx, 1+dy), i.e. dx to the left and dy above
%       the axes box as a fraction of its width/height. LaTeX interpreter
%       is off by default.
%
%   labelSubplots(axList, labelText, offset, useLatex)
%       useLatex is a logical flag:
%           true  -> use 'Interpreter','latex'
%           false -> use 'Interpreter','none'
%
%   labelSubplots(..., 'Name', Value, ...)
%       Extra name-value pairs are passed to TEXT (e.g. 'FontSize', 12).
%       If an 'Interpreter' name-value pair is provided here, it overrides
%       the useLatex flag.

    % --- Defaults ---
    if nargin < 1 || isempty(axList)
        axList = gca;
    end
    axList = axList(:);

    if nargin < 2 || isempty(labelText)
        labelText = {'A'};  % default single label
    end

    if nargin < 3 || isempty(offset)
        offset = [0.02 0.02]; % 2% left, 2% above
    end
    if numel(offset) ~= 2
        error('offset must be [dx dy] in normalised units.');
    end
    dx = offset(1);
    dy = offset(2);

    if nargin < 4 || isempty(useLatex)
        useLatex = false;  % default: no LaTeX
    end
    if ~islogical(useLatex) && ~ismember(useLatex, [0 1])
        error('useLatex must be a logical (true/false).');
    end

    % --- Normalise labels to cellstr with one entry per axes ---
    if ischar(labelText) || isstring(labelText)
        % Single label string: replicate for all axes
        labelText = cellstr(labelText);
    end
    if iscell(labelText)
        if numel(labelText) == 1 && numel(axList) > 1
            labelText = repmat(labelText, numel(axList), 1);
        elseif numel(labelText) ~= numel(axList)
            error('Number of labels must be 1 or match number of axes.');
        end
    else
        error('labelText must be char, string, or cell array of char/string.');
    end

    % Check if user already specified 'Interpreter' in varargin
    hasInterpreter = false;
    if ~isempty(varargin)
        nvNames = varargin(1:2:end);
        hasInterpreter = any(strcmpi(nvNames, 'Interpreter'));
    end

    % --- Place labels ---
    for k = 1:numel(axList)
        ax = axList(k);
        if ~ishandle(ax) || ~strcmp(get(ax, 'Type'), 'axes')
            continue;
        end

        % Use normalised axes coordinates
        xPos = -dx;
        yPos = 1 + dy;

        set(ax, 'Clipping', 'off'); % so text outside is visible

        args = {};
        % Set interpreter only if user did not already specify it
        if ~hasInterpreter
            if useLatex
                args = {'Interpreter','latex'};
            else
                args = {'Interpreter','none'};
            end
        end

        text(ax, xPos, yPos, labelText{k}, ...
            'Units', 'normalized', ...
            'HorizontalAlignment', 'left', ...
            'VerticalAlignment', 'bottom', ...
            'FontWeight', 'bold', ...
            args{:}, ...
            varargin{:});
    end
end
