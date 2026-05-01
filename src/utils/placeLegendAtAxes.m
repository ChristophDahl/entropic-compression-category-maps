function hLeg = placeLegendAtAxes(ax, x, yTop, hObjs, labels, varargin)
% placeLegendAtAxes
%
% Places a legend relative to an axes object.
%
% x    = left edge of legend in axes-normalized coordinates
% yTop = top edge of legend in axes-normalized coordinates
%
% Example:
%   hLeg = placeLegendAtAxes(ax, 0.05, 0.95, hb, labels, ...
%       'Interpreter','latex', ...
%       'Orientation','vertical', ...
%       'Box','off');

    hLeg = legend(ax, hObjs, labels, varargin{:});

    set(hLeg, ...
        'Units','normalized', ...
        'AutoUpdate','off', ...
        'Location','none');

    drawnow;

    ax.Units = 'normalized';
    axPos  = get(ax, 'Position');      % figure-normalized
    legPos = get(hLeg, 'Position');    % figure-normalized

    % x is the legend left edge.
    legPos(1) = axPos(1) + x * axPos(3);

    % yTop is the legend top edge.
    legPos(2) = axPos(2) + yTop * axPos(4) - legPos(4);

    set(hLeg, 'Position', legPos);
end