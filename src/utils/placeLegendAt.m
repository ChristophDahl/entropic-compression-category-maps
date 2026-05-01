function hLeg = placeLegendAt(x, y, labels, varargin)
% x, y are normalized figure coordinates for the legend's lower-left corner
% example:
%   hLeg = placeLegendAt(0.52, 0.42, {'A','B'}, 'Interpreter','latex');

    hLeg = legend(labels, varargin{:});
    set(hLeg, 'Units', 'normalized');
    pos = get(hLeg, 'Position');
    pos(1) = x;
    pos(2) = y;
    set(hLeg, 'Position', pos);
end