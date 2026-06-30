function Icond = ec_conditional_mutual_information(C, Y, V)
%EC_CONDITIONAL_MUTUAL_INFORMATION Conditional MI I(C;Y|V) in bits.
C = categorical(C(:));
Y = categorical(Y(:));
V = categorical(V(:));
if numel(C) ~= numel(Y) || numel(C) ~= numel(V)
    error('C, Y, and V must have the same number of observations.');
end
V = removecats(V);
levels = categories(V);
Icond = 0;
N = numel(V);
for i = 1:numel(levels)
    idx = V == levels{i};
    if any(idx)
        Icond = Icond + (sum(idx) / N) * ec_mutual_information_discrete(C(idx), Y(idx));
    end
end
Icond = max(Icond, 0);
end
