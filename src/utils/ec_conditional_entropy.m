function Hcond = ec_conditional_entropy(Y, V)
%EC_CONDITIONAL_ENTROPY Conditional entropy H(Y|V) in bits.
Y = categorical(Y(:));
V = categorical(V(:));
if numel(Y) ~= numel(V)
    error('Y and V must have the same number of observations.');
end
V = removecats(V);
levels = categories(V);
Hcond = 0;
N = numel(V);
for i = 1:numel(levels)
    idx = V == levels{i};
    if any(idx)
        Hcond = Hcond + (sum(idx) / N) * ec_entropy_discrete(Y(idx));
    end
end
end
