function Yperm = ec_shuffle_within_group(Y, G)
%EC_SHUFFLE_WITHIN_GROUP Randomly permute labels Y within levels of grouping variable G.
Yperm = Y(:);
G = categorical(G(:));
levels = categories(removecats(G));
for i = 1:numel(levels)
    idx = find(G == levels{i});
    if numel(idx) > 1
        Yperm(idx) = Yperm(idx(randperm(numel(idx))));
    end
end
end
