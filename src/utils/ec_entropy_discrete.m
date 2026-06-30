function H = ec_entropy_discrete(X)
%EC_ENTROPY_DISCRETE Entropy of a discrete variable in bits.
X = categorical(X(:));
counts = countcats(removecats(X));
p = counts(:) ./ sum(counts);
p = p(p > 0);
H = -sum(p .* log2(p));
end
