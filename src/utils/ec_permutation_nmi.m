function nullVals = ec_permutation_nmi(C, T, nPerm)
%EC_PERMUTATION_NMI Permutation baseline for NMI(C,T).
if nargin < 3, nPerm = 100; end
T = T(:);
nullVals = nan(nPerm, 1);
for p = 1:nPerm
    nullVals(p) = ec_nmi_target(C, T(randperm(numel(T))));
end
end
