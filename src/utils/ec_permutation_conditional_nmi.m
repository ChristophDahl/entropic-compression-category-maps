function nullVals = ec_permutation_conditional_nmi(C, Y, V, nPerm)
%EC_PERMUTATION_CONDITIONAL_NMI Within-nuisance permutation baseline for conditional NMI.
if nargin < 4, nPerm = 100; end
nullVals = nan(nPerm, 1);
for p = 1:nPerm
    Yperm = ec_shuffle_within_group(Y, V);
    nullVals(p) = ec_conditional_nmi_target(C, Yperm, V);
end
end
