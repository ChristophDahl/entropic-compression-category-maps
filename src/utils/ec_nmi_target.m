function val = ec_nmi_target(C, T)
%EC_NMI_TARGET Normalised mutual information I(C;T)/H(T).
HT = ec_entropy_discrete(T);
if HT <= 0
    val = 0;
else
    val = ec_mutual_information_discrete(C, T) ./ HT;
end
end
