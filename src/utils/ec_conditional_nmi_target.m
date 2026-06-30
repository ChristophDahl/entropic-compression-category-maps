function val = ec_conditional_nmi_target(C, Y, V)
%EC_CONDITIONAL_NMI_TARGET Conditional NMI I(C;Y|V)/H(Y|V).
HYV = ec_conditional_entropy(Y, V);
if HYV <= 0
    val = 0;
else
    val = ec_conditional_mutual_information(C, Y, V) ./ HYV;
end
end
