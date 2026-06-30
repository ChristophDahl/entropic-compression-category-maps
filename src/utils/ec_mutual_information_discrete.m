function I = ec_mutual_information_discrete(X, Y)
%EC_MUTUAL_INFORMATION_DISCRETE Mutual information I(X;Y) in bits.
I = ec_entropy_discrete(X) + ec_entropy_discrete(Y) - ec_joint_entropy_discrete(X, Y);
I = max(I, 0); % numerical guard
end
