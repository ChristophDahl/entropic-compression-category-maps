function H = ec_joint_entropy_discrete(X, Y)
%EC_JOINT_ENTROPY_DISCRETE Joint entropy H(X,Y) for discrete variables in bits.
X = categorical(X(:));
Y = categorical(Y(:));
if numel(X) ~= numel(Y)
    error('X and Y must have the same number of observations.');
end
joint = categorical(strcat(string(X), "__", string(Y)));
H = ec_entropy_discrete(joint);
end
