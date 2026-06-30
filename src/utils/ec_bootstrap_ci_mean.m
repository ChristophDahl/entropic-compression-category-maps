function ci = ec_bootstrap_ci_mean(x, nBoot, alpha)
%EC_BOOTSTRAP_CI_MEAN Percentile bootstrap confidence interval for the mean.
if nargin < 2 || isempty(nBoot), nBoot = 1000; end
if nargin < 3 || isempty(alpha), alpha = 0.05; end
x = x(:);
x = x(isfinite(x));
if isempty(x)
    ci = [NaN NaN];
    return;
end
B = nan(nBoot, 1);
n = numel(x);
for b = 1:nBoot
    B(b) = mean(x(randi(n, n, 1)));
end
ci = prctile(B, [100*alpha/2, 100*(1-alpha/2)]);
end
