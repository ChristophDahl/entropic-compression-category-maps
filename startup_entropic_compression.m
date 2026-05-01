function startup_entropic_compression
% Add source directories for the entropic-compression repository.
thisFile = mfilename('fullpath');
repoRoot = fileparts(thisFile);
addpath(genpath(fullfile(repoRoot, 'src')));
fprintf('Added src/ directories under: %s
', repoRoot);
end
