function opt = GEODE_opt()
% Default GEODE parameters
opt = zeros(7,1);
opt(1) = 1000; % number of burn-ins
opt(2) = 2000; % number of posterior samples
opt(3) = 1e-2; % threshold for adaptively removing trivial dimensions
opt(4) = 1/20; % a
opt(5) = 1/2;  % a_sigma
opt(6) = 1/2;  % b_sigma
opt(7) = -1;   % alpha_0
opt(8) = -5e-3;% alpha_1
opt(9) = 800;  % stop time for adaptive pruning, opt(9) < opt(1)