function opt = GEODE_opt()
% Default GEODE parameters
opt = zeros(7,1);
opt(1) = 1000; % number of burn-ins
opt(2) = 2000; % number of posterior samples
opt(3) = 1e-2; % threshold for adaptively removing trivial dimensions
opt(4) = 1/20; % a
opt(5) = 1/2;  % a_sigma
opt(6) = 1/2;  % b_sigma
opt(7) = 100;   % step
opt(8) = floor(0.1*opt(1));  % starting time for adaptive pruning, opt(9) < opt(1)
opt(9) = floor(0.8*opt(1));  % stop time for adaptive pruning, opt(9) < opt(1)