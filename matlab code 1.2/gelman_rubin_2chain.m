function r = gelman_rubin_2chain(s1,s2)
% Gelman Rubin MCMC diagnostic for two chains
% s1: samples from the first chain
% s2: samples from the second chain
n = length(s1);
w = (std(s1)^2 + std(s2)^2)/2;
m1 = mean(s1); m2 = mean(s2);
if w > 0
    mm = (m1+m2)/2;
    b = n*((m1-mm)^2 + (m2-mm)^2);
    v = (1-1/n)*w + b/n;
    r = sqrt(v/w);
else
    r = sqrt(1-1/n);
end