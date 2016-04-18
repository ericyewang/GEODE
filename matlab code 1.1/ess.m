function n_e = ess(y)
% Effective sample size of a Markov Chain
% y: m-by-n matrix with each row being the samples for one variable;
if (isvector(y))
    acf = autocorr(y);
    n_e = length(y)/(1+2*sum(acf));
else
    [m,n] = size(y);
    n_e = ones(m,1);
    for i = 1:m
        acf = autocorr(y(i,:));
        n_e(i) = n/(1+2*sum(acf));
    end
end
