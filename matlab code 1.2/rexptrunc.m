function r = rexptrunc(A,Range)
% tuncated exponential random number generator
% A: exponential distribution parameter(s)
% Range: Range(1) is the lower bound and Range(2) is the upper bound
smallvalue = 1e-8;
a1 = Range(1); a2 = Range(2);
cdf1 = expcdf(a1,1/A);
cdf2 = expcdf(a2,1/A);
if (cdf2-cdf1) < smallvalue
    r = a1;
else
    r = expinv(cdf1+rand(1)*(cdf2-cdf1),1/A);
end
