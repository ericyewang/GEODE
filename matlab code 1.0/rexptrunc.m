function r = rexptrunc(A,Range)

smallvalue = 1e-8;
a1 = Range(1); a2 = Range(2);
cdf1 = expcdf(a1,1/A);
cdf2 = expcdf(a2,1/A);
if (cdf2-cdf1) < smallvalue
    r = a1;
else
    r = expinv(cdf1+rand(1)*(cdf2-cdf1),1/A);
end
