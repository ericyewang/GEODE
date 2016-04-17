function u = gamrndtruncated(A,B,Range)

smallvalue = 1e-8;
a1 = Range(1); a2 = Range(2);
cdf1 = gamcdf(repmat(a1,[length(A),1]),A,1./B);
cdf2 = gamcdf(repmat(a2,[length(A),1]),A,1./B);
% if cdf1>1-smallvalue && cdf2>1-smallvalue
%     u = a1;
% elseif cdf1<smallvalue && cdf2 < smallvalue
%     u = a2;
% else
%     u = gaminv(cdf1+rand(N,1)*(cdf2-cdf1),A,1/B);
% end

cond1 = ((cdf1>1-smallvalue) & (cdf2>1-smallvalue));
cond2 = ((cdf1<smallvalue) & (cdf2 < smallvalue));
cond3 = (~cond1&(~cond2));
u = a1*cond1 + a2*cond2;
if sum(cond3)>=1
u(cond3==1) = gaminv(cdf1(cond3==1)+rand(sum(cond3),1).*(cdf2(cond3==1)...
    -cdf1(cond3==1)),A(cond3==1),1./B(cond3==1));
end