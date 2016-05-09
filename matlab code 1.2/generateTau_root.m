function t = generateTau_root(U,a,IND,TAU_p)
% Update tau
t = TAU_p;
temp = 0;
for dim = IND(end:-1:1)
    temp = temp + log(U(dim));
    t(dim) = rexptrunc(a-temp,[1,inf]);
end
