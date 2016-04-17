function u = generateU_root(Z,SIGMAS,N,TAU,IND,U_p)
%%
u = U_p;
prodTau     = cumprod(TAU(IND));
u(IND) = gamrndtruncated(prodTau+N/2,1+sum(Z(:,...
    IND).^2,1)'./SIGMAS/2,[0,1]);