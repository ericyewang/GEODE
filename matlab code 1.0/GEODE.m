function [InD,adapt,u,tau,sigmaS,Lambda,mu,id_m,pos_m,yms] = GEODE(y,dim,opt)
%% test if there is missing data
if any(any(isnan(y)))
    [InD,adapt,u,tau,sigmaS,Lambda,mu,id_m,pos_m,yms] = GEODE_root_m(y,dim,opt);
else
    [InD,adapt,u,tau,sigmaS,Lambda,mu] = GEODE_root(y,dim,opt);
end