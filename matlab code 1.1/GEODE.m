function [InD,adapt,u,tau,sigmaS,Lambda,mu,id_m,pos_m,yms] = GEODE(y,dim,opt)
% main function of GEODE, can automaitically handle missing data
% y:   N-by-D data matrix;
% dim: the initial guess of the intrinsic dimension
% opt: tuning parameters (see GEODE_opt for details)
if any(any(isnan(y)))
    % If there is missing data
    [InD,adapt,u,tau,sigmaS,Lambda,mu,id_m,pos_m,yms] = GEODE_root_m(y,dim,opt);
else
    % If there is no missing data
    [InD,adapt,u,tau,sigmaS,Lambda,mu] = GEODE_root(y,dim,opt);
end