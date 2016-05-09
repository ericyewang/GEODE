function [InD,adapt,u,tau,sigmaS,Lambda,mu,id_m,pos_m,yms] = GEODE(y,dim,opt,fast)
% main function of GEODE, can automaitically handle missing data
% y:   N-by-D data matrix;
% dim: the initial guess of the intrinsic dimension
% opt: tuning parameters (see GEODE_opt for details)
% Check number of inputs.
if nargin > 4
    error('myfuns:somefun2:TooManyInputs', ...
        'requires at most 3 optional inputs');
end
% Fill in unset optional values.
switch nargin
    case 3
        fast = true;
end
% main
if any(any(isnan(y)))
    % If there is missing data
    [InD,adapt,u,tau,sigmaS,Lambda,mu,id_m,pos_m,yms] = GEODE_root_m(y,dim,opt,fast);
else
    % If there is no missing data
    [InD,adapt,u,tau,sigmaS,Lambda,mu] = GEODE_root(y,dim,opt,fast);
end