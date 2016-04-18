function sigmaS = generateSigmaS_root(YY,Z,U,N,a_sigma,b_sigma,D,IND)
% Update sigma square
SS     = sum(YY-(Z(:,IND).^2)*(1-U(IND)));
sigmaS = 1/gamrnd(a_sigma+D*N/2,1/(b_sigma+SS/2));