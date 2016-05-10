function [InD,adapt,u,tau,sigmaS,Lambda,mu] = GEODE_root(y,dim,opt,fast)
% fit GEODE on dataset with no missing data
%% Sufficient Statistics
[N,D] = size(y); d = dim;
mu = mean(y)';
y_c = bsxfun(@minus,y,mu');
if fast
    [Lambda,~,~] = randPCA(y_c',dim);
else
    [Lambda,~,~] = svd(y_c','econ');
    Lambda = Lambda(:,1:dim);
end
YY = sum(y_c.^2,2);
Z = bsxfun(@minus,y,mu')*Lambda;
%% Preparation
nb = opt(1); nc = opt(2); tol = opt(3);
a = opt(4); a_sigma = opt(5); b_sigma = opt(6);
step = opt(7); stoptime = opt(9); starttime = opt(8);
T = nb + nc;
%% Store the data
InD = cell(stoptime,1); adapt = zeros(stoptime,1); 
for t = 1:stoptime 
    InD{t} = 1:d;
end
u      = ones(d,T)/2;
tau    = ones(d,T);
sigmaS = ones(T,1);
for i = 1:d
    tau(i,i) = rexptrunc(a,[1,inf]);
end
sigmaS(1) = 1/gamrnd(a_sigma,1/b_sigma);
u_accum = zeros(d,1);
nadpt = floor((stoptime-starttime)/step);
adptpos = [(1:(nadpt-1))*step stoptime];
%% MCMC
InDtmp = 1:d;
for iter = 2:T
   %fprintf('Iteration %d\n',iter);
       u(:,iter) = generateU_root(Z,sigmaS(iter-1),...
           N,tau(:,iter-1),InDtmp,u(:,iter-1));
       tau(:,iter) = generateTau_root(u(:,iter),a,InDtmp,tau(:,iter-1));
       sigmaS(iter) = generateSigmaS_root(YY,Z,u(:,iter),...
           N,a_sigma,b_sigma,D,InDtmp);
   
    % Adaptively prune the intrinsic dimension
    if (iter <= stoptime && iter > starttime)
        u_accum = u_accum + ( u(:,iter)==1 );
        if any(adptpos == iter)
            %fprintf('adapt!\n');
            adapt(iter) = 1;
            ind = InD{iter-1};
            tmp = u_accum(ind);
            vec = (1./u(InD{iter-1},iter)-1)*sigmaS(iter);
            d1 = ind;
            if ( sum(tmp > (stoptime-starttime)*tol) + sum(vec/max(vec) < tol) )>0
                ind( d1 >= min(d1( (tmp > (stoptime-starttime)*tol) | (vec/max(vec) < tol) )) ) = [];
            end
            InD{iter} = ind;
        else
            InD{iter} = InD{iter-1};
        end
        InDtmp = InD{iter};
    end
end