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
alpha0 = opt(7); alpha1 = opt(8); stoptime = opt(9);
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
    if (iter < stoptime)
        if rand(1) <= exp(alpha0+alpha1*iter)
            %fprintf('adapt!\n');
            adapt(iter) = 1;
            ind = InD{iter-1};
            vec = (1./u(InD{iter-1},iter)-1)*sigmaS(iter);
            if sum(vec/max(vec) < tol)==0 && length(InD{iter-1})<d
            % If all dimensions are non-trivial, then add another
                Index = 1:d; Index(InD{iter-1}) = [];
                ind(end+1) = min(Index);
                ind = sort(ind);
            else
                ind(vec/max(vec) < tol) = [];
            end
            InD{iter} = ind;
        else
            InD{iter} = InD{iter-1};
        end
        InDtmp = InD{iter};
    elseif (iter == stoptime)
        adapt(iter) = 1;
        ind = InD{iter-1};
        vec = (1./u(InD{iter-1},iter)-1)*sigmaS(iter);
        if sum(vec/max(vec) < tol)>0
            % delete all trivial dimensions
            ind(vec/max(vec) < tol) = [];
        end
        InDtmp = ind;
        InD{iter} = ind;
    end
end