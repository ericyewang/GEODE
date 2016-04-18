function [InD,adapt,u,tau,sigmaS,Lambda,mu,id_m,pos_m,yms] = GEODE_root_m(y,dim,opt)
% fit GEODE on dataset with missing data
%% Prepraration
nb = opt(1); nc = opt(2); tol = opt(3);
a = opt(4); a_sigma = opt(5); b_sigma = opt(6);
alpha0 = opt(7); alpha1 = opt(8); stoptime = opt(9);
[N,D] = size(y); d = dim;
T = nb + nc;
%% Sufficient Statistics
% 1. Extract Missing Positions
id_m = find(any(isnan(y),2));
pos_m = cell(length(id_m),1);
D_m = zeros(length(id_m),1);
for i = 1:length(id_m)
    pos_m{i} = find(isnan(y(id_m(i),:)));
    D_m(i) = length(pos_m{i});
end
% 2. Learn Principal Axes
y_complete = y(setdiff(1:N,id_m),:);
mu = mean(y_complete)';
y_c = bsxfun(@minus,y_complete,mu');
[Lambda,S,~] = randPCA(y_c',d);
% 3. Store Sufficient Statistics
fun1 = @(n) sum((y(n,~isnan(y(n,:)))-mu(~isnan(y(n,:)))').^2);
YY = arrayfun(fun1,1:N)';
fun2 = @(n) (y(n,~isnan(y(n,:)))-mu(~isnan(y(n,:)))')*Lambda(~isnan(y(n,:)),:);
Z = arrayfun(fun2,1:N, 'UniformOutput',0);
Z = cat(1,Z{:});
AA = zeros(d,d,length(id_m));
for k = 1:length(id_m)
    AA(:,:,k) = Lambda(setdiff(1:D,pos_m{i}),:)'*Lambda(setdiff(1:D,pos_m{i}),:);
end
%% Store the data
InD = cell(stoptime,1); adapt = zeros(stoptime,1); 
for t = 1:stoptime 
    InD{t} = 1:d;
end
u      = ones(d,T)/2;
tau    = ones(d,T);
sigmaS = ones(T,1);
yms = cell(T,1);
for i = 1:d
    tau(i,i) = rexptrunc(a,[1,inf]);
end
sigmaS(1) = 1/gamrnd(a_sigma,1/b_sigma);
%% MCMC
InDtmp = 1:d;
for iter = 2:T
    %fprintf('Iteration %d\n',iter);
    
    % Impute Missing Data
    y_m = GEODE_impute(u(:,iter-1),sigmaS(iter-1),Lambda,mu,AA,Z,id_m,pos_m,InDtmp,1);
    yms{iter} = y_m;
    % Update Sufficient Statistics
    YY_temp = YY;
    Z_temp = Z;
    for i = 1:length(id_m)
        YY_temp(id_m(i)) = YY_temp(id_m(i)) + sum((y_m{i}'-mu(pos_m{i})).^2);
        Z_temp(id_m(i),:) = Z_temp(id_m(i),:) + y_m{i}*Lambda(pos_m{i},:);
    end
    % Ordinary Update
    u(:,iter) = generateU_root(Z_temp,sigmaS(iter-1),...
        N,tau(:,iter-1),InDtmp,u(:,iter-1));
    tau(:,iter) = generateTau_root(u(:,iter),a,InDtmp,tau(:,iter-1));
    sigmaS(iter) = generateSigmaS_root(YY_temp,Z_temp,u(:,iter),...
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