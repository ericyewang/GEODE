% Demo 4
% Ye (Eric) Wang
% <eric.ye.wang@duke.edu>
% Accuracy and Time Complexity Comparison between SVD, iterative SVD and fast rank-d SVD.
%% Time complexity comparison between iterative SVD and fast rank-d SVD
n = 500; % sample size
vD = 10.^(2:5);
d = 20;
nD = length(vD);
ctime = zeros(nD,2);
for rep = 1:nD
    D = vD(rep); % ambient dimension
    Lam = 5*randn(D,d);
    sigS = abs(randn(1));
    eta = randn(n,d);
    y = eta*Lam'+sqrt(sigS)*randn(n,D);
    tic;
    [Lambda,~,~] = randPCA(y',d);
    ctime(rep,1) = toc;
    tic;
    [Lambda,~,~] = svds(y',d);
    ctime(rep,2) = toc;
end
disp('Run time for fast rank-d SVD:');
ctime(:,1)
disp('Run time for iterative SVD:');
ctime(:,2)
%--------------------------------------------------------------------------
% Note: SVD is computing all the singular values and vectors and is almost
% certainly using a much longer running time than both iterative SVD and fast
% rank-d SVD. Hence we did not consider it in this comparison.
%--------------------------------------------------------------------------
%% Accuracy Comparison between SVD, fast rank-d SVD and PPCA
vD = [50,100,150,200,250,300];
vd = [20,20,20,20,20,20];
nD = length(vD);
n = 500; % sample size
n_test = 20;
vmse = zeros(nD,3);
vmsesd = zeros(nD,3);
vs = abs(randn(nD,1));
for rep = 1:nD
    rep
    % Training set
    D = vD(rep); % ambient dimension
    d = vd(rep); % intrinsic dimension

    d_m = 5; % number of missing features
    Lam = 5*randn(D,d);
    sigS = vs(rep);
    eta = randn(n,d);
    y = eta*Lam'+sqrt(sigS)*randn(n,D);
    % Testing set
    eta_test = randn(n_test,d);
    y_test = eta_test*Lam'+sqrt(sigS)*randn(n_test,D);
    y_true = [];
    for ii = 1:n_test % for those adta vectors with missing values, 5 out of the 10000 dimensions are missing
        tmp = sort(randsample(1:D,d_m,false));
        y_true = [y_true y_test(ii,tmp)];
        y_test(ii,tmp) = nan*ones(d_m,1);
    end
    
    d_guess = 30;
    opt = GEODE_opt();
    % fast rank-d SVD
    [InD1,~,u1,~,sigmaS1,Lambda1,mu] = GEODE(y,d_guess,opt,true);
    [pred_summary1,~] = GEODE_predict(y_test,Lambda1,mu,InD1{opt(9)},...
        u1(:,(opt(1)+1):(opt(1)+opt(2))),sigmaS1((opt(1)+1):(opt(1)+opt(2))));
    % Predictive MSE
    vmse(rep,1) = mean((y_true - pred_summary1(:,3)').^2);
    vmsesd(rep,1) = std((y_true - pred_summary1(:,3)').^2);
    
    % standard SVD
    [InD2,~,u2,~,sigmaS2,Lambda2,mu] = GEODE(y,d_guess,opt,false);
    [pred_summary2,~] = GEODE_predict(y_test,Lambda2,mu,InD2{opt(9)},...
        u2(:,(opt(1)+1):(opt(1)+opt(2))),sigmaS2((opt(1)+1):(opt(1)+opt(2))));
    % Predictive MSE
    vmse(rep,2) = mean((y_true - pred_summary2(:,3)').^2);
    vmsesd(rep,2) = std((y_true - pred_summary2(:,3)').^2);
    
    % PPCA
    [coeff,score,pcvariance,mu,v,S] = ppca(y,d);
    mC = coeff*diag(pcvariance)*coeff'+v*eye(D);
    ppca_pred = [];
    for j = 1:n_test
        pos = find(isnan(y_test(j,:)));
        V2 = mC(setdiff(1:D,pos),setdiff(1:D,pos));
        R = mC(pos,setdiff(1:D,pos));
        A1 = R*inv(V2);
        m2 = mu(setdiff(1:D,pos));
        m1 = mu(pos);
        ppca_pred = [ppca_pred, (m1' + A1*(y_test(j,setdiff(1:D,pos))-m2)')'];
    end
    vmse(rep,3) = mean((y_true - ppca_pred).^2);
    vmsesd(rep,3) = std((y_true - ppca_pred).^2);
end
%--------------------------------------------------------------------------
% Note: When D is small, the predictive accuracy (in terms of both the MSE 
% and the standard deviation of the square error) of GEODE, GEODE based on
% standard SVD and PPCA is very close. 
%--------------------------------------------------------------------------
%% Accuracy Comparison between GEODE and PPCA
vD = [50,100,150,200,250,300,350,400,450,500];
vd = [20,20,20,20,20,20,20,20,20,20];
nD = length(vD);
n = 500; % sample size
n_test = 20;
vmse1 = zeros(nD,10);
vmse2 = zeros(nD,10);
vmse3 = zeros(nD,10);
vmse4 = zeros(nD,10);
vs = 0.2*ones(nD,1);
for jj = 1:10
for rep = 1:nD
    rep
    % Training set
    D = vD(rep); % ambient dimension
    d = vd(rep); % intrinsic dimension

    d_m = 5; % number of missing features
    Lam = 5*randn(D,d);
    sigS = vs(rep);
    eta = randn(n,d);
    y = eta*Lam'+sqrt(sigS)*randn(n,D);
    % Testing set
    eta_test = randn(n_test,d);
    y_test = eta_test*Lam'+sqrt(sigS)*randn(n_test,D);
    y_true = [];
    for ii = 1:n_test % for those adta vectors with missing values, 5 out of the 10000 dimensions are missing
        tmp = sort(randsample(1:D,d_m,false));
        y_true = [y_true y_test(ii,tmp)];
        y_test(ii,tmp) = nan*ones(d_m,1);
    end
    
    d_guess = 30;
    opt = GEODE_opt();
    % fast rank-d SVD
    [InD1,~,u1,~,sigmaS1,Lambda1,mu] = GEODE(y,d_guess,opt,true);
    [pred_summary1,~] = GEODE_predict(y_test,Lambda1,mu,InD1{opt(9)},...
        u1(:,(opt(1)+1):(opt(1)+opt(2))),sigmaS1((opt(1)+1):(opt(1)+opt(2))));
    % Predictive MSE
    vmse1(rep,jj) = mean((y_true - pred_summary1(:,3)').^2);
    
    % PPCA (correct dim)
    [coeff,score,pcvariance,mu,v,S] = ppca(y,d);
    mC = coeff*diag(pcvariance)*coeff'+v*eye(D);
    ppca_pred = [];
    for j = 1:n_test
        pos = find(isnan(y_test(j,:)));
        V2 = mC(setdiff(1:D,pos),setdiff(1:D,pos));
        R = mC(pos,setdiff(1:D,pos));
        A1 = R*inv(V2);
        m2 = mu(setdiff(1:D,pos));
        m1 = mu(pos);
        ppca_pred = [ppca_pred, (m1' + A1*(y_test(j,setdiff(1:D,pos))-m2)')'];
    end
    vmse2(rep,jj) = mean((y_true - ppca_pred).^2);
    
    % PPCA (dim+5)
    [coeff,score,pcvariance,mu,v,S] = ppca(y,d+5);
    mC = coeff*diag(pcvariance)*coeff'+v*eye(D);
    ppca_pred = [];
    for j = 1:n_test
        pos = find(isnan(y_test(j,:)));
        V2 = mC(setdiff(1:D,pos),setdiff(1:D,pos));
        R = mC(pos,setdiff(1:D,pos));
        A1 = R*inv(V2);
        m2 = mu(setdiff(1:D,pos));
        m1 = mu(pos);
        ppca_pred = [ppca_pred, (m1' + A1*(y_test(j,setdiff(1:D,pos))-m2)')'];
    end
    vmse3(rep,jj) = mean((y_true - ppca_pred).^2);
    
    % PPCA (dim+10)
    [coeff,score,pcvariance,mu,v,S] = ppca(y,d+10);
    mC = coeff*diag(pcvariance)*coeff'+v*eye(D);
    ppca_pred = [];
    for j = 1:n_test
        pos = find(isnan(y_test(j,:)));
        V2 = mC(setdiff(1:D,pos),setdiff(1:D,pos));
        R = mC(pos,setdiff(1:D,pos));
        A1 = R*inv(V2);
        m2 = mu(setdiff(1:D,pos));
        m1 = mu(pos);
        ppca_pred = [ppca_pred, (m1' + A1*(y_test(j,setdiff(1:D,pos))-m2)')'];
    end
    vmse4(rep,jj) = mean((y_true - ppca_pred).^2);
end
end
[mean(vmse1,2),mean(vmse2,2),mean(vmse3,2),mean(vmse4,2)]'