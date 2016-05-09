% Demo 3
% Ye (Eric) Wang 
% <eric.ye.wang@duke.edu>
% Prediction coverage.
%% Set up the experiments
vD = [100,100,100,1000,1000,1000,10000,10000,10000,100000,100000,100000];
vd = [10,20,30,10,20,30,10,20,30,10,20,30];
nD = length(vD);
vc = zeros(nD,1);
n = 500; % training set size
d_m = 5; % number of missing features
n_test = 100; % testing set size
%% Begin the experiments
for rep = 1:nD
    rep
    % Training set
    D = vD(rep); % ambient dimension
    d = vd(rep); % intrinsic dimension
    Lam = 5*randn(D,d);
    sigS = abs(randn(1));
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
    d_guess = 40;
    opt = GEODE_opt();
    [InD,adapt,u,tau,sigmaS,Lambda,mu] = GEODE(y,d_guess,opt);
    [pred_summary,psamp] = GEODE_predict(y_test,Lambda,mu,InD{opt(9)},...
        u(:,(opt(1)+1):(opt(1)+opt(2))),sigmaS((opt(1)+1):(opt(1)+opt(2))));
    % Predictive MSE
    vc(rep) = mean( (y_true' < pred_summary(:,6)).*(y_true' > pred_summary(:,5)) );
end