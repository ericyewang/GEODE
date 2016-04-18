% Demo
% Ye (Eric) Wang 
% <eric.ye.wang@duke.edu>
% This demo contains two data experiments
%% 1. 10000-dimensional data vectors, sample size 500, intrinsic dimension 10, without missing data.
% 1.1 Simulate data
% 1.1.1 Training set
rng('default');
rng(1234);
D = 10000; % ambient dimension
d = 10; % intrinsic dimension
n = 500; % sample size
Lam = 5*randn(D,d);
sigS = abs(randn(1));
eta = randn(n,d);
y = eta*Lam'+sqrt(sigS)*randn(n,D);
% 1.1.2 Testing set
n_test = 100;
eta_test = randn(n_test,d);
y_test = eta_test*Lam'+sqrt(sigS)*randn(n_test,D);
y_true = [];
for ii = 1:n_test % for those adta vectors with missing values, 5 out of the 10000 dimensions are missing
    tmp = sort(randsample(1:D,d_m,false));
    y_true = [y_true y_test(ii,tmp)];
    y_test(ii,tmp) = nan*ones(d_m,1);
end

% 1.2 fit GEODE on the training dataset
d_guess = 30;
opt = GEODE_opt();
[InD,adapt,u,tau,sigmaS,Lambda,mu] = GEODE(y,d_guess,opt);

% 1.3 Explore the posterior
% 1.3.1 Inclusion probability
figure(1)
IProb = inclusProb(d_guess,InD,adapt,opt(9));
% 1.3.2 Estimating sigma^2
figure(2)
subplot(1,2,1), plot(1:opt(2),sigmaS((opt(1)+1):(opt(1)+opt(2))));
title('Traceplot of \sigma^2 after burn-in');
subplot(1,2,2), histogram(sigmaS((opt(1)+1):(opt(1)+opt(2))),30);
title('Posterior Distribution of \sigma^2');
disp('Posterior mean of sigma^2');
mean(sigmaS((opt(1)+1):(opt(1)+opt(2)))) % Posterior mean
disp('Absolute deviance from the true sigma^2');
abs(sigS - mean(sigmaS((opt(1)+1):(opt(1)+opt(2))))) % Absolute deviance from the truth
% 1.3.3 Mixing of the MCMC (effective sample size)
disp('Effective sample size of sigma^2');
ess(sigmaS((opt(1)+1):(opt(1)+opt(2))))
disp('Effective sample sizes of u');
ess(u(InD{opt(9)},(opt(1)+1):(opt(1)+opt(2))))

% 1.4 Make prediction on the testing dataset based on the fitted GEODE
[pred_summary,pred_sample] = GEODE_predict(y_test,Lambda,mu,InD{opt(9)},...
    u(:,(opt(1)+1):(opt(1)+opt(2))),sigmaS((opt(1)+1):(opt(1)+opt(2))));
% 1.4.1 Predictive MSE
disp('Predictive Mean Square Error');
mean((y_true - pred_summary(:,3)').^2)
%--------------------------------------------------------------------------
% Note: If the prediction is accurate, the predictive MSE should be close to
% the estimate of sigmaS. Intuitively, this means that the only thing that 
% the preditive distribution cannot explain is the random noise.
%--------------------------------------------------------------------------

%% 2. 10000-dimensional data vectors, sample size 500, intrinsic dimension 10, with missing data
% 2.1 Simulate data
% 2.1.1 Training set
rng('default');
rng(1234);
D = 10000; % ambient dimension
d = 10; % intrinsic dimension
n = 500; % sample size
n_m = 25; % number of data vectors containing missing features
d_m = 5; % number of missing features
Lam = 5*randn(D,d);
sigS = abs(randn(1));
eta = randn(n,d);
y = eta*Lam'+sqrt(sigS)*randn(n,D);
id = sort(randsample(1:N,n_m,false)); % 25 out of the 500 data vectors contain missing values
for ii = 1:n_m % for those adta vectors with missing values, 5 out of the 10000 dimensions are missing
    tmp = sort(randsample(1:D,d_m,false));
    y(id(ii),tmp) = nan*ones(d_m,1);
end
% 2.1.2 Testing set
n_test = 100;
eta_test = randn(n_test,d);
y_test = eta_test*Lam'+sqrt(sigS)*randn(n_test,D);
y_true = [];
for ii = 1:n_test % for those adta vectors with missing values, 5 out of the 10000 dimensions are missing
    tmp = sort(randsample(1:D,d_m,false));
    y_true = [y_true y_test(ii,tmp)];
    y_test(ii,tmp) = nan*ones(d_m,1);
end
%--------------------------------------------------------------------------
% Note: GEODE imputes the missing values by sampling from their predictive
% distributions at each iteration of the MCMC. The computational time
% will increase linearly in terms of the amount of missing. Hence GEODE is
% most suitable when the missing amount is not large. When the missingness
% is very severe, one can either do a one-time imputation prior to running
% GEODE, or one can run GEODE for a short amount of time, say 100
% iterations, and use these initial draws to compute the predictive means
% and impute the values to the missing data. Then one can fix the
% imputation and run the GEODE as if the dataset is complete.
%--------------------------------------------------------------------------

% 2.2 GEODE
d_guess = 30;
opt = GEODE_opt();
[InD,adapt,u,tau,sigmaS,Lambda,mu] = GEODE(y,d_guess,opt);

% 2.3 Explore the posterior
% 2.3.1 Inclusion probability
figure(1)
IProb = inclusProb(d_guess,InD,adapt,opt(9));
% 2.3.2 Estimating sigma^2
figure(2)
subplot(1,2,1), plot(1:opt(2),sigmaS((opt(1)+1):(opt(1)+opt(2))));
title('Traceplot of \sigma^2 after burn-in');
subplot(1,2,2), histogram(sigmaS((opt(1)+1):(opt(1)+opt(2))),30);
title('Posterior Distribution of \sigma^2');
disp('Posterior mean of sigma^2');
mean(sigmaS((opt(1)+1):(opt(1)+opt(2)))) % Posterior mean
disp('Absolute deviance from the true sigma^2');
abs(sigS - mean(sigmaS((opt(1)+1):(opt(1)+opt(2))))) % Absolute deviance from the truth
% 2.3.3 Mixing of the MCMC (effective sample size)
disp('Effective sample size of sigma^2');
ess(sigmaS((opt(1)+1):(opt(1)+opt(2))))
disp('Effective sample sizes of u');
ess(u(InD{opt(9)},(opt(1)+1):(opt(1)+opt(2))))

% 2.4 Make prediction on the testing dataset based on the fitted GEODE
[pred_summary,pred_sample] = GEODE_predict(y_test,Lambda,mu,InD{opt(9)},...
    u(:,(opt(1)+1):(opt(1)+opt(2))),sigmaS((opt(1)+1):(opt(1)+opt(2))));
% 2.4.1 Predictive MSE
disp('Predictive Mean Square Error');
mean((y_true - pred_summary(:,3)').^2)