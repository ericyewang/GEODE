% Demo 1
% Ye (Eric) Wang 
% <eric.ye.wang@duke.edu>
% 10000-dimensional data vectors, sample size 500, intrinsic dimension 10, without missing data.
%% Simulate data
% Training set
D = 50000; % ambient dimension
d = 10; % intrinsic dimension
n = 500; % sample size
d_m = 5; % number of missing features
Lam = 5*randn(D,d);
sigS = abs(randn(1));
eta = randn(n,d);
y = eta*Lam'+sqrt(sigS)*randn(n,D);
% Testing set
n_test = 100;
eta_test = randn(n_test,d);
y_test = eta_test*Lam'+sqrt(sigS)*randn(n_test,D);
y_true = [];
for ii = 1:n_test % for those adta vectors with missing values, 5 out of the 10000 dimensions are missing
    tmp = sort(randsample(1:D,d_m,false));
    y_true = [y_true y_test(ii,tmp)];
    y_test(ii,tmp) = nan*ones(d_m,1);
end
%% fit GEODE on the training dataset
d_guess = 20;
opt = GEODE_opt();
[InD,adapt,u,tau,sigmaS,Lambda,mu] = GEODE(y,d_guess,opt);
% Gelman-Rubin diagnostic
[InD1,adapt1,u1,tau1,sigmaS1] = GEODE(y,d_guess,opt);
disp('Potential scale reduction factor for sigma^2');
gelman_rubin_2chain(sigmaS((opt(1)+1):(opt(1)+opt(2))),sigmaS1((opt(1)+1):(opt(1)+opt(2))))
disp('Do both chain selects the same dimensions?')
all(InD{opt(9)} == InD1{opt(9)})
disp('Potential scale reduction factor for u');
for i = InD{opt(9)}
gelman_rubin_2chain(u(i,(opt(1)+1):(opt(1)+opt(2))),u1(i,(opt(1)+1):(opt(1)+opt(2))))
end
%--------------------------------------------------------------------------
% Note: If the potential scale reduction factor is close to one for all 
% scalar random variables, then one can conclude that the markov chain
% has likely converged.
%--------------------------------------------------------------------------
% Mixing of the MCMC (effective sample size)
disp('Effective sample size of sigma^2');
ess(sigmaS((opt(1)+1):(opt(1)+opt(2))))
disp('Effective sample sizes of u');
ess(u(InD{opt(9)},(opt(1)+1):(opt(1)+opt(2))))

%% Explore the posterior
% Inclusion probability
figure(1)
IProb = inclusProb(d_guess,InD,adapt,opt(9));
% Estimating sigma^2
figure(2)
subplot(1,2,1), plot(1:opt(2),sigmaS((opt(1)+1):(opt(1)+opt(2))));
title('Traceplot of \sigma^2 after burn-in');
subplot(1,2,2), histogram(sigmaS((opt(1)+1):(opt(1)+opt(2))),30);
title('Posterior Distribution of \sigma^2');
disp('Posterior mean of sigma^2');
mean(sigmaS((opt(1)+1):(opt(1)+opt(2)))) % Posterior mean
disp('Absolute deviance from the true sigma^2');
abs(sigS - mean(sigmaS((opt(1)+1):(opt(1)+opt(2))))) % Absolute deviance from the truth

%% Make prediction on the testing dataset based on the fitted GEODE
[pred_summary,pred_sample] = GEODE_predict(y_test,Lambda,mu,InD{opt(9)},...
    u(:,(opt(1)+1):(opt(1)+opt(2))),sigmaS((opt(1)+1):(opt(1)+opt(2))));
% Predictive MSE
disp('Predictive Mean Square Error');
mean((y_true - pred_summary(:,3)').^2)
%--------------------------------------------------------------------------
% Note: If the prediction is accurate, the predictive MSE should be close to
% the estimate of sigmaS. Intuitively, this means that the only thing that 
% the preditive distribution cannot explain is the random noise.
%--------------------------------------------------------------------------