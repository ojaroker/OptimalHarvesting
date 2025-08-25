%% test_backward_additive_noise.m
% Sanity test for backward.m using additive noise

clear; clc;

%% Parameters
N  = 20000;    % time steps
M  = 256;    % Monte Carlo paths
dt = 0.0001;   % time step

params = struct();
params.alpha   = 1.0;
params.beta    = 0.1;
params.gamma   = 1.2;
params.delta   = 0.15;
params.sigma1  = 0.3;  % now additive noise
params.sigma2  = 0.25; 
params.gamma1  = 4.0;
params.gamma2  = 6.0;
params.F1      = 1.2;
params.F2      = 1.4;
params.epsilon = 1e-8;

%% Initial states
x1_0 = 80;
x2_0 = 25;

%% Controls
u1 = 0.1 * ones(N, M);
u2 = 0.1 * ones(N, M);

%% --- Generate xi explicitly ---
xi1 = randn(N, M);
xi2 = randn(N, M);

%% --- FORWARD with additive noise ---
x1 = zeros(N+1, M);
x2 = zeros(N+1, M);

x1(1,:) = x1_0;
x2(1,:) = x2_0;

for i = 1:N
    f1 = params.alpha*x1(i,:) - params.beta*x1(i,:).*x2(i,:) - u1(i,:);
    f2 = -params.gamma*x2(i,:) + params.delta*x1(i,:).*x2(i,:) - u2(i,:);
    
    g1 = params.sigma1;  % additive
    g2 = params.sigma2;  % additive
    
    x1_next = x1(i,:) + f1*dt + g1*sqrt(dt)*xi1(i,:);
    x2_next = x2(i,:) + f2*dt + g2*sqrt(dt)*xi2(i,:);
    
    % enforce minimum population
    x1(i+1,:) = max(x1_next,1);
    x2(i+1,:) = max(x2_next,1);
end

%% --- BACKWARD with consistent xi ---
[p1, p2, q1, q2] = backward(x1, x2, u1, u2, xi1, xi2, dt, N, M, params);

%% --- Check xi reconstruction ---
xi1_recon = (x1(2:end,:) - x1(1:end-1,:) - (params.alpha*x1(1:end-1,:) - params.beta*x1(1:end-1,:).*x2(1:end-1,:) - u1)*dt) / (params.sigma1*sqrt(dt));
xi2_recon = (x2(2:end,:) - x2(1:end-1,:) - (-params.gamma*x2(1:end-1,:) + params.delta*x1(1:end-1,:).*x2(1:end-1,:) - u2)*dt) / (params.sigma2*sqrt(dt));

fprintf('xi1_recon: mean=%.3f, std=%.3f\n', mean(xi1_recon(:)), std(xi1_recon(:)));
fprintf('xi2_recon: mean=%.3f, std=%.3f\n', mean(xi2_recon(:)), std(xi2_recon(:)));

%% --- Inspect p and q ---
fprintf('Max abs p1=%.3e, p2=%.3e\n', max(abs(p1(:))), max(abs(p2(:))));
fprintf('Max abs q1=%.3e, q2=%.3e\n', max(abs(q1(:))), max(abs(q2(:))));
