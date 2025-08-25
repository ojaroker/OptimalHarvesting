% test_backward_debug.m
% Debug version: fixed xi, checks for blowups in backward.m (Model A)

clear; clc;

%% Parameters
N  = 2000;              
M  = 256;             
dt = 0.01;             

params = struct();
% Forward parameters
params.alpha  = 1.0;   % prey growth
params.beta   = 0.05;  % predation rate (lower)
params.gamma  = 0.5;   % predator death rate
params.delta  = 0.1;   % predator growth per prey
params.sigma1 = 0.1;   % prey noise
params.sigma2 = 0.05;  % predator noise
params.gamma1  = 2.0;
params.gamma2  = 4.0;
params.F1      = 1;
params.F2      = 1.4;
params.epsilon = 1e-8;

x1_0 = 80;
x2_0 = 25;

% Controls
u1 = 0.1*ones(N,M);
u2 = 0.1*ones(N,M);

%% 1) Generate fixed standard normal noise for forward
rng(123); % reproducible
xi1 = randn(N,M);
xi2 = randn(N,M);

%% 2) Run forward with fixed xi
[x1,x2] = forward(x1_0,x2_0,u1,u2,dt,xi1,xi2,N,M,params);

% Sanity check
if any(x1(:)<params.epsilon)
    warning('Some x1 values hit epsilon (very small) — may cause blowups.');
end

%% 3) Reconstruct xi_hat for backward (should be close to xi)
xi1_hat = xi1;  % use same xi as forward
xi2_hat = xi2;

%% 4) Run backward
[p1,p2,q1,q2] = backward(x1,x2,u1,u2,xi1_hat,xi2_hat,dt,N,M,params);

%% 5) Diagnostics
fprintf('Max abs p1 = %.3e, p2 = %.3e\n', max(abs(p1(:))), max(abs(p2(:))));
fprintf('Max abs q1 = %.3e, q2 = %.3e\n', max(abs(q1(:))), max(abs(q2(:))));

% Reconstruct "xi_hat" from q to check consistency
xi1_recon = zeros(N,M);
xi2_recon = zeros(N,M);

for i = 1:N
    xi1_recon(i,:) = q1(i,:) ./ max(params.sigma1*x1(i,:),1e-12)/sqrt(dt);
    xi2_recon(i,:) = q2(i,:) ./ max(params.sigma2*x2(i,:),1e-12)/sqrt(dt);
end

fprintf('xi1_recon: mean=%.3f, std=%.3f\n', mean(xi1_recon(:)), std(xi1_recon(:)));
fprintf('xi2_recon: mean=%.3f, std=%.3f\n', mean(xi2_recon(:)), std(xi2_recon(:)));

%% 6) Plot q1, q2 mean ± 2 std
% t = (1:N)*dt;
% 
% figure;
% subplot(1,2,1);
% mq1 = mean(q1,2); sq1 = std(q1,0,2);
% fill([t, fliplr(t)], [(mq1-2*sq1)'; fliplr((mq1+2*sq1)')], [0.9 0.9 1], 'EdgeColor','none');
% hold on; plot(t, mq1,'b','LineWidth',1.5);
% title('q1: mean ± 2sd'); xlabel('t'); ylabel('q1'); grid on;
% 
% subplot(1,2,2);
% mq2 = mean(q2,2); sq2 = std(q2,0,2);
% fill([t, fliplr(t)], [(mq2-2*sq2)'; fliplr((mq2+2*sq2)')], [1 0.9 0.9], 'EdgeColor','none');
% hold on; plot(t, mq2,'r','LineWidth',1.5);
% title('q2: mean ± 2sd'); xlabel('t'); ylabel('q2'); grid on;
