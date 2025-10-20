% main.m - Forward-Backward Sweep Algorithm for Model A

clear; clc; close all;

%% Parameters
T=2.5;
N = 3200;            % Number of time steps
M = 100;           % Monte Carlo paths
dt = T/N;           % Time step
tol = 1e-1;         % Convergence tolerance
maxIterations = 60;
uMax = 50;
omega=0.85;

% Model parameters
params = struct();
params.alpha   = 4.5; 
params.beta    = 0.5; 
params.gamma   = 4.5; 
params.delta   = 0.5; 
params.sigma1  = 0.008;
params.sigma2  = 0.004;
params.gamma1  = 2;     % control cost for species 1
params.gamma2  = 1;     % control cost for species 2
params.F1      = 0.8;     % unit price for species 1
params.F2      = 1.2;     % unit price for species 2
params.epsilon = 1e-8;

% Initial states
x1_0 = 30;
x2_0 = 5;

%% Initialization
u1 = zeros(N, M);
u2 = zeros(N, M);


% Pre-allocate storage
x1 = zeros(N+1, M);
x2 = zeros(N+1, M);
p1 = zeros(N+1, M);
p2 = zeros(N+1, M);
q1 = zeros(N, M);
q2 = zeros(N, M);
xi1=zeros(N,M);
xi2=zeros(N,M);

%% Begin Forward-Backward Sweep
for k = 1:maxIterations
    fprintf('Iteration %d...\n', k);

    % --- Forward step ---
    xi1 = randn(N, M);
    xi2 = randn(N, M);
    [x1, x2] = forward(x1_0, x2_0, u1, u2, dt, xi1,xi2,N, M, params);

    % --- Backward step ---
    [p1, p2, q1, q2] = backward(x1, x2, u1, u2, xi1, xi2, dt, N, M, params);
   
    % --- Control update ---
    u1_new = min(uMax,max(0, x1(1:N,:) .* (params.F1 - p1(1:N,:)) / (2 * params.gamma1)));
    u2_new = min(uMax,max(0, x2(1:N,:) .* (params.F2 - p2(1:N,:)) / (2 * params.gamma2)));
    
    u1_new = omega*u1+(1-omega)*u1_new;
    u2_new = omega*u2+(1-omega)*u2_new;
           
        % --- Check convergence ---
    err1 = max(abs(u1_new(:) - u1(:)));
    err2 = max(abs(u2_new(:) - u2(:)));
    fprintf('Max update: u1 = %.5f, u2 = %.5f\n', err1, err2);

    if max(err1, err2) < tol
        fprintf('Converged in %d iterations.\n', k);
        break;
    end
    if min(err1,err2) >= uMax
        disp('exited, reached uMax');
        break;
    end

    % --- Update controls ---
    u1 = u1_new;
    u2 = u2_new;
    
end

%% Output
fprintf('Final mean control:\n');
fprintf('u1(t): %.4f\n', mean(u1(:)));
fprintf('u2(t): %.4f\n', mean(u2(:)));



%% Optimality Validation
% Function to compute objective J
compute_J = @(x1, x2, u1, u2, params) mean(sum( ...
    (params.F1 * u1 - params.gamma1 * (u1.^2) ./ max(x1(1:N,:).^2, params.epsilon) + ...
     params.F2 * u2 - params.gamma2 * (u2.^2) ./ max(x2(1:N,:).^2, params.epsilon)) * dt, 1));

% Compute J for computed controls
J_opt = compute_J(x1, x2, u1, u2, params);
fprintf('Objective J (computed controls): %.4f\n', J_opt);

% Hamiltonian gradient check
H_grad_u1 = params.F1 - 2 * params.gamma1 * u1 ./ max(x1(1:N,:).^2, params.epsilon) - p1(1:N,:);
H_grad_u2 = params.F2 - 2 * params.gamma2 * u2 ./ max(x2(1:N,:).^2, params.epsilon) - p2(1:N,:);
mean_grad_u1 = mean(abs(H_grad_u1(:)));
mean_grad_u2 = mean(abs(H_grad_u2(:)));
fprintf('Mean |dH/du1|: %.4e, Mean |dH/du2|: %.4e\n', mean_grad_u1, mean_grad_u2);

%  martingale check 
martingale_error = max(abs(sum(p1(1:N,:) .* xi1, 2)));
fprintf('Martingale error for p1: %.6e\n', martingale_error);

 
[J_values] = benchmarkJ(J_opt, x1_0, x2_0, u1, u2, xi1, xi2, dt, N, M, params);
disp(sum(J_values>0))




%% Plotting
t = (0:N)*dt;
figure;

% ===== States (x1, x2) =====
subplot(2,1,1); 
hold on;

% Prey (x1)
mx1 = mean(x1, 2);
sx1 = 2 * std(x1, 0, 2);
fill([t fliplr(t)], [mx1'+sx1' fliplr(mx1'-sx1')], ...
    [0.7 0.85 1], 'EdgeColor', 'none', 'FaceAlpha', 0.3, ...
    'HandleVisibility', 'off');
pl1 = plot(t, mx1, 'b', 'LineWidth', 1.5, 'DisplayName', 'x_1 ± 2σ (Prey)');

% Predator (x2)
mx2 = mean(x2, 2);
sx2 = 2 * std(x2, 0, 2);
fill([t fliplr(t)], [mx2'+sx2' fliplr(mx2'-sx2')], ...
    [1 0.7 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.3, ...
    'HandleVisibility', 'off');
pl2 = plot(t, mx2, 'r', 'LineWidth', 1.5, 'DisplayName', 'x_2 ± 2σ (Predator)');

xlabel('Time');
ylabel('Population');
title('State Trajectories');
ylim([0 40])
grid on;
legend([pl1 pl2], 'Location', 'north');

xticks([0 T]);
xticklabels({'0','T'});

% ===== Controls (u1, u2) =====
subplot(2,1,2);
hold on;

% Prey control (u1)
mu1 = mean(u1, 2);
su1 = 2 * std(u1, 0, 2);
fill([t(1:N) fliplr(t(1:N))], [mu1'+su1' fliplr(mu1'-su1')], ...
    [0.7 0.85 1], 'EdgeColor', 'none', 'FaceAlpha', 0.3, ...
    'HandleVisibility', 'off');
pl3 = plot(t(1:N), mu1, 'b', 'LineWidth', 1.5, 'DisplayName', 'u_1 ± 2σ (Prey Control)');

% Predator control (u2)
mu2 = mean(u2, 2);
su2 = 2 * std(u2, 0, 2);
fill([t(1:N) fliplr(t(1:N))], [mu2'+su2' fliplr(mu2'-su2')], ...
    [1 0.7 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.3, ...
    'HandleVisibility', 'off');
pl4 = plot(t(1:N), mu2, 'r', 'LineWidth', 1.5, 'DisplayName', 'u_2 ± 2σ (Predator Control)');

xlabel('Time');
ylabel('Control');
title('Control Trajectories');
ylim([0 inf])
grid on;
legend([pl3 pl4], 'Location', 'north');

xticks([0 T]);
xticklabels({'0','T'});