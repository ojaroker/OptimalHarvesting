function [p1, p2, q1, q2] = backward(x1, x2, u1, u2, xi1, xi2, dt, N, M, params)
% Stable backward BSDE solver for Model A

alpha  = params.alpha;
beta   = params.beta;
gamma  = params.gamma;
delta  = params.delta;
sigma1 = params.sigma1;
sigma2 = params.sigma2;
gamma1 = params.gamma1;
gamma2 = params.gamma2;
epsilon = params.epsilon;

% Initialize arrays
p1 = zeros(N+1,M); p2 = zeros(N+1,M);
q1 = zeros(N,M);   q2 = zeros(N,M);

% Terminal condition
p1(N+1,:) = 0; p2(N+1,:) = 0;

state_floor = epsilon;      % Prevent division by zero
q_max = 1e3;                % Optional max cap on q

for i = N:-1:1
    % Drift updates with floor
    f1 = (alpha - beta*x2(i,:)) .* p1(i+1,:) + delta*x2(i,:) .* p2(i+1,:) ...
        + sigma1*q1(i,:) + gamma1*(u1(i,:).^2) ./ max(x1(i,:).^2,state_floor);

    f2 = -beta*x1(i,:) .* p1(i+1,:) + (-gamma + delta*x1(i,:)) .* p2(i+1,:) ...
        + sigma2*q2(i,:) + gamma2*(u2(i,:).^2) ./ max(x2(i,:).^2,state_floor);

    % Euler update
    p1(i,:) = p1(i+1,:) + f1*dt + q1(i,:) .* sqrt(dt) .* xi1(i,:);
    p2(i,:) = p2(i+1,:) + f2*dt + q2(i,:) .* sqrt(dt) .* xi2(i,:);

    % --- Regression for q ---
    Phi = [ones(M,1), x1(i,:)', x2(i,:)'];   % Simple linear basis

    % Targets: project p-increment onto xi
    Y1 = (p1(i,:)-p1(i+1,:))' ./ sqrt(dt);   % Mx1
    Y2 = (p2(i,:)-p2(i+1,:))' ./ sqrt(dt);

    % Solve q via linear regression
    lambda = 1e-3;
    beta1 = (Phi'*Phi + lambda*eye(size(Phi,2))) \ (Phi'* (Y1 .* xi1(i,:)'));
    beta2 = (Phi'*Phi + lambda*eye(size(Phi,2))) \ (Phi'* (Y2 .* xi2(i,:)'));

    % Compute q and cap extreme values
    q1(i,:) = min(max((Phi*beta1)', -q_max), q_max);
    q2(i,:) = min(max((Phi*beta2)', -q_max), q_max);
end
end
