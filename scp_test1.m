% A 2D mobile robot with attitude sense, docking with a rotating (2D plane)
% robot. The orientation of docking should be head to head and must
% satisfy the objective function (least distance/time) and constraints,
% both convexified using SCP in matlab with good visualization



clc; clear; close all;

%% Parameters
alpha = 0.05;          % regularization
beta  = 20;            % docking penalty weight
trust_radius = 0.5;
N_scp = 20;

% Target
p_target = [5; 5];
theta_t0 = pi/4;
omega_t = 0.15;

% Initial robot state
x_prev = [0; 0; 0];
traj = zeros(3, N_scp);

%% SCP loop
for k = 1:N_scp

    % Target orientation
    theta_t = theta_t0 + omega_t * k;

    % Linearization
    theta_bar = x_prev(3);
    f_bar  = cos(theta_bar - theta_t) + 1;
    df_bar = -sin(theta_bar - theta_t);

    cvx_begin quiet
        cvx_solver sedumi
        variables x(3)

        % Cost function
        minimize( ...
            sum_square(x(1:2) - p_target) + ...
            alpha * sum_square(x - x_prev) + ...
            beta  * square( f_bar + df_bar * (x(3) - theta_bar) ) ...
        )

        subject to
            norm(x - x_prev, 2) <= trust_radius;
    cvx_end

    traj(:,k) = x;
    x_prev = x;
end

%% Visualization
figure; hold on; grid on; axis equal

plot(traj(1,:), traj(2,:), 'bo-', 'LineWidth', 2)
plot(p_target(1), p_target(2), 'rx', 'MarkerSize', 12, 'LineWidth', 3)

for k = 1:N_scp
    quiver(traj(1,k), traj(2,k), ...
           cos(traj(3,k)), sin(traj(3,k)), ...
           0.5, 'k', 'LineWidth', 1.5);
end

title('Stable SCP Head-to-Head Docking (CVX)')
xlabel('x'); ylabel('y')
legend('Robot trajectory','Docking target')
