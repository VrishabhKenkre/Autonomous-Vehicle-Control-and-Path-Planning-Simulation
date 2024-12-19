%Q1-1
syms x_dot real
A = [0, 1, 0, 0;
     0, -42.3594/x_dot, 42.3594, -3.3888/x_dot;
     0, 0, 0, 1;
     0, -0.2475/x_dot, 0.2475, -6.7063/x_dot];

B = [0, 0;
     21.1797, 0;
     0, 0;
     2.3981, 0];

C = eye(4);

symbols = x_dot;
values = [2, 5, 8];

for i = 1:3
    value = values(i);
    A_sub = subs(A, symbols, value);  
    P = ctrb(A_sub, B);
    Q = obsv(A_sub, C); 
    
    fprintf('Controllability Matrix P for V = %.2f:\n', values(i));
    disp(P);
    fprintf("Rank of P = %.4f\n", rank(P))
    
    fprintf('Observability Matrix Q for V = %.2f:\n', values(i));
    disp(Q);
    fprintf("Rank of Q = %.4f\n", rank(Q))
end

%Q1-2
v_values = 1:40; 


log_singular_value_ratio = zeros(1, length(v_values));  % for log10(σ1/σn)
real_parts_of_poles = zeros(4, length(v_values));  % for Re(pi) of each pole


for idx = 1:length(v_values)
    v = v_values(idx);  

   
    A = [0, 1, 0, 0;
         0, -42.3594/v, 42.3594, -3.3888/v;
         0, 0, 0, 1;
         0, -0.2475/v, 0.2475, -6.7063/v];

    B = [0, 0;
         21.1797, 0;
         0, 0;
         2.3981, 0];

    
    P = ctrb(A, B);

    
    singular_values = svd(P);
    log_singular_value_ratio(idx) = log10(singular_values(1) / singular_values(end));
   
    poles = eig(A);
    real_parts_of_poles(:, idx) = real(poles);
end


figure;
plot(v_values, log_singular_value_ratio, 'LineWidth', 1.5);
xlabel('Velocity v (m/s)');
ylabel('log_{10}(\sigma_1 / \sigma_n)');
title('Logarithm of Max to Min Singular Value of P vs Velocity');

for i = 1:4
    figure;
    plot(v_values, real_parts_of_poles(i, :), 'LineWidth', 1.5);
    xlabel('Velocity v (m/s)');
    ylabel(['Re(p_' num2str(i) ')']);
    title(['Real Part of Pole ' num2str(i) ' vs Velocity']);
end