clear; clc; close all;

% =========================
% GLOBAL FOR GA HISTORY
% =========================
global ga_history
ga_history = [];

% =========================
% SET WORKING DIRECTORY
% =========================
cd('/Users/saifnizami/Documents/cu_ai_course_repo/subjects/4_7031SCN_Neural_and_Evolutionary_Computing/Report/MATLAB')

% =========================
% LOAD FIS
% =========================
fis = readfis('kidney_fis');

% =========================
% TEST DATA
% =========================
X = [
    1.0 110 30 2;
    1.5 130 40 5;
    2.0 120 50 8;
    2.5 150 60 6;
    3.0 140 65 7;
    3.5 160 70 9;
    4.0 170 75 8
];

Y = [
    20;
    45;
    55;
    65;
    75;
    90;
    95
];

% =========================
% GA SETUP
% =========================
nvars = 6;

lb = [1.0 2.0 110 130 40 60];
ub = [2.5 3.5 140 170 60 90];

opts = optimoptions('ga', ...
    'PopulationSize', 100, ...
    'MaxGenerations', 100, ...
    'Display', 'iter', ...
    'OutputFcn', @gaoutfun);

% =========================
% RUN GA
% =========================
obj = @(x) localFitness(x, fis, X, Y);
[xbest, fbest] = ga(obj, nvars, [], [], [], [], lb, ub, [], opts);

disp('===== BEST PARAMETERS =====')
disp(xbest)

disp('===== BEST ERROR =====')
disp(fbest)

% =========================
% GA CONVERGENCE PLOT
% =========================
figure;
plot(ga_history, '-o', 'LineWidth', 1.5);
xlabel('Generation');
ylabel('Best Fitness (MSE)');
title('GA Convergence Curve');
grid on;

% =========================
% BUILD OPTIMISED FIS
% =========================
optFis = fis;

% Creatinine
optFis.Inputs(1).MembershipFunctions(2).Parameters = [xbest(1)-0.8 xbest(1) xbest(1)+0.8];
optFis.Inputs(1).MembershipFunctions(3).Parameters = [xbest(2)-1.0 xbest(2) xbest(2)+1.0];

% Blood Pressure
optFis.Inputs(2).MembershipFunctions(2).Parameters = [xbest(3)-20 xbest(3) xbest(3)+20];
optFis.Inputs(2).MembershipFunctions(3).Parameters = [xbest(4)-20 xbest(4) xbest(4)+20];

% Output Risk
optFis.Outputs(1).MembershipFunctions(2).Parameters = [xbest(5)-15 xbest(5) xbest(5)+15];
optFis.Outputs(1).MembershipFunctions(3).Parameters = [xbest(6)-15 xbest(6) xbest(6)+15];

% =========================
% BEFORE VS AFTER
% =========================
n = size(X,1);
before = zeros(n,1);
after = zeros(n,1);

for i = 1:n
    before(i) = evalfis(fis, X(i,:));
    after(i) = evalfis(optFis, X(i,:));
end

% =========================
% RESULT TABLE
% =========================
result_table = table(X(:,1), X(:,2), X(:,3), X(:,4), before, after, Y, ...
    'VariableNames', {'Creatinine','BP','Age','Diabetes','Before','After','Target'});

disp('===== RESULT TABLE =====')
disp(result_table)

% =========================
% ERROR METRICS
% =========================
mse_before = mean((before - Y).^2);
mse_after = mean((after - Y).^2);

mae_before = mean(abs(before - Y));
mae_after = mean(abs(after - Y));

improvement = ((mae_before - mae_after) / mae_before) * 100;

disp('===== ERROR METRICS =====')
disp(['MSE Before  : ', num2str(mse_before)])
disp(['MSE After   : ', num2str(mse_after)])
disp(['MAE Before  : ', num2str(mae_before)])
disp(['MAE After   : ', num2str(mae_after)])
disp(['Improvement : ', num2str(improvement)])

% =========================
% PLOT: BEFORE VS AFTER
% =========================
figure;
plot(1:length(Y), Y, '-o', 'LineWidth', 1.5); hold on;
plot(before, '-s', 'LineWidth', 1.5);
plot(after, '-d', 'LineWidth', 1.5);
xlabel('Test Case');
ylabel('Kidney Risk Score');
legend('Target','Before GA','After GA','Location','best');
title('Kidney Risk Prediction Before vs After GA');
grid on;

% =========================
% CONTROL SURFACES
% =========================
figure;
gensurf(optFis,[1 2],1)
title('Creatinine vs BP')

figure;
gensurf(optFis,[1 3],1)
title('Creatinine vs Age')

figure;
gensurf(optFis,[2 4],1)
title('BP vs Diabetes')

% =========================
% MEMBERSHIP FUNCTIONS
% =========================
figure;
plotmf(fis,'input',1);
title('Before: Creatinine MFs')

figure;
plotmf(optFis,'input',1);
title('After: Creatinine MFs')

figure;
plotmf(fis,'input',2);
title('Before: BP MFs')

figure;
plotmf(optFis,'input',2);
title('After: BP MFs')

figure;
plotmf(fis,'output',1);
title('Before: Risk MFs')

figure;
plotmf(optFis,'output',1);
title('After: Risk MFs')

% =========================
% RULE VIEWER
% =========================
ruleview(optFis)

% =========================
% SAVE
% =========================
writefis(optFis, 'kidney_optimized.fis');

disp('===== DONE =====')
disp('Optimised FIS saved as: kidney_optimized.fis')

% ==========================================================
% FITNESS FUNCTION
% ==========================================================
function err = localFitness(x, fis, X, Y)

    f = fis;

    % Creatinine
    f.Inputs(1).MembershipFunctions(2).Parameters = [x(1)-0.8 x(1) x(1)+0.8];
    f.Inputs(1).MembershipFunctions(3).Parameters = [x(2)-1.0 x(2) x(2)+1.0];

    % BP
    f.Inputs(2).MembershipFunctions(2).Parameters = [x(3)-20 x(3) x(3)+20];
    f.Inputs(2).MembershipFunctions(3).Parameters = [x(4)-20 x(4) x(4)+20];

    % Output
    f.Outputs(1).MembershipFunctions(2).Parameters = [x(5)-15 x(5) x(5)+15];
    f.Outputs(1).MembershipFunctions(3).Parameters = [x(6)-15 x(6) x(6)+15];

    % Predictions
    n = size(X,1);
    pred = zeros(n,1);

    for k = 1:n
        pred(k) = evalfis(f, X(k,:));
    end

    % MSE
    err = mean((pred - Y).^2);
end

% ==========================================================
% GA OUTPUT FUNCTION
% ==========================================================
function [state, options, optchanged] = gaoutfun(options, state, flag)

    global ga_history
    optchanged = false;

    if strcmp(flag, 'iter')
        ga_history(end+1) = state.Best(end);
    end
end
