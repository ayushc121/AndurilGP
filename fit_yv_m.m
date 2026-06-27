function [Yv_m, Yvv_m] = fit_yv_m(csv_paths, meta_paths, varargin)
% fit_yv_m   Identify Yv_m and Yvv_m from lateral trim-sweep test(s).
%
% Usage (single rep):
%   [Yv_m, Yvv_m] = fit_yv_m('sysid_P0_lateral_1.csv', ...
%                              'sysid_P0_lateral_1_meta.json')
%
% Usage (pool multiple reps):
%   [Yv_m, Yvv_m] = fit_yv_m( ...
%       {'sysid_P0_lateral_1.csv','sysid_P0_lateral_2.csv'}, ...
%       {'sysid_P0_lateral_1_meta.json','sysid_P0_lateral_2_meta.json'})
%
% IDENTIFICATION APPROACH
% =======================
% The test banks the drone through alternating angles (+10°, -10°, +18°, ...),
% each held for LAT_STEP_DUR seconds.  After a brief phi-transition (SKIP_DUR),
% the FULL remaining trajectory is used — NOT just the quasi-steady tail.
%
% Why the full trajectory is essential:
%   Yv_m and Yvv_m are separated by the VARIATION in |vy_b| across the window.
%   Using only the quasi-steady tail gives |vy_b| in a narrow range (5–9 m/s)
%   where vy_b and vy_b·|vy_b| are nearly collinear → Yv_m collapses to zero.
%   Including the transient (where vy_b passes through 0 during sign reversals)
%   widens |vy_b| from 0 to 9 m/s, making the two columns linearly independent.
%
% The regression identity  z_v = Yv_m·v + Yvv_m·v·|v|  holds at ALL times,
% not just steady state, so the transient data is equally valid.
%
% SIGN CONVENTION NOTE
% ====================
% vy_b in this simulator is measured with OPPOSITE sign to the NED body-y axis
% (positive = leftward).  The correct simulator lateral dynamics are:
%   d(vy_b)/dt = -(p·vz_b - r·vx_b) - g·cos(θ)·sin(φ) + Yv_m·vy_b + Yvv_m·vy_b·|vy_b|
% Both Yv_m and Yvv_m remain negative (dissipative drag).
%
% After identifying Yv_m and Yvv_m, update:
%   analyze_segment.m  —  PRIOR_Yv_m, PRIOR_Yvv_m
%   drone_sysid_main.m —  HARD_PRIOR(3), HARD_PRIOR(4)
% Then delete all *_result.mat files and re-run drone_sysid_main.m.

pr = inputParser;
addParameter(pr, 'skip_dur', 2.0, @isnumeric);  % skip first N s (phi transition)
addParameter(pr, 'plot',     true, @islogical);
parse(pr, varargin{:});
SKIP_DUR = pr.Results.skip_dur;
DO_PLOT  = pr.Results.plot;

g = 9.81;

if ischar(csv_paths);  csv_paths  = {csv_paths};  end
if ischar(meta_paths); meta_paths = {meta_paths}; end

% Accumulate data from all reps
z_all   = [];   % residual: Yv_m·vy_b + Yvv_m·vy_b·|vy_b|
Phi_all = [];   % [vy_b,  vy_b·|vy_b|]

% For diagnostic: per-step quasi-SS means (last 20% of each step)
all_vy_ss   = [];
all_phi_ss  = [];

for kk = 1:length(csv_paths)

    % ── Load metadata ─────────────────────────────────────────────────────
    fid  = fopen(meta_paths{kk},'r');
    meta = jsondecode(fread(fid, inf, 'uint8=>char')');
    fclose(fid);

    if ~isfield(meta,'lat_phi_sequence_deg') || ~isfield(meta,'lat_step_dur')
        error('[Rep %d] Metadata missing lat_phi_sequence_deg / lat_step_dur.\n Re-run the lateral test with the updated sysid_excitation.py.', kk);
    end

    phi_seq_deg = meta.lat_phi_sequence_deg;
    phi_seq     = deg2rad(phi_seq_deg(:)');   % row vector
    step_dur    = meta.lat_step_dur;
    n_steps     = length(phi_seq);

    % ── Load CSV ──────────────────────────────────────────────────────────
    opts                    = detectImportOptions(csv_paths{kk});
    opts.DataLines          = [2 Inf];
    opts.VariableNamesLine  = 1;
    T                       = readtable(csv_paths{kk}, opts);

    exc_mask = strcmp(T.phase_tag, 'excitation');
    if sum(exc_mask) < 20
        warning('[Rep %d] Fewer than 20 excitation rows — skipping.', kk);
        continue;
    end

    time  = T.t(exc_mask);   time = time - time(1);
    phi_e = T.phi(exc_mask);
    the_e = T.theta(exc_mask);
    vy_b  = T.vy_b(exc_mask);
    vx_b  = T.vx_b(exc_mask);
    vz_b  = T.vz_b(exc_mask);
    p_e   = T.p(exc_mask);
    r_e   = T.r(exc_mask);

    dt   = mean(diff(time));
    vdot = deriv(vy_b, dt);

    fprintf('\n[Rep %d] %s\n', kk, csv_paths{kk});
    fprintf('  %d steps × %.1f s  |  dt=%.4f s  |  skip_dur=%.1f s\n', ...
            n_steps, step_dur, dt, SKIP_DUR);

    for s = 1:n_steps
        t0_step = (s-1) * step_dur;
        t1_step =  s    * step_dur;

        % Skip SKIP_DUR seconds at the start of each step (phi transition).
        % Use ALL remaining data — critical for capturing the vy_b ≈ 0 crossing
        % during sign-reversal steps (steps 2, 4, 6).
        t_data_start = t0_step + SKIP_DUR;
        mask_full    = time >= t_data_start & time < t1_step;

        % Quasi-SS window (last 20%) — for diagnostic mean only, NOT for fitting
        t_ss_start   = t0_step + 0.80 * step_dur;
        mask_ss      = time >= t_ss_start & time < t1_step;

        if sum(mask_full) < 5
            fprintf('  Step %d (%+.0f°): only %d usable points — skipped\n', ...
                    s, rad2deg(phi_seq(s)), sum(mask_full));
            continue;
        end

        % Residual isolating drag terms (see sign convention note in header)
        coriolis  = p_e(mask_full).*vz_b(mask_full) - r_e(mask_full).*vx_b(mask_full);
        gravity   = g .* cos(the_e(mask_full)) .* sin(phi_e(mask_full));
        z_full    = vdot(mask_full) + coriolis + gravity;

        vy_full   = vy_b(mask_full);
        Phi_full  = [vy_full,  vy_full.*abs(vy_full)];

        % Quasi-SS mean for reporting / right-plot diagnostic
        if sum(mask_ss) >= 3
            vy_ss_mean  = mean(vy_b(mask_ss));
            phi_ss_mean = mean(phi_e(mask_ss));
        else
            vy_ss_mean  = mean(vy_full);
            phi_ss_mean = mean(phi_e(mask_full));
        end

        fprintf('  Step %d: cmd=%+.0f°  phi_ss=%+.1f°  vy_b_ss=%+.3f m/s  N=%d (full), %d (SS)\n', ...
                s, rad2deg(phi_seq(s)), rad2deg(phi_ss_mean), vy_ss_mean, ...
                sum(mask_full), sum(mask_ss));

        z_all   = [z_all;   z_full];    %#ok<AGROW>
        Phi_all = [Phi_all; Phi_full];  %#ok<AGROW>

        all_vy_ss(end+1)  = vy_ss_mean;   %#ok<AGROW>
        all_phi_ss(end+1) = phi_ss_mean;  %#ok<AGROW>
    end
end

if isempty(z_all)
    error('No usable data found across all reps.');
end

fprintf('\nTotal data points for fit: %d  (vy_b range: [%.2f, %.2f] m/s)\n', ...
        length(z_all), min(Phi_all(:,1)), max(Phi_all(:,1)));

% =============================================================================
% Equation-error fit:  min ||Phi·p − z||²   s.t.  Yv_m < 0,  Yvv_m < 0
% =============================================================================
lsq_opts = optimoptions('lsqlin', 'Display', 'off');
lb = [-Inf; -Inf];
ub = [-1e-4; -1e-4];
p_hat = lsqlin(Phi_all, z_all, [], [], [], [], lb, ub, [], lsq_opts);
Yv_m  = p_hat(1);
Yvv_m = p_hat(2);

z_fit  = Phi_all * p_hat;
ss_res = sum((z_all - z_fit).^2);
ss_tot = sum((z_all - mean(z_all)).^2);
R2     = 1 - ss_res / max(ss_tot, 1e-12);

% =============================================================================
% Check collinearity — warn if Yv_m is at the upper bound
% =============================================================================
Yv_m_at_bound = abs(Yv_m - ub(1)) < 1e-5;
vy_b_range    = max(Phi_all(:,1)) - min(Phi_all(:,1));

fprintf('\n============================================================\n');
fprintf('  IDENTIFIED LATERAL DRAG PARAMETERS\n');
fprintf('============================================================\n');
fprintf('  Yv_m  = %8.5f  (linear drag coefficient)\n', Yv_m);
fprintf('  Yvv_m = %8.5f  (quadratic drag coefficient)\n', Yvv_m);
fprintf('  R²    = %8.4f  [>0.90 = good fit]\n', R2);
fprintf('  vy_b data range: %.2f to %.2f m/s\n', min(Phi_all(:,1)), max(Phi_all(:,1)));

if Yv_m_at_bound
    fprintf('\n  [WARNING] Yv_m hit the upper bound — collinearity issue.\n');
    fprintf('  The vy_b range (%.1f m/s) is not wide enough to separate\n', vy_b_range);
    fprintf('  Yv_m from Yvv_m.  SKIP_DUR may still be too large.\n');
    fprintf('  Try: fit_yv_m(..., ''skip_dur'', 1.0)\n');
    fprintf('  OR:  re-run test with LAT_STEP_DUR >= 12s to get wider vy_b range.\n');
else
    fprintf('\n  Effective damping at 5 m/s:  %.4f  (Yv_m + Yvv_m*5 = %.4f)\n', ...
            abs(Yv_m)+abs(Yvv_m)*5, Yv_m+Yvv_m*5);
    fprintf('  Effective damping at 10 m/s: %.4f  (Yv_m + Yvv_m*10 = %.4f)\n', ...
            abs(Yv_m)+abs(Yvv_m)*10, Yv_m+Yvv_m*10);
    fprintf('  Quadratic share at 5 m/s:  %.0f%%\n', 100*abs(Yvv_m)*5/(abs(Yv_m)+abs(Yvv_m)*5));
    fprintf('  Quadratic share at 10 m/s: %.0f%%\n', 100*abs(Yvv_m)*10/(abs(Yv_m)+abs(Yvv_m)*10));
end

fprintf('\n  UPDATE analyze_segment.m priors:\n');
fprintf('    PRIOR_Yv_m   = %.5f;\n', Yv_m);
fprintf('    PRIOR_Yvv_m  = %.5f;\n', Yvv_m);
fprintf('\n  UPDATE drone_sysid_main.m HARD_PRIOR:\n');
fprintf('    HARD_PRIOR(3) = %.5f;  %% Yv_m\n', Yv_m);
fprintf('    HARD_PRIOR(4) = %.5f;  %% Yvv_m\n', Yvv_m);
fprintf('============================================================\n\n');

% =============================================================================
% Diagnostic plots
% =============================================================================
if ~DO_PLOT; return; end

figure('Name','Lateral drag identification','Position',[50 50 1300 520]);

% ── Left: raw regression scatter ─────────────────────────────────────────────
subplot(1,2,1);
vy_test = linspace(min(Phi_all(:,1))-1, max(Phi_all(:,1))+1, 400);
scatter(Phi_all(:,1), z_all, 3, 'k', 'filled', 'MarkerFaceAlpha', 0.10);
hold on;
plot(vy_test, Yv_m*vy_test + Yvv_m*vy_test.*abs(vy_test), 'r-', 'LineWidth', 2.5);
scatter(all_vy_ss, Yv_m*all_vy_ss + Yvv_m*all_vy_ss.*abs(all_vy_ss), ...
        70, [0 0.45 0.74], 'filled', 'MarkerEdgeColor','k');
hold off;
xlabel('v_b  (m/s)');
ylabel('z_v = \dot{v} + Coriolis + gravity  (m/s²)');
title(sprintf('Drag fit:  Yv\\_m=%.4f  Yvv\\_m=%.5f  R²=%.4f', Yv_m, Yvv_m, R2));
legend('All transient data','Fit curve','Quasi-SS means','Location','best');
grid on;

% ── Right: steady-state |vy_b| vs bank angle ─────────────────────────────────
subplot(1,2,2);
phi_deg_plot = linspace(0, 32, 200);

% Solve  |Yv_m|·v + |Yvv_m|·v² = g·sin(phi)  analytically (positive root).
% Avoids fzero and the starting-point overflow issue when Yv_m ≈ 0.
a_coef = abs(Yvv_m);
b_coef = abs(Yv_m);
rhs    = g .* sind(phi_deg_plot);

if a_coef > 1e-6
    % Full quadratic: v = (-b + sqrt(b² + 4a·rhs)) / (2a)
    vy_ss_full = (-b_coef + sqrt(max(b_coef^2 + 4*a_coef.*rhs, 0))) / (2*a_coef);
else
    % Essentially linear
    vy_ss_full = rhs / max(b_coef, 1e-6);
end

% Linear-only prediction (cap at 20 m/s for display)
if abs(Yv_m) > 0.05
    vy_ss_lin = min(rhs / abs(Yv_m), 20);
    plot(phi_deg_plot, vy_ss_lin, 'b--', 'LineWidth', 1.5, 'DisplayName', ...
         sprintf('Linear only (Yv\\_m=%.3f)', Yv_m));
    hold on;
end
plot(phi_deg_plot, vy_ss_full, 'r-', 'LineWidth', 2.5, 'DisplayName', 'Full model');
hold on;
scatter(rad2deg(abs(all_phi_ss)), abs(all_vy_ss), 70, 'k', 'filled', ...
        'DisplayName', 'Quasi-SS step means');
hold off;
ylim([0, max(abs(all_vy_ss))*1.4 + 1]);
xlabel('\phi_{trim}  (deg)');
ylabel('|v_b|  quasi-steady  (m/s)');
title('Steady-state lateral speed vs bank angle');
legend('Location','best');
grid on;

sgtitle('Lateral drag identification  (trim sweep)');