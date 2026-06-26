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
% The test commands the drone through a sequence of held bank angles
% (e.g. +10°, -10°, +18°, -18°, +26°, -26°, each for 8 s).  After the
% initial transient (≈2 time-constants), vy_b approaches the steady-state
% value where  g·cos(θ)·sin(φ) + Yv_m·v + Yvv_m·v·|v| ≈ 0.
%
% This script fits those parameters using equation-error regression on the
% quasi-steady windows (last STEADY_FRAC of each step), which avoids the
% altitude-hold coupling that plagued the OE dynamic excitation approach.
%
% After identifying Yv_m and Yvv_m, update these values in:
%   analyze_segment.m  lines:  PRIOR_Yv_m, PRIOR_Yvv_m
%   drone_sysid_main.m lines:  HARD_PRIOR(3), HARD_PRIOR(4)
% Then delete all *_result.mat files and re-run drone_sysid_main.m.

pr = inputParser;
addParameter(pr, 'steady_frac', 0.40, @isnumeric);  % last 40% of each step
addParameter(pr, 'plot', true, @islogical);
parse(pr, varargin{:});
STEADY_FRAC = pr.Results.steady_frac;
DO_PLOT     = pr.Results.plot;

g = 9.81;

if ischar(csv_paths);  csv_paths  = {csv_paths};  end
if ischar(meta_paths); meta_paths = {meta_paths}; end

% =============================================================================
% Accumulate quasi-steady data from all reps
% =============================================================================
z_all   = [];   % vdot − (p·vz_b − r·vx_b) − g·cos(θ)·sin(φ)
Phi_all = [];   % [vy_b,  vy_b·|vy_b|]

% For diagnostic plot across reps
all_vy_step_mean  = [];
all_phi_step_mean = [];

for kk = 1:length(csv_paths)

    % ── Load metadata ─────────────────────────────────────────────────────
    fid  = fopen(meta_paths{kk},'r');
    meta = jsondecode(fread(fid, inf, 'uint8=>char')');
    fclose(fid);

    if ~isfield(meta,'lat_phi_sequence_deg') || ~isfield(meta,'lat_step_dur')
        error('[Rep %d] Metadata missing lat_phi_sequence_deg / lat_step_dur.\n' ...
              'Re-run the lateral test with the updated sysid_excitation.py.', kk);
    end

    phi_seq_deg = meta.lat_phi_sequence_deg;   % [1 x n_steps] or [n_steps x 1]
    phi_seq     = deg2rad(phi_seq_deg(:)');     % enforce row vector
    step_dur    = meta.lat_step_dur;
    n_steps     = length(phi_seq);

    % ── Load CSV ──────────────────────────────────────────────────────────
    opts               = detectImportOptions(csv_paths{kk});
    opts.DataLines     = [2 Inf];
    opts.VariableNamesLine = 1;
    T                  = readtable(csv_paths{kk}, opts);

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
    fprintf('  %d steps × %.1f s  |  dt=%.4f s  |  steady_frac=%.0f%%\n', ...
            n_steps, step_dur, dt, STEADY_FRAC*100);

    % ── Extract quasi-steady window from each step ─────────────────────────
    for s = 1:n_steps
        t0_step  = (s-1) * step_dur;
        t1_step  =  s    * step_dur;
        t_ss_start = t0_step + (1-STEADY_FRAC) * step_dur;

        mask = time >= t_ss_start & time < t1_step;
        if sum(mask) < 5
            fprintf('  Step %d (%+.0f°): only %d points in steady window — skipped\n', ...
                    s, rad2deg(phi_seq(s)), sum(mask));
            continue;
        end

        % Residual after removing measured Coriolis and gravity forcing
        coriolis = p_e(mask).*vz_b(mask) - r_e(mask).*vx_b(mask);
        gravity  = g .* cos(the_e(mask)) .* sin(phi_e(mask));
        z_step   = vdot(mask) - coriolis - gravity;

        vy_step  = vy_b(mask);
        Phi_step = [vy_step,  vy_step.*abs(vy_step)];

        phi_mean = mean(phi_e(mask));
        vy_mean  = mean(vy_step);

        fprintf('  Step %d: cmd=%+.0f°  actual=%+.1f°  vy_b_ss=%+.3f m/s  N=%d\n', ...
                s, rad2deg(phi_seq(s)), rad2deg(phi_mean), vy_mean, sum(mask));

        z_all   = [z_all;   z_step];    %#ok<AGROW>
        Phi_all = [Phi_all; Phi_step];  %#ok<AGROW>

        all_vy_step_mean(end+1)  = vy_mean;   %#ok<AGROW>
        all_phi_step_mean(end+1) = phi_mean;  %#ok<AGROW>
    end
end

if isempty(z_all)
    error('No usable quasi-steady data found across all reps.');
end

fprintf('\nTotal data points for fit: %d\n', length(z_all));

% =============================================================================
% Equation-error fit: min ||Phi·p − z||²  s.t. Yv_m < 0, Yvv_m < 0
% =============================================================================
lsq_opts = optimoptions('lsqlin', 'Display', 'off');
lb = [-Inf; -Inf];
ub = [-1e-4; -1e-4];   % both strictly negative
p_hat = lsqlin(Phi_all, z_all, [], [], [], [], lb, ub, [], lsq_opts);
Yv_m  = p_hat(1);
Yvv_m = p_hat(2);

% Goodness of fit
z_fit  = Phi_all * p_hat;
ss_res = sum((z_all - z_fit).^2);
ss_tot = sum((z_all - mean(z_all)).^2);
R2     = 1 - ss_res / max(ss_tot, 1e-12);

% Implied steady-state velocities
vy_test = linspace(-15, 15, 300);
vy_ss_check = abs(g .* sin(all_phi_step_mean)) ./ (abs(Yv_m) + abs(Yvv_m).*abs(all_vy_step_mean));

fprintf('\n============================================================\n');
fprintf('  IDENTIFIED LATERAL DRAG PARAMETERS\n');
fprintf('============================================================\n');
fprintf('  Yv_m  = %8.5f  m/s²/(m/s)  [expect −0.2 to −0.5]\n', Yv_m);
fprintf('  Yvv_m = %8.5f  m/s²/(m/s)² [expect −0.01 to −0.06]\n', Yvv_m);
fprintf('  R²    = %8.4f              [>0.90 = good]\n', R2);
fprintf('\n  Implied time constant 1/|Yv_m| = %.2f s\n', 1/abs(Yv_m));
fprintf('  Quadratic contribution at 5 m/s: %.1f%%\n', ...
        100*abs(Yvv_m)*5/abs(Yv_m));
fprintf('  Quadratic contribution at 10 m/s: %.1f%%\n', ...
        100*abs(Yvv_m)*10/abs(Yv_m));
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
if DO_PLOT
    figure('Name','Lateral drag fit','Position',[50 50 1200 500]);

    % --- Left: drag curve fit ---
    subplot(1,2,1);
    scatter(Phi_all(:,1), z_all, 4, 'k', 'filled', 'MarkerFaceAlpha', 0.15);
    hold on;
    plot(vy_test, Yv_m*vy_test + Yvv_m*vy_test.*abs(vy_test), 'r-', 'LineWidth', 2);
    % Mark the per-step mean points
    scatter(all_vy_step_mean, Yv_m*all_vy_step_mean + Yvv_m*all_vy_step_mean.*abs(all_vy_step_mean), ...
            60, 'b', 'filled');
    hold off;
    xlabel('v_b  (m/s)');
    ylabel('Residual \dot{v}  (m/s²)');
    title(sprintf('Yv\\_m=%.4f  Yvv\\_m=%.5f  R²=%.3f', Yv_m, Yvv_m, R2));
    legend('All quasi-steady pts','Fit curve','Step means','Location','best');
    grid on;

    % --- Right: steady-state comparison ---
    subplot(1,2,2);
    phi_plot = linspace(0, 30, 200);
    % Solve for vy_ss at each phi numerically (implicit in Yvv_m term)
    vy_ss_lin = (g * sind(phi_plot)) / abs(Yv_m);   % linear-only prediction
    vy_ss_full = zeros(size(phi_plot));
    for ip = 1:length(phi_plot)
        f = @(v) abs(Yv_m)*v + abs(Yvv_m)*v^2 - g*sind(phi_plot(ip));
        if g*sind(phi_plot(ip)) < 0.01; continue; end
        vy_ss_full(ip) = fzero(f, vy_ss_lin(ip));
    end
    plot(phi_plot, vy_ss_lin,  'b--', 'LineWidth', 1.5, 'DisplayName', 'Linear only (Yvv=0)');
    hold on;
    plot(phi_plot, vy_ss_full, 'r-',  'LineWidth', 2,   'DisplayName', 'Full model');
    scatter(rad2deg(abs(all_phi_step_mean)), abs(all_vy_step_mean), ...
            60, 'k', 'filled', 'DisplayName', 'Step quasi-SS data');
    hold off;
    xlabel('\phi_{trim}  (deg)');
    ylabel('|v_b|  steady-state  (m/s)');
    title('Steady-state lateral velocity vs bank angle');
    legend('Location','best');
    grid on;

    sgtitle('Lateral drag identification (trim sweep)');
end

end
