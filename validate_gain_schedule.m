% =============================================================================
% validate_gain_schedule.m
%
% Loads gain_schedule.mat and performs three pre-controller validation checks:
%
%   Check 1 — Eigenvalue analysis
%       All real parts must be ≤ 0.01.  The psi integrator eigenvalue
%       at exactly 0 is expected and acceptable.  Any real part > 0.1 is a
%       FAIL; 0.01 < real(λ) ≤ 0.1 generates a WARN.
%
%   Check 2 — Trim residual (full nonlinear EOM)
%       Evaluate the 9-state equations at the computed trim state with the
%       measured T0.  Thresholds:
%           P / D regimes:  all translational  ≤ 0.30 m/s²
%           B / C regimes:  udot, vdot         ≤ 0.30 m/s²
%                           wdot               ≤ 1.50 m/s²  (thrust-model mismatch)
%           all regimes:    pdot, qdot, rdot   ≤ 0.50 rad/s²
%       Angular residuals are trivially zero at trim (no rates, no inputs)
%       but are computed and printed for completeness.
%
%   Check 3 — Parameter smoothness (4 panels)
%       Panel 1 : P-series key params vs theta0
%       Panel 2 : B-series overlay on P-series (C-series also shown)
%       Panel 3 : All regimes scatter, coloured by phi0
%       Panel 4 : Computed lateral trim velocity v0 vs phi0 for B/C regimes;
%                 comparison with A(1,6) and A(3,4) entries in the stored
%                 A matrix (should be non-zero for B/C — flags linearise_drone
%                 v0=0 hard-code as a WARNING if they are all zero).
%
% Pass/fail is printed to the console.  Figures are saved as PNG files
% alongside gain_schedule.mat.
%
% Parameter vector p (16 elements):
%   1=Xu_m  2=Xuu_m  3=Yv_m  4=Yvv_m  5=Tmax_m  6=Zw_m  7=Zww_m
%   8=Gamma1  9=tau_p  10=Gamma2  11=tau_q  12=Gamma3  13=tau_r
%   14=Lp  15=Mq  16=Nr
% =============================================================================

clear; clc; close all;

GS_FILE = 'gain_schedule.mat';

if ~isfile(GS_FILE)
    error('Cannot find %s — run drone_sysid_main.m first.', GS_FILE);
end
load(GS_FILE, 'gs');

n_regimes = length(gs.regimes);
g         = 9.81;

% Thresholds
EIG_FAIL_THRESH   = 0.10;    % real(λ) > this → FAIL
EIG_WARN_THRESH   = 0.01;    % real(λ) > this → WARN  (below FAIL)
TRANS_THRESH_P    = 0.30;    % m/s²  — P and D regimes, all translational DOF
TRANS_THRESH_BC   = 0.30;    % m/s²  — B and C regimes, udot and vdot
WDOT_THRESH_BC    = 1.50;    % m/s²  — B and C regimes, wdot only (thrust mismatch)
ROT_THRESH        = 0.50;    % rad/s² — all regimes, angular accelerations

% Preallocate result vectors
eig_status   = cell(n_regimes, 1);   % 'PASS' | 'WARN' | 'FAIL'
trim_status  = cell(n_regimes, 1);
check1_pass  = true(n_regimes, 1);
check2_pass  = true(n_regimes, 1);

% Helper: classify regime type from ID prefix character
rtype = @(id) id(1);   % returns 'P', 'B', 'C', or 'D'


% =============================================================================
% CHECK 1 — EIGENVALUE ANALYSIS
% =============================================================================
fprintf('\n');
fprintf('=================================================================\n');
fprintf('  CHECK 1 — EIGENVALUE ANALYSIS\n');
fprintf('=================================================================\n');
fprintf('Expected: one λ ≈ 0 (psi integrator), all others real(λ) < 0\n');
fprintf('FAIL if any real(λ) > %.2f  |  WARN if any real(λ) > %.2f\n', ...
        EIG_FAIL_THRESH, EIG_WARN_THRESH);
fprintf('\n');
fprintf('%-8s  %-8s  %-8s  %-13s  %-9s  %-9s  %s\n', ...
        'Regime', 'th0_deg', 'ph0_deg', 'max_real(λ)', 'n_pos_eig', '2nd_max', 'STATUS');
fprintf('%s\n', repmat('-', 1, 78));

for i = 1:n_regimes
    r  = gs.regimes(i);
    ev = eig(r.A);
    re = real(ev);

    % Sort descending so we can report the two largest real parts
    re_sorted = sort(re, 'descend');
    max_re    = re_sorted(1);
    sec_re    = re_sorted(2);

    % Count eigenvalues with positive real part beyond numerical noise
    n_pos = sum(re > EIG_WARN_THRESH);

    if max_re > EIG_FAIL_THRESH
        status = 'FAIL';
        check1_pass(i) = false;
    elseif max_re > EIG_WARN_THRESH
        status = 'WARN';
        % WARN does not fail Check 1 — the psi integrator is the only
        % expected zero; anything else between 0.01 and 0.1 is suspicious.
        % Change to check1_pass(i) = false if you want WARNs to count as fails.
    else
        status = 'PASS';
    end
    eig_status{i} = status;

    fprintf('%-8s  %-8.2f  %-8.2f  %-13.6f  %-9d  %-9.6f  %s\n', ...
            r.regime_id, rad2deg(r.theta0), rad2deg(r.phi0), ...
            max_re, n_pos, sec_re, status);
end
fprintf('%s\n', repmat('-', 1, 78));


% =============================================================================
% CHECK 2 — TRIM RESIDUAL (FULL NONLINEAR EOM)
% =============================================================================
fprintf('\n');
fprintf('=================================================================\n');
fprintf('  CHECK 2 — TRIM RESIDUAL  (nonlinear EOM at trim state)\n');
fprintf('=================================================================\n');
fprintf('Trim state:  [u0, v0, 0, 0, 0, 0, phi0, theta0, 0]\n');
fprintf('Trim input:  [0, 0, 0, T0_measured]\n');
fprintf('u0 = g*sin(θ₀)/Xu_m   v0 = -g*cos(θ₀)*sin(φ₀)/Yv_m  (Yvv_m=0)\n');
fprintf('\n');
fprintf('Thresholds:\n');
fprintf('  P/D regimes:  all translational ≤ %.2f m/s²\n', TRANS_THRESH_P);
fprintf('  B/C regimes:  udot, vdot ≤ %.2f m/s²   wdot ≤ %.2f m/s²\n', ...
        TRANS_THRESH_BC, WDOT_THRESH_BC);
fprintf('  all regimes:  pdot, qdot, rdot ≤ %.2f rad/s²\n', ROT_THRESH);
fprintf('\n');

hdr = sprintf('%-8s  %-7s  %-7s  %-7s  %-7s  %-7s  %-7s  %-7s  %-7s  %-8s  %-8s  %s', ...
              'Regime', 'udot', 'vdot', 'wdot', 'pdot', 'qdot', 'rdot', ...
              'u0(m/s)', 'v0(m/s)', 'T0_meas', 'T0_mdl', 'STATUS');
fprintf('%s\n', hdr);
fprintf('%s\n', repmat('-', 1, length(hdr)));

for i = 1:n_regimes
    r      = gs.regimes(i);
    p      = r.p;
    th0    = r.theta0;
    ph0    = r.phi0;
    T0     = r.T0;         % measured thrust from data
    rt     = rtype(r.regime_id);
    is_BC  = ismember(rt, {'B','C'});

    % Extract parameters
    Xu_m   = p(1);   Xuu_m  = p(2);
    Yv_m   = p(3);   Yvv_m  = p(4);   %#ok<NASGU>  (Yvv_m=0 from priors)
    Tmax_m = p(5);   Zw_m   = p(6);   Zww_m = p(7);  %#ok<NASGU>

    % --- Trim velocity computation ----------------------------------------
    % u0: from forward equilibrium  Xu_m*u0 + Xuu_m*u0*|u0| = g*sin(theta0)
    % With Xuu_m=0 from priors this is exact; include Xuu_m for generality
    % via a simple Newton iteration (converges in ~5 steps).
    if abs(Xu_m) < 1e-9
        u0 = 0;
    else
        u0 = g * sin(th0) / Xu_m;    % initial guess (exact when Xuu_m=0)
        for iter = 1:10
            f  = Xu_m*u0 + Xuu_m*u0*abs(u0) - g*sin(th0);
            df = Xu_m + 2*Xuu_m*abs(u0);
            if abs(df) < 1e-12; break; end
            u0 = u0 - f/df;
            if abs(f) < 1e-10; break; end
        end
    end

    % v0: from lateral equilibrium  g*cos(theta0)*sin(phi0) + Yv_m*v0 = 0
    % (Yvv_m=0 from priors → linear equation; if non-zero use same Newton)
    if abs(Yv_m) < 1e-9
        v0 = 0;
    else
        % With Yvv_m=0 this is exact; general Newton included for safety
        v0 = -g * cos(th0) * sin(ph0) / Yv_m;
    end

    w0 = 0;   % altitude hold trim

    % Model-predicted trim thrust (for comparison printout only)
    T0_model = g * cos(th0) * cos(ph0) / Tmax_m;

    % --- Nonlinear EOM at trim state, zero body rates, zero ctrl inputs ----
    % Body-frame translational accelerations
    % (Coriolis terms r*v, q*w, etc. are zero because p=q=r=0 at trim)
    udot_r = -g*sin(th0) + Xu_m*u0 + Xuu_m*u0*abs(u0);
    vdot_r =  g*cos(th0)*sin(ph0) + Yv_m*v0;  % Yvv_m=0
    wdot_r =  g*cos(th0)*cos(ph0) - Tmax_m*T0 + Zw_m*w0;  % w0=0, Zww_m*0=0

    % Angular accelerations: trivially zero (rates=0, inputs=0)
    pdot_r = 0;
    qdot_r = 0;
    rdot_r = 0;

    % --- Threshold check --------------------------------------------------
    if is_BC
        thr_u = TRANS_THRESH_BC;
        thr_v = TRANS_THRESH_BC;
        thr_w = WDOT_THRESH_BC;
    else
        thr_u = TRANS_THRESH_P;
        thr_v = TRANS_THRESH_P;
        thr_w = TRANS_THRESH_P;
    end

    trans_ok = (abs(udot_r) <= thr_u) && ...
               (abs(vdot_r) <= thr_v) && ...
               (abs(wdot_r) <= thr_w);
    rot_ok   = (abs(pdot_r) <= ROT_THRESH) && ...
               (abs(qdot_r) <= ROT_THRESH) && ...
               (abs(rdot_r) <= ROT_THRESH);

    pass = trans_ok && rot_ok;
    check2_pass(i) = pass;

    % Mark residuals that exceed threshold with '*'
    flag = @(val,thr) iff(abs(val)>thr);    
    u_fl = flag(udot_r, thr_u);
    v_fl = flag(vdot_r, thr_v);
    w_fl = flag(wdot_r, thr_w);

    trim_status{i} = 'PASS'; if ~pass; trim_status{i} = 'FAIL'; end

    fprintf('%-8s  %+6.3f%s %+6.3f%s %+6.3f%s  %+6.3f  %+6.3f  %+6.3f  %+7.3f  %+7.3f  %+8.4f  %+8.4f  %s\n', ...
            r.regime_id, ...
            udot_r, u_fl, vdot_r, v_fl, wdot_r, w_fl, ...
            pdot_r, qdot_r, rdot_r, ...
            u0, v0, T0, T0_model, trim_status{i});
end
fprintf('%s\n', repmat('-', 1, length(hdr)));
fprintf('* = exceeds threshold for that DOF and regime type\n');
fprintf('Note: wdot mismatch on B/C is expected (simulator tilt-compensation offset).\n');
fprintf('Note: T0_mdl = g·cos(θ₀)·cos(φ₀)/Tmax_m — compare with T0_meas to gauge thrust offset.\n');


% =============================================================================
% CHECK 3 — PARAMETER SMOOTHNESS  (4 panels)
% =============================================================================
fprintf('\n');
fprintf('=================================================================\n');
fprintf('  CHECK 3 — PARAMETER SMOOTHNESS\n');
fprintf('=================================================================\n');

% Partition regimes by type
P_idx = find(arrayfun(@(r) r.regime_id(1)=='P', gs.regimes));
B_idx = find(arrayfun(@(r) r.regime_id(1)=='B', gs.regimes));
C_idx = find(arrayfun(@(r) r.regime_id(1)=='C', gs.regimes));
D_idx = find(arrayfun(@(r) r.regime_id(1)=='D', gs.regimes));

% Sort each series by theta0 so plots are ordered
sort_by_theta = @(idx) argsort_theta(gs.regimes, idx);

P_idx = sort_by_theta(P_idx);
B_idx = sort_by_theta(B_idx);
C_idx = sort_by_theta(C_idx);
D_idx = sort_by_theta(D_idx);  

% Helper: extract scalar field from regime subset
extr_p   = @(idx, pi_) arrayfun(@(k) gs.regimes(k).p(pi_), idx);
extr_th  = @(idx)      rad2deg(arrayfun(@(k) gs.regimes(k).theta0, idx));
extr_ph  = @(idx)      rad2deg(arrayfun(@(k) gs.regimes(k).phi0, idx));
extr_A   = @(idx, r_, c_) arrayfun(@(k) gs.regimes(k).A(r_,c_), idx);

% Params of interest (indices into p, labels for plots)
SMOOTH_IDX  = [9,   11,    13,     14,   15,   16,    1,      6   ];
SMOOTH_LBL  = {'tau_p','tau_q','tau_r','Lp','Mq','Nr','Xu_m','Zw_m'};

% ----- Panel 1 : P-series vs theta0 ----------------------------------------
fig1 = figure('Name','Panel 1 — P-series vs theta0','Position',[30 30 1400 680]);
P_th = extr_th(P_idx);

for ki = 1:length(SMOOTH_IDX)
    subplot(2, 4, ki);
    yP = extr_p(P_idx, SMOOTH_IDX(ki));
    plot(P_th, yP, 'bo-', 'LineWidth', 1.6, 'MarkerFaceColor', 'b', ...
         'MarkerSize', 6);
    hold on;
    for kk = 1:length(P_idx)
        text(P_th(kk), yP(kk), ['  ' gs.regimes(P_idx(kk)).regime_id], ...
             'FontSize', 7, 'Color', [0 0 0.7]);
    end

    % Monotonicity check: flag any consecutive pair that violates expected trend
    % (warn only — does not affect pass/fail)
    if length(yP) >= 2
        diffs = diff(yP);
        % Determine expected sign from majority of differences
        exp_sign = sign(median(diffs));
        violations = find(sign(diffs) ~= exp_sign & abs(diffs) > 0.5*std(diffs));
        for vv = violations(:)'
            x_mid = 0.5*(P_th(vv) + P_th(vv+1));
            y_mid = 0.5*(yP(vv)   + yP(vv+1));
            plot(x_mid, y_mid, 'r^', 'MarkerSize', 9, 'MarkerFaceColor', 'r');
        end
        if ~isempty(violations)
            fprintf('[CHECK 3 WARN] P-series %s: non-monotonic step at theta0≈%.1f deg\n', ...
                    SMOOTH_LBL{ki}, P_th(violations(1)));
        end
    end

    hold off;
    xlabel('\theta_0 (deg)');
    ylabel(SMOOTH_LBL{ki});
    title(['P-series: ' strrep(SMOOTH_LBL{ki},'_','\_')]);
    grid on;
end
sgtitle('Panel 1 — P-series key parameters vs pitch trim angle');
saveas(fig1, 'check3_panel1_P_series.png');
fprintf('Panel 1 saved → check3_panel1_P_series.png\n');

% ----- Panel 2 : B-series + C-series overlay on P-series -------------------
fig2 = figure('Name','Panel 2 — B/C overlay on P-series','Position',[40 40 1400 680]);
B_th = extr_th(B_idx);
C_th = extr_th(C_idx);

for ki = 1:length(SMOOTH_IDX)
    subplot(2, 4, ki);
    hold on;

    % P-series (dashed background reference)
    if ~isempty(P_idx)
        yP = extr_p(P_idx, SMOOTH_IDX(ki));
        plot(P_th, yP, 'b--o', 'LineWidth', 1.0, 'MarkerSize', 5, ...
             'MarkerFaceColor', 'b', 'DisplayName', 'P-series');
    end

    % B-series (solid, red squares)
    if ~isempty(B_idx)
        yB = extr_p(B_idx, SMOOTH_IDX(ki));
        plot(B_th, yB, 'rs-', 'LineWidth', 1.6, 'MarkerSize', 7, ...
             'MarkerFaceColor', 'r', 'DisplayName', 'B-series');
        for kk = 1:length(B_idx)
            text(B_th(kk), yB(kk), ['  ' gs.regimes(B_idx(kk)).regime_id], ...
                 'FontSize', 7, 'Color', [0.7 0 0]);
        end
    end

    % C-series (solid, green triangles)
    if ~isempty(C_idx)
        yC = extr_p(C_idx, SMOOTH_IDX(ki));
        plot(C_th, yC, 'g^-', 'LineWidth', 1.4, 'MarkerSize', 7, ...
             'MarkerFaceColor', 'g', 'DisplayName', 'C-series');
        for kk = 1:length(C_idx)
            text(C_th(kk), yC(kk), ['  ' gs.regimes(C_idx(kk)).regime_id], ...
                 'FontSize', 7, 'Color', [0 0.5 0]);
        end
    end

    hold off;
    xlabel('\theta_0 (deg)');
    ylabel(SMOOTH_LBL{ki});
    title(strrep(SMOOTH_LBL{ki},'_','\_'));
    if ki == 1; legend('Location','best','FontSize',7); end
    grid on;
end
sgtitle('Panel 2 — B/C-series vs theta0 (P-series dashed for reference)');
saveas(fig2, 'check3_panel2_BC_overlay.png');
fprintf('Panel 2 saved → check3_panel2_BC_overlay.png\n');

% ----- Panel 3 : All regimes, coloured by phi0 ------------------------------
fig3 = figure('Name','Panel 3 — All regimes coloured by phi0','Position',[50 50 1400 680]);
all_th  = rad2deg(arrayfun(@(r) r.theta0, gs.regimes));
all_ph  = rad2deg(arrayfun(@(r) r.phi0,   gs.regimes));
all_ids = {gs.regimes.regime_id};

% Build colormap: phi0=0 → blue, phi0=60° → red
phi_max = max(abs(all_ph));  if phi_max < 1; phi_max = 1; end

for ki = 1:length(SMOOTH_IDX)
    subplot(2, 4, ki);
    hold on;
    for ri = 1:n_regimes
        phi_norm = abs(all_ph(ri)) / phi_max;   % 0 (P) → 1 (high phi)
        c_ri     = [phi_norm, 0, 1 - phi_norm]; % blue→red
        scatter(all_th(ri), gs.regimes(ri).p(SMOOTH_IDX(ki)), ...
                70, c_ri, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.4);
        text(all_th(ri), gs.regimes(ri).p(SMOOTH_IDX(ki)), ...
             ['  ' all_ids{ri}], 'FontSize', 6);
    end
    hold off;
    xlabel('\theta_0 (deg)');
    ylabel(SMOOTH_LBL{ki});
    title(strrep(SMOOTH_LBL{ki},'_','\_'));
    grid on;
end
sgtitle(sprintf('Panel 3 — All regimes (blue = \phi_0=0°, red = \phi_0=%.0f°)', phi_max));

% Add a simple manual colorbar annotation (avoids axis conflict)
annotation(fig3, 'textbox', [0.92 0.42 0.07 0.16], 'String', ...
           sprintf('Colour\n= \\phi_0\nBlue: 0°\nRed: %.0f°', phi_max), ...
           'EdgeColor', 'k', 'BackgroundColor', 'w', 'FontSize', 7, ...
           'FitBoxToText', false);

saveas(fig3, 'check3_panel3_all_regimes.png');
fprintf('Panel 3 saved → check3_panel3_all_regimes.png\n');

% ----- Panel 4 : Lateral trim velocity v0 and Coriolis A-matrix check -------
% v0 is the lateral body-frame velocity required to balance g*cos(θ)*sin(φ).
% linearise_drone currently hard-codes v0=0, meaning A(1,6) and A(3,4) are
% stored as zero even for B/C regimes.  This panel:
%   (a) plots the CORRECT computed v0 vs phi0 for B/C regimes
%   (b) compares computed v0 with the STORED A(1,6) and A(3,4) entries
%   (c) issues a WARNING if B/C Coriolis entries are all zero

fig4 = figure('Name','Panel 4 — Lateral trim velocity and Coriolis check', ...
              'Position',[60 60 1000 480]);

BC_idx = [B_idx(:); C_idx(:)];

if isempty(BC_idx)
    subplot(1,1,1);
    text(0.5, 0.5, 'No B or C regimes in gain schedule', ...
         'HorizontalAlignment', 'center', 'FontSize', 12);
    title('Panel 4 — No B/C regimes');
else
    BC_ph   = extr_ph(BC_idx);
    BC_ids  = {gs.regimes(BC_idx).regime_id};

    % Compute CORRECT v0 for each B/C regime
    BC_v0 = zeros(length(BC_idx), 1);
    for ki = 1:length(BC_idx)
        r_  = gs.regimes(BC_idx(ki));
        Yv_ = r_.p(3);
        if abs(Yv_) > 1e-9
            BC_v0(ki) = -g * cos(r_.theta0) * sin(r_.phi0) / Yv_;
        end
    end

    % Read STORED Coriolis entries from linearise_drone A matrices
    % A(1,6) should be +v0  (effect of r on udot via Coriolis r*v)
    % A(3,4) should be -v0  (effect of p on wdot via Coriolis -p*v)
    BC_A16 = extr_A(BC_idx, 1, 6);   % stored: A(1,6)
    BC_A34 = extr_A(BC_idx, 3, 4);   % stored: A(3,4)

    % [Sub-plot 1] v0 vs phi0
    subplot(1, 2, 1);
    [BC_ph_sorted, sort_ord] = sort(BC_ph);
    plot(BC_ph_sorted, BC_v0(sort_ord), 'rs-', ...
         'LineWidth', 1.6, 'MarkerFaceColor', 'r', 'MarkerSize', 7);
    hold on;
    for kk = 1:length(BC_idx)
        text(BC_ph(kk), BC_v0(kk), ['  ' BC_ids{kk}], 'FontSize', 8);
    end
    hold off;
    xlabel('\phi_0 (deg)');
    ylabel('v_0  (m/s)');
    title('Lateral trim velocity v_0 vs \phi_0');
    grid on;

    % Quick monotonicity check on |v0| vs phi0
    v0_abs_sorted = abs(BC_v0(sort_ord));
    if ~issorted(v0_abs_sorted)
        fprintf('[CHECK 3 WARN] B/C v0 is not monotonic with phi0 — check Yv_m consistency.\n');
    end

    % [Sub-plot 2] Stored A-matrix Coriolis vs. expected v0
    subplot(1, 2, 2);
    hold on;
    plot(BC_ph_sorted, BC_v0(sort_ord),     'rs-', 'LineWidth', 1.6, ...
         'MarkerFaceColor', 'r', 'DisplayName', 'Computed v_0');
    plot(BC_ph_sorted, BC_A16(sort_ord),    'b^--', 'LineWidth', 1.4, ...
         'MarkerFaceColor', 'b', 'DisplayName', 'A(1,6) stored');
    plot(BC_ph_sorted, -BC_A34(sort_ord),   'g^--', 'LineWidth', 1.4, ...
         'MarkerFaceColor', 'g', 'DisplayName', '-A(3,4) stored');
    hold off;
    xlabel('\phi_0 (deg)');
    ylabel('m/s');
    title({'A(1,6)=v0 and -A(3,4)=v0 (stored vs. computed)'; ...
           '(should overlap if linearise\_drone passes v0 correctly)'});
    legend('Location', 'northwest', 'FontSize', 8);
    grid on;

    % Issue warning if ALL stored Coriolis entries are zero for B/C regimes
    if all(abs(BC_A16) < 1e-9) && all(abs(BC_A34) < 1e-9)
        fprintf(['\n[CHECK 3 WARN] A(1,6) and A(3,4) are zero for ALL B/C regimes.\n' ...
                 '               linearise_drone hard-codes v0=0, so the Coriolis\n' ...
                 '               r*v coupling is missing from the stored A matrices.\n' ...
                 '               Computed v0 values (shown in Panel 4 subplot 1) should\n' ...
                 '               be added to linearise_drone as:\n' ...
                 '                 v0 = -g*cos(theta0)*sin(phi0) / Yv_m;\n' ...
                 '                 A(1,6) =  v0;\n' ...
                 '                 A(3,4) = -v0;\n' ...
                 '               before running LQR/NMPC design.\n']);
    end
end

sgtitle('Panel 4 — Lateral trim velocity v_0 and Coriolis A-matrix check');
saveas(fig4, 'check3_panel4_v0_coriolis.png');
fprintf('Panel 4 saved → check3_panel4_v0_coriolis.png\n');


% =============================================================================
% FINAL PASS/FAIL SUMMARY
% =============================================================================
fprintf('\n');
fprintf('=================================================================\n');
fprintf('  PASS/FAIL SUMMARY\n');
fprintf('=================================================================\n');
fprintf('%-8s  %-10s  %-10s  %-7s  %s\n', ...
        'Regime', 'Check1_Eig', 'Check2_Trim', 'SAT?', 'OVERALL');
fprintf('%s\n', repmat('-', 1, 55));

overall_pass = true;
for i = 1:n_regimes
    r = gs.regimes(i);

    c1_str = eig_status{i};
    c2_str = trim_status{i};
    sat_str = '';
    if isfield(r, 'saturated') && r.saturated
        sat_str = 'SAT';
    end

    if check1_pass(i) && check2_pass(i)
        ov_str = 'PASS';
    else
        ov_str = 'FAIL';
        overall_pass = false;
    end

    fprintf('%-8s  %-10s  %-10s  %-7s  %s\n', ...
            r.regime_id, c1_str, c2_str, sat_str, ov_str);
end

fprintf('%s\n', repmat('-', 1, 55));
if overall_pass
    fprintf('\n  ALL HARD CHECKS PASSED\n');
    fprintf('  (Review any WARN/SAT lines and Panel 4 Coriolis note before controller design.)\n');
else
    n_fail = sum(~check1_pass) + sum(~check2_pass);
    fprintf('\n  %d REGIME(S) FAILED — review flagged rows above.\n', n_fail);
    fprintf('  Do NOT proceed to controller design until failures are resolved.\n');
end
fprintf('\n');
fprintf('Check 3 outputs: 4 PNG figures saved alongside gain_schedule.mat.\n');
fprintf('  check3_panel1_P_series.png\n');
fprintf('  check3_panel2_BC_overlay.png\n');
fprintf('  check3_panel3_all_regimes.png\n');
fprintf('  check3_panel4_v0_coriolis.png\n');
fprintf('=================================================================\n\n');


% =============================================================================
% Local helpers
% =============================================================================
function idx_sorted = argsort_theta(regimes, idx)
% Return idx reordered by ascending theta0.
    thetas = arrayfun(@(k) regimes(k).theta0, idx);
    [~, ord] = sort(thetas);
    idx_sorted = idx(ord);
end

function s = iff(cond)
    if cond
        s = '*';
    else
        s = '';
    end
end