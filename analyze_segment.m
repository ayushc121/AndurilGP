function result = analyze_segment(csv_path, meta_path, varargin)
% analyze_segment   Run equation-error + output-error sysid on one test segment.
%
% Usage:
%   result = analyze_segment('sysid_P2_pitch_1.csv', 'sysid_P2_pitch_1_meta.json')
%   result = analyze_segment(..., 'plot', true)
%   result = analyze_segment(..., 'save', true)
%
% OE is run with a MINIMAL parameter vector containing only the parameters
% that are directly identifiable from the excited axis. Fixed parameters are
% passed through the constants vector c and are not touched by the optimizer.
%
% Per-axis free parameters:
%   roll  : [Gamma1, tau_p]                     (2 params)
%   pitch : [Gamma2, tau_q, Xu_m, Tmax_m, Zw_m] (5 params)
%   yaw   : [Gamma3, tau_r]                     (2 params)
%   heave : [Tmax_m, Zw_m, Zww_m]              (3 params)
%
% The full 13-parameter vector is assembled at the end by merging OE results
% with priors for non-excited parameters.

pr = inputParser;
addParameter(pr, 'plot', false, @islogical);
addParameter(pr, 'save', true,  @islogical);
parse(pr, varargin{:});
DO_PLOT = pr.Results.plot;
DO_SAVE = pr.Results.save;

g = 9.81;

% Full parameter vector (15 elements):
%   1=Xu_m  2=Xuu_m  3=Yv_m  4=Yvv_m  5=Tmax_m  6=Zw_m  7=Zww_m
%   8=Gamma1  9=tau_p  10=Gamma2  11=tau_q  12=Gamma3  13=tau_r
%   14=Lp  15=Mq  16=Nr
param_names = {'Xu_m','Xuu_m','Yv_m','Yvv_m','Tmax_m','Zw_m','Zww_m', ...
               'Gamma1','tau_p','Gamma2','tau_q','Gamma3','tau_r', ...
               'Lp','Mq','Nr'};

% =============================================================================
% 1. Load metadata
% =============================================================================
fid  = fopen(meta_path, 'r');
raw_text = fread(fid, inf, 'uint8=>char')';
fclose(fid);
meta = jsondecode(raw_text);

regime_id   = meta.regime_id;
excite_axis = meta.excite_axis;
rep         = meta.repetition;
theta0_tgt  = deg2rad(meta.theta0_deg);
phi0_tgt    = deg2rad(meta.phi0_deg);
settle_dur  = meta.settle_dur;
hold_dur    = meta.hold_dur;

tag = sprintf('%s_%s_%d', regime_id, excite_axis, rep);
fprintf('\n=== analyze_segment: %s ===\n', tag);

% =============================================================================
% 2. Load CSV — numeric columns + phase_tag text column
% =============================================================================
raw = readmatrix(csv_path);

opts           = detectImportOptions(csv_path);
opts.DataLines = [2, Inf];
T_full         = readtable(csv_path, opts);
phase_col      = T_full{:, 'phase_tag'};

time_all       = raw(:,1);
vx_b_all       = raw(:,5);
vy_b_all       = raw(:,6);
vz_b_all       = raw(:,7);
phi_all        = raw(:,8);
theta_all      = raw(:,9);
psi_all        = raw(:,10);
p_all          = raw(:,11);
q_all          = raw(:,12);
r_all          = raw(:,13);
cmd_roll_all   = raw(:,17);
cmd_pitch_all  = raw(:,18);
cmd_yaw_all    = raw(:,19);
cmd_thrust_all = raw(:,20);

% =============================================================================
% 3. Trim verification from hold phase
% =============================================================================
hold_mask = strcmp(phase_col, 'hold');

if sum(hold_mask) < 5
    warning('[%s] Fewer than 5 hold rows — trim check skipped.', tag);
    theta0_actual = theta0_tgt;
    phi0_actual   = phi0_tgt;
    T0_actual     = 0.265;
    trim_ok       = false;
else
    theta0_actual = mean(theta_all(hold_mask));
    phi0_actual   = mean(phi_all(hold_mask));
    T0_actual     = mean(cmd_thrust_all(hold_mask));

    theta_err = abs(rad2deg(theta0_actual - theta0_tgt));
    phi_err   = abs(rad2deg(phi0_actual   - phi0_tgt));
    trim_ok   = (theta_err < 7.0) && (phi_err < 7.0);

    fprintf('Trim:  theta target=%.1f  actual=%.1f  err=%.1f deg\n', ...
            rad2deg(theta0_tgt), rad2deg(theta0_actual), theta_err);
    fprintf('       phi   target=%.1f  actual=%.1f  err=%.1f deg\n', ...
            rad2deg(phi0_tgt), rad2deg(phi0_actual), phi_err);
    fprintf('       T0 actual=%.4f   Trim OK: %d\n', T0_actual, trim_ok);
    if ~trim_ok
        warning('[%s] Trim not reached within 7 deg.', tag);
    end
end

% =============================================================================
% 4. Extract excitation window
% =============================================================================
exc_mask = strcmp(phase_col, 'excitation');
if sum(exc_mask) < 20
    error('[%s] Fewer than 20 excitation rows.', tag);
end

time  = time_all(exc_mask);       time = time - time(1);
vx_b  = vx_b_all(exc_mask);
vy_b  = vy_b_all(exc_mask);
vz_b  = vz_b_all(exc_mask);
phi_e = phi_all(exc_mask);
the_e = theta_all(exc_mask);
psi_e = psi_all(exc_mask);
p_e   = p_all(exc_mask);
q_e   = q_all(exc_mask);
r_e   = r_all(exc_mask);
cr    = cmd_roll_all(exc_mask);
cp_   = cmd_pitch_all(exc_mask);
cy    = cmd_yaw_all(exc_mask);
ct    = cmd_thrust_all(exc_mask);

N  = length(time);
dt = mean(diff(time));
fprintf('Excitation: N=%d  dt=%.4f s  dur=%.2f s\n', N, dt, time(end));

% =============================================================================
% 5. Numerical differentiation
% =============================================================================
udot = deriv(vx_b, dt);
vdot = deriv(vy_b, dt);
wdot = deriv(vz_b, dt);
pdot = deriv(p_e,  dt);
qdot = deriv(q_e,  dt);
rdot = deriv(r_e,  dt);

% =============================================================================
% 6. Equation-error (lesq / lsqlin) — identifies FREE parameters only
% =============================================================================
fprintf('\n--- Equation-error ---\n');
lsq_opts = optimoptions('lsqlin', 'Display', 'off');

% Priors for non-excited parameters (update these after P0 runs)
% =========================================================================
% PRIOR PARAMETER VALUES
% =========================================================================
% After running P0 (hover) identification, replace these values with your
% actual identified results. Use p_oe_free from each P0 test:
%
%   From P0_roll:    PRIOR_Lp, PRIOR_G1, PRIOR_tau_p
%   From P0_pitch:   PRIOR_Mq, PRIOR_G2, PRIOR_tau_q, PRIOR_Xu_m,
%                    PRIOR_Tmax_m, PRIOR_Zw_m
%   From P0_yaw:     PRIOR_Nr, PRIOR_G3, PRIOR_tau_r
%   From P0_heave:   PRIOR_Tmax_m, PRIOR_Zw_m, PRIOR_Zww_m
%                    (heave gives a cleaner Tmax_m — prefer this over pitch)
%
% These priors only affect:
%   (a) Non-excited parameters in each segment (held near prior by crb0)
%   (b) Fallback values in build_gain_schedule.m for regimes missing a test
% They do NOT affect identification quality for the excited axis.
% =========================================================================
PRIOR_Xu_m   = -0.30;  PRIOR_Xuu_m = 0.0;   % <-- update from P0_pitch
PRIOR_Yv_m   = -0.30;  PRIOR_Yvv_m = 0.0;   % <-- update from B1 roll tests
PRIOR_Tmax_m =  37.0;  PRIOR_Zw_m  = -1.0;  PRIOR_Zww_m = 0.0;  % <-- P0_heave
PRIOR_G1     =  0.0;   PRIOR_tau_p =  5.5;   % <-- update from P0_roll
PRIOR_G2     =  0.0;   PRIOR_tau_q = -0.8;   % <-- update from P0_pitch
PRIOR_G3     =  0.0;   PRIOR_tau_r =  3.8;   % <-- update from P0_yaw
PRIOR_Lp     = -2.0;   % <-- update from P0_roll  (was -22 in your test)
PRIOR_Mq     = -2.0;   % <-- update from P0_pitch
PRIOR_Nr     = -2.0;   % <-- update from P0_yaw

switch excite_axis

    % ── ROLL: free = [Lp, Gamma1, tau_p] ────────────────────────────────────
    case 'roll'
        % p-equation: pdot = Lp*p + Gamma1*q*r + tau_p*cmd_roll
        %
        % Lp < 0 is roll-rate aerodynamic damping — essential for the model
        % to represent p returning to zero after a step input. Without it,
        % lesq conflates damping with tau_p and produces a badly biased estimate.
        %
        % Constraint: Lp < 0 (damping must oppose rotation)
        z_p   = pdot;
        Phi_p = [p_e,  q_e.*r_e,  cr];
        lb_p  = [-Inf; -Inf; -Inf];
        ub_p  = [-1e-4; Inf; Inf];   % Lp strictly negative
        pp    = lsqlin(Phi_p, z_p, [], [], [], [], lb_p, ub_p, [], lsq_opts);
        Lp_hat     = pp(1);
        Gamma1_hat = pp(2);
        tau_p_hat  = pp(3);
        fprintf('p-eq:  Lp=%.5f  Gamma1=%.5f  tau_p=%.5f\n', ...
                Lp_hat, Gamma1_hat, tau_p_hat);

        p0_free    = [Lp_hat; Gamma1_hat; tau_p_hat];
        free_names = {'Lp','Gamma1','tau_p'};

    % ── PITCH: free = [Mq, Gamma2, tau_q, Xu_m, Tmax_m, Zw_m] ──────────────
    case 'pitch'
        % q-equation: qdot = Mq*q + Gamma2*p*r + tau_q*cmd_pitch
        % Mq < 0 is pitch-rate aerodynamic damping
        z_q   = qdot;
        Phi_q = [q_e,  p_e.*r_e,  cp_];
        lb_q  = [-Inf; -Inf; -Inf];
        ub_q  = [-1e-4; Inf; Inf];   % Mq strictly negative
        pq    = lsqlin(Phi_q, z_q, [], [], [], [], lb_q, ub_q, [], lsq_opts);
        Mq_hat     = pq(1);
        Gamma2_hat = pq(2);
        tau_q_hat  = pq(3);
        fprintf('q-eq:  Mq=%.5f  Gamma2=%.5f  tau_q=%.5f\n', ...
                Mq_hat, Gamma2_hat, tau_q_hat);

        % u-equation (Xu_m only — Xuu_m fixed 0)
        z_u   = udot - (r_e.*vy_b - q_e.*vz_b) + g.*sin(the_e);
        Phi_u = vx_b;
        Xu_m_hat = lsqlin(Phi_u, z_u, [], [], [], [], -Inf, -1e-6, [], lsq_opts);
        fprintf('u-eq:  Xu_m=%.5f\n', Xu_m_hat);

        % w-equation (Tmax_m, Zw_m — Zww_m fixed 0)
        z_w   = wdot - (q_e.*vx_b - p_e.*vy_b) - g.*cos(the_e).*cos(phi_e);
        Phi_w = [-ct,  vz_b];
        lb_w  = [1e-6; -Inf];   ub_w = [Inf; -1e-6];
        pw    = lsqlin(Phi_w, z_w, [], [], [], [], lb_w, ub_w, [], lsq_opts);
        Tmax_m_hat = pw(1);
        Zw_m_hat   = pw(2);
        fprintf('w-eq:  Tmax_m=%.5f  Zw_m=%.5f\n', Tmax_m_hat, Zw_m_hat);

        p0_free    = [Mq_hat; Gamma2_hat; tau_q_hat; Xu_m_hat; Tmax_m_hat; Zw_m_hat];
        free_names = {'Mq','Gamma2','tau_q','Xu_m','Tmax_m','Zw_m'};

    % ── YAW: free = [Nr, Gamma3, tau_r] ─────────────────────────────────────
    case 'yaw'
        % r-equation: rdot = Nr*r + Gamma3*p*q + tau_r*cmd_yaw
        % Nr < 0 is yaw-rate aerodynamic damping
        z_r   = rdot;
        Phi_r = [r_e,  p_e.*q_e,  cy];
        lb_r  = [-Inf; -Inf; -Inf];
        ub_r  = [-1e-4; Inf; Inf];   % Nr strictly negative
        pr    = lsqlin(Phi_r, z_r, [], [], [], [], lb_r, ub_r, [], lsq_opts);
        Nr_hat     = pr(1);
        Gamma3_hat = pr(2);
        tau_r_hat  = pr(3);
        fprintf('r-eq:  Nr=%.5f  Gamma3=%.5f  tau_r=%.5f\n', ...
                Nr_hat, Gamma3_hat, tau_r_hat);

        p0_free    = [Nr_hat; Gamma3_hat; tau_r_hat];
        free_names = {'Nr','Gamma3','tau_r'};

    % ── HEAVE / DROP: free = [Tmax_m, Zw_m, Zww_m] ──────────────────────
    % 'drop' is free-fall excitation — same parameters, different excitation.
    case {'heave','drop'}
        z_w   = wdot - (q_e.*vx_b - p_e.*vy_b) - g.*cos(the_e).*cos(phi_e);
        Phi_w = [-ct,  vz_b,  vz_b.*abs(vz_b)];
        lb_w  = [1e-6; -Inf; -Inf];   ub_w = [Inf; -1e-6; -1e-6];
        pw    = lsqlin(Phi_w, z_w, [], [], [], [], lb_w, ub_w, [], lsq_opts);
        Tmax_m_hat = pw(1);
        Zw_m_hat   = pw(2);
        Zww_m_hat  = pw(3);
        fprintf('w-eq:  Tmax_m=%.5f  Zw_m=%.5f  Zww_m=%.5f\n', ...
                Tmax_m_hat, Zw_m_hat, Zww_m_hat);

        p0_free    = [Tmax_m_hat; Zw_m_hat; Zww_m_hat];
        free_names = {'Tmax_m','Zw_m','Zww_m'};

    otherwise
        error('Unknown excite_axis: %s  (valid: roll|pitch|yaw|heave|drop)', excite_axis);
end

fprintf('\nlesq free params:\n');
for i = 1:length(p0_free)
    fprintf('  %-10s = %.5f\n', free_names{i}, p0_free(i));
end

% =============================================================================
% 7. Build OE inputs: measured states passed as exogenous columns of u
%
% u_oe columns depend on axis — see drone_statespace_reduced.m header.
% All measured time-series passed here are NOT integrated; they are used
% as known inputs inside the dynamics function.
% =============================================================================
switch excite_axis
    case 'roll'
        % States: [p_rate, phi]
        x0_oe  = [p_e(1); phi_e(1)];
        z_oe   = [p_e,  phi_e];
        u_oe   = [cr,  q_e,  r_e,  the_e];   % [N x 4]
        axis_code = 1;

    case 'pitch'
        % States: [q_rate, theta, vx_b, vz_b]
        x0_oe  = [q_e(1); the_e(1); vx_b(1); vz_b(1)];
        z_oe   = [q_e,  the_e,  vx_b,  vz_b];
        u_oe   = [cp_,  ct,  p_e,  r_e,  phi_e];   % [N x 5]
        axis_code = 2;

    case 'yaw'
        % States: [r_rate, psi]
        x0_oe  = [r_e(1); psi_e(1)];
        z_oe   = [r_e,  psi_e];
        u_oe   = [cy,  p_e,  q_e];   % [N x 3]
        axis_code = 3;

    case {'heave','drop'}
        % States: [vz_b]
        x0_oe  = vz_b(1);
        z_oe   = vz_b(:);            % [N x 1] column
        u_oe   = ct(:);              % [N x 1]
        axis_code = 4;
end

% =============================================================================
% 8. Output scaling
%
% Normalise each column of z_oe by its RMS so all output channels contribute
% equally to the OE cost regardless of amplitude or physical units.
% Without this, a large-amplitude channel (p in rad/s) dominates and OE
% ignores small-amplitude channels (phi in rad).
% =============================================================================
nz      = size(z_oe, 2);
z_scale = zeros(1, nz);
for kk = 1:nz
    rms_k = sqrt(mean(z_oe(:,kk).^2));
    if rms_k < 1e-6; rms_k = 1.0; end
    z_scale(kk) = rms_k;
end
z_oe_sc  = z_oe ./ z_scale;         % [N x nz] normalised measurements

% x0 stays in physical units — the scale factors are passed through c_oe
% so that drone_statespace_reduced can scale its OWN output to match z_oe_sc.
% This keeps the RK4 integration in physical units (numerically stable)
% while OE sees normalised residuals.
x0_oe_sc = x0_oe(:);               % physical units — dsname scales output

fprintf('Output RMS scales: ');
fprintf('%.4f ', z_scale);  fprintf('\n');

% c passed as row vector [g, axis_code, z_scale...] — z_scale appended so
% drone_statespace_reduced can normalise its output to match z_oe_sc.
c_oe = [g, axis_code, z_scale];

% =============================================================================
% 9. crb0 — tight priors anchored to lesq estimates
%
% Prior penalty:  (p - p0)' * inv(crb0) * (p - p0)
% We set sig = 2% of the lesq value so that a 20% parameter move costs
% ~100 units — comparable to a good data fit cost.
% Gamma terms (near zero, poorly excited) get wide priors.
% =============================================================================
switch excite_axis
    case 'roll'
        sig_free = [abs(p0_free(1)) * 0.05;       % Lp     — 5% (may be small)
                    1.0;                           % Gamma1 — wide, near zero
                    abs(p0_free(3)) * 0.02];       % tau_p  — 2% tight

    case 'pitch'
        sig_free = [abs(p0_free(1)) * 0.05;       % Mq     — 5%
                    1.0;                           % Gamma2 — wide
                    abs(p0_free(3)) * 0.02;       % tau_q  — 2%
                    abs(p0_free(4)) * 0.02;       % Xu_m   — 2%
                    abs(p0_free(5)) * 0.02;       % Tmax_m — 2%
                    abs(p0_free(6)) * 0.02];      % Zw_m   — 2%

    case 'yaw'
        sig_free = [abs(p0_free(1)) * 0.05;       % Nr     — 5%
                    1.0;                           % Gamma3 — wide
                    abs(p0_free(3)) * 0.02];       % tau_r  — 2%

    case {'heave','drop'}
        sig_free = [abs(p0_free(1)) * 0.02;      % Tmax_m
                    abs(p0_free(2)) * 0.02;       % Zw_m
                    max(abs(p0_free(3)), 0.1) * 0.10];  % Zww_m (may be ~0)
end
sig_free = max(sig_free, 1e-4);   % guard against exact-zero lesq params
crb0     = diag(sig_free.^2);

% del: 1% finite difference step (default 0.1% is too small for noisy data)
del_vec = 0.01 * ones(length(p0_free), 1);

% =============================================================================
% 10. Output-error optimisation on FREE parameters only
% =============================================================================
fprintf('\n--- Output-error (%d free params) ---\n', length(p0_free));
fprintf('Prior sigmas: ');  fprintf('%.5f ', sig_free);  fprintf('\n');

[y_oe_sc, p_oe_free, crb_oe_free, ~, cost_oe] = ...
    oe('drone_statespace_reduced', p0_free, u_oe, time, x0_oe_sc, ...
       c_oe, z_oe_sc, 1, crb0, del_vec);

% dsname outputs already scaled; unscale back to physical units
y_oe = y_oe_sc .* z_scale;

% ── Diagnostic: confirm column identity ──────────────────────────────────
fprintf('\n[DIAG] y_oe vs z_oe column RMS:\n');
for kk = 1:size(y_oe,2)
    fprintf('  col %d:  y_oe RMS=%.5f   z_oe RMS=%.5f\n', ...
            kk, sqrt(mean(y_oe(:,kk).^2)), sqrt(mean(z_oe(:,kk).^2)));
end
fprintf('[DIAG] z_scale = '); fprintf('%.5f ', z_scale); fprintf('\n');
fprintf('[DIAG] x0_oe = '); fprintf('%.5f ', x0_oe(:)'); fprintf('\n');
fprintf('[DIAG] y_oe(1,:) = '); fprintf('%.5f ', y_oe(1,:)); fprintf('\n');
fprintf('[DIAG] z_oe(1,:) = '); fprintf('%.5f ', z_oe(1,:)); fprintf('\n');
% ─────────────────────────────────────────────────────────────────────────

fprintf('\nFree parameter results:\n');
for i = 1:length(p0_free)
    fprintf('  %-10s  lesq=%9.5f  oe=%9.5f  delta=%+.5f  std=%.5f\n', ...
            free_names{i}, p0_free(i), p_oe_free(i), ...
            p_oe_free(i)-p0_free(i), sqrt(crb_oe_free(i,i)));
end
fprintf('OE cost: %.6f\n', cost_oe);

% =============================================================================
% 10. Assemble full 13-parameter vector
%     Free params come from OE; all others from priors.
%     Non-excited params are marked not well-identified.
% =============================================================================

% Start from priors (16 parameters)
p_full = [PRIOR_Xu_m; PRIOR_Xuu_m; PRIOR_Yv_m; PRIOR_Yvv_m;
          PRIOR_Tmax_m; PRIOR_Zw_m; PRIOR_Zww_m;
          PRIOR_G1; PRIOR_tau_p;
          PRIOR_G2; PRIOR_tau_q;
          PRIOR_G3; PRIOR_tau_r;
          PRIOR_Lp; PRIOR_Mq; PRIOR_Nr];

% Overwrite with OE results for free params
%   Index map: param_names order is
%   1=Xu_m 2=Xuu_m 3=Yv_m 4=Yvv_m 5=Tmax_m 6=Zw_m 7=Zww_m
%   8=Gamma1 9=tau_p 10=Gamma2 11=tau_q 12=Gamma3 13=tau_r
% Index map for p_full (16 elements):
%  1=Xu_m 2=Xuu_m 3=Yv_m 4=Yvv_m 5=Tmax_m 6=Zw_m 7=Zww_m
%  8=Gamma1 9=tau_p 10=Gamma2 11=tau_q 12=Gamma3 13=tau_r
%  14=Lp 15=Mq 16=Nr
switch excite_axis
    case 'roll'
        p_full(14) = p_oe_free(1);   % Lp
        p_full(8)  = p_oe_free(2);   % Gamma1
        p_full(9)  = p_oe_free(3);   % tau_p
    case 'pitch'
        p_full(15) = p_oe_free(1);   % Mq
        p_full(10) = p_oe_free(2);   % Gamma2
        p_full(11) = p_oe_free(3);   % tau_q
        p_full(1)  = p_oe_free(4);   % Xu_m
        p_full(5)  = p_oe_free(5);   % Tmax_m
        p_full(6)  = p_oe_free(6);   % Zw_m
    case 'yaw'
        p_full(16) = p_oe_free(1);   % Nr
        p_full(12) = p_oe_free(2);   % Gamma3
        p_full(13) = p_oe_free(3);   % tau_r
    case {'heave','drop'}
        p_full(5)  = p_oe_free(1);   % Tmax_m
        p_full(6)  = p_oe_free(2);   % Zw_m
        p_full(7)  = p_oe_free(3);   % Zww_m
end

% Well-identified flags: only free params can be well-identified
% Criterion: std/|value| < 0.5
well_id = false(13,1);
for i = 1:length(p0_free)
    val = p_oe_free(i);
    std_i = sqrt(crb_oe_free(i,i));
    if abs(val) > 1e-6 && std_i/abs(val) < 0.5
        % Map back to full index
        switch excite_axis
            case 'roll'
                full_idx = [14, 8, 9];
            case 'pitch'
                full_idx = [15, 10, 11, 1, 5, 6];
            case 'yaw'
                full_idx = [16, 12, 13];
            case {'heave','drop'}
                full_idx = [5, 6, 7];
        end
        well_id(full_idx(i)) = true;
    end
end

% Build full crb — free params on diagonal, large values elsewhere
crb_full = eye(16) * 25.0;   % prior uncertainty for non-excited params
switch excite_axis
    case 'roll'
        idx = [14, 8, 9];
        crb_full(idx,idx) = crb_oe_free;
    case 'pitch'
        idx = [15, 10, 11, 1, 5, 6];
        crb_full(idx,idx) = crb_oe_free;
    case 'yaw'
        idx = [16, 12, 13];
        crb_full(idx,idx) = crb_oe_free;
    case {'heave','drop'}
        idx = [5, 6, 7];
        crb_full(idx,idx) = crb_oe_free;
end

% =============================================================================
% 11. Diagnostic plots
% =============================================================================
if DO_PLOT
    switch excite_axis
        case 'roll';   state_labels = {'p (rad/s)', '\phi (rad)'};
        case 'pitch';  state_labels = {'q (rad/s)', '\theta (rad)', 'u_b (m/s)', 'w_b (m/s)'};
        case 'yaw';    state_labels = {'r (rad/s)', '\psi (rad)'};
        case {'heave','drop'};  state_labels = {'w_b (m/s)'};
    end

    ns = size(z_oe, 2);
    figure('Name', ['Segment: ' tag], 'Position', [50 50 1100 500]);
    for i = 1:ns
        subplot(1, ns, i);
        hold on;
        plot(time, z_oe(:,i),  'b',  'LineWidth', 1.3, 'DisplayName', 'Measured');
        plot(time, y_oe(:,i),  'r--','LineWidth', 1.3, 'DisplayName', 'OE Model');
        hold off;
        title(state_labels{i});
        xlabel('t (s)');  ylabel(state_labels{i});
        legend('Location','best');
        grid on;
    end
    sgtitle(sprintf('Segment %s  |  cost=%.3f', tag, cost_oe));

    figure('Name', ['Input: ' tag], 'Position', [50 600 800 180]);
    switch excite_axis
        case 'roll';   plot(time, cr,  'k');  ylabel('cmd\_roll');
        case 'pitch';  plot(time, cp_, 'k');  ylabel('cmd\_pitch');
        case 'yaw';    plot(time, cy,  'k');  ylabel('cmd\_yaw');
        case {'heave','drop'};  plot(time, ct,  'k');  ylabel('cmd\_thrust');
    end
    xlabel('t (s)');  title(['Excitation input: ' tag]);  grid on;
end

% =============================================================================
% 12. Assemble and save result
% =============================================================================
result.tag             = tag;
result.regime_id       = regime_id;
result.axis            = excite_axis;
result.repetition      = rep;
result.theta0_target   = theta0_tgt;
result.phi0_target     = phi0_tgt;
result.theta0_actual   = theta0_actual;
result.phi0_actual     = phi0_actual;
result.T0_actual       = T0_actual;
result.trim_ok         = trim_ok;
result.free_names      = free_names;
result.p0_free         = p0_free;
result.p_oe_free       = p_oe_free;
result.crb_oe_free     = crb_oe_free;
result.p_full          = p_full;
result.crb_full        = crb_full;
result.cost_oe         = cost_oe;
result.param_names     = param_names;
result.well_identified = well_id;
result.t_excite        = time;
result.y_oe            = y_oe;
result.z_meas          = z_oe;

if DO_SAVE
    out_path = strrep(csv_path, '.csv', '_result.mat');
    save(out_path, 'result');
    fprintf('Result saved -> %s\n', out_path);
end

end
