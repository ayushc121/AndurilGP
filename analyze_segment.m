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

% Full parameter vector (16 elements):
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
% 2. Load CSV using a single readtable call for consistency
%
% Using readtable for everything avoids row-count mismatches between
% readmatrix (numeric only) and readtable (all columns). The CSV column
% order is fixed:
%   1=t  2=x  3=y  4=z  5=vx_b  6=vy_b  7=vz_b
%   8=phi  9=theta  10=psi  11=p  12=q  13=r
%   14=ax_b  15=ay_b  16=az_b  (unused)
%   17=cmd_roll  18=cmd_pitch  19=cmd_yaw  20=cmd_thrust
%   21=phase_tag  22=regime_id
% =============================================================================
opts               = detectImportOptions(csv_path);
opts.DataLines     = [2, Inf];
opts.VariableNamesLine = 1;
T_full             = readtable(csv_path, opts);

% Verify expected columns exist
expected_cols = {'t','x','y','z','vx_b','vy_b','vz_b', ...
                 'phi','theta','psi','p','q','r', ...
                 'cmd_roll','cmd_pitch','cmd_yaw','cmd_thrust','phase_tag'};
for ec = expected_cols
    if ~ismember(ec{1}, T_full.Properties.VariableNames)
        error('[%s] CSV missing expected column: %s', tag, ec{1});
    end
end

% Extract all columns by name — immune to column order changes
time_all       = T_full.t;
vx_b_all       = T_full.vx_b;
vy_b_all       = T_full.vy_b;
vz_b_all       = T_full.vz_b;
phi_all        = T_full.phi;
theta_all      = T_full.theta;
psi_all        = T_full.psi;
p_all          = T_full.p;
q_all          = T_full.q;
r_all          = T_full.r;
cmd_roll_all   = T_full.cmd_roll;
cmd_pitch_all  = T_full.cmd_pitch;
cmd_yaw_all    = T_full.cmd_yaw;
cmd_thrust_all = T_full.cmd_thrust;
phase_col      = T_full.phase_tag;

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
    % Use 10 deg tolerance — banked regimes take longer to settle
    % and trim verification is done on hold phase which may have slight error
    trim_ok   = (theta_err < 10.0) && (phi_err < 10.0);

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

% ── Sign and continuity corrections ──────────────────────────────────────────
% PITCH: simulator reports pitchspeed with opposite sign to thetadot.
% Confirmed: positive cmd_pitch → negative q_raw → positive thetadot (nose-up).
% Negate q_e so it is consistent with thetadot = q_e * cos(phi).
if strcmp(excite_axis, 'pitch')
    q_e = -q_e;
    fprintf('[INFO] q_e negated for pitch convention correction.\n');
end

% YAW: psi wraps at ±pi. Unwrap so OE integrates a continuous signal.
% Without this, the model integrates smoothly while measured psi jumps ±2pi.
if strcmp(excite_axis, 'yaw')
    psi_e = unwrap(psi_e);
    fprintf('[INFO] psi_e unwrapped. Range: [%.3f, %.3f] rad\n', ...
            min(psi_e), max(psi_e));
end
% ─────────────────────────────────────────────────────────────────────────────

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
PRIOR_Xu_m   = -0.0117; PRIOR_Xuu_m = -0.0493;  % Xu_m_true and Xuu_m from sub-task 1 regression
PRIOR_Yv_m   = -0.08899; PRIOR_Yvv_m = -0.03909; % identified from lateral trim sweep fit_yv_m.m
PRIOR_Tmax_m =  36.24; PRIOR_Zw_m  = -0.18;   PRIOR_Zww_m = -0.043; % heave/D1 OE
PRIOR_G1     =  0.0;   PRIOR_tau_p =  44.0;   % avg unsaturated regimes P0-P3, B1-B2
PRIOR_G2     =  0.0;   PRIOR_tau_q =  57.0;   % avg unsaturated regimes P0-P3, B1-B2
PRIOR_G3     =  0.0;   PRIOR_tau_r =  29.7;   % avg P0+P1 yaw OE
PRIOR_Lp     = -19.0;  % avg unsaturated regimes P0-P3, B1-B2
PRIOR_Mq     = -24.0;  % avg unsaturated regimes P0-P3, B1-B2
PRIOR_Nr     = -13.5;  % very consistent across all regimes

switch excite_axis

    % ── ROLL: free = [Lp, tau_p] ─────────────────────────────────────────────
    case 'roll'
        % p-equation: pdot = Lp*p + Gamma1*q*r + tau_p*cmd_roll
        %
        % Gamma1 = (Iyy-Izz)/Ixx is NOT identifiable from roll tests because
        % q*r is near-zero during pure roll excitation. lesq produces wildly
        % varying estimates (e.g. -14125 at B3). Fix Gamma1=0 and only
        % identify [Lp, tau_p].
        %
        % Lp < 0: roll-rate aerodynamic damping
        % tau_p > 0: roll control effectiveness (confirmed positive by data)
        z_p   = pdot;
        Phi_p = [p_e,  cr];           % only Lp and tau_p regressors
        lb_p  = [-Inf; 1e-4];         % tau_p > 0
        ub_p  = [-1e-4; Inf];         % Lp strictly negative
        pp    = lsqlin(Phi_p, z_p, [], [], [], [], lb_p, ub_p, [], lsq_opts);
        Lp_hat     = pp(1);
        Gamma1_hat = PRIOR_G1;        % fixed to 0 — not identifiable
        tau_p_hat  = pp(2);
        fprintf('p-eq:  Lp=%.5f  Gamma1 fixed=%.5f  tau_p=%.5f\n', ...
                Lp_hat, Gamma1_hat, tau_p_hat);

        p0_free    = [Lp_hat; tau_p_hat];
        free_names = {'Lp','tau_p'};

    % ── PITCH: free = [Mq, tau_q, Xu_m] ─────────────────────────────────────
    % Gamma2 = (Izz-Ixx)/Iyy is NOT identifiable from pitch tests because
    % p*r is near-zero during pure pitch excitation. lesq produces wildly
    % varying estimates. Fix Gamma2=0, only identify [Mq, tau_q, Xu_m].
    %
    % Tmax_m and Zw_m are fixed to heave/drop prior values.
    %
    % Xu_m at hover (P0) is essentially unidentifiable since vx_b ≈ 0.
    % At P0 we skip Xu_m identification and use the prior from P1.
    case 'pitch'
        % q-equation: qdot = Mq*q + tau_q*cmd_pitch
        % Gamma2 fixed to 0 — not identifiable from pitch test
        z_q   = qdot;
        Phi_q = [q_e,  cp_];          % only Mq and tau_q regressors
        lb_q  = [-Inf; 1e-4];         % tau_q > 0 after q_e sign correction
        ub_q  = [-1e-4; Inf];         % Mq strictly negative
        pq    = lsqlin(Phi_q, z_q, [], [], [], [], lb_q, ub_q, [], lsq_opts);
        Mq_hat     = pq(1);
        Gamma2_hat = PRIOR_G2;        % fixed to 0 — not identifiable
        tau_q_hat  = pq(2);
        fprintf('q-eq:  Mq=%.5f  Gamma2 fixed=%.5f  tau_q=%.5f\n', ...
                Mq_hat, Gamma2_hat, tau_q_hat);

        % u-equation: Xu_m only (Xuu_m fixed 0)
        % Skip at hover (theta0 ≈ 0) where vx_b ≈ 0 makes Xu_m unidentifiable.
        % Use prior value from P1 instead.
        % Skip Xu_m identification when trim pitch is near hover (|theta0| < 10 deg)
        % because vx_b ≈ 0 at hover makes Xu_m unidentifiable — OE wanders.
        % Use theta0_tgt (the intended trim angle) not mean(the_e) which
        % oscillates around theta0 during excitation and gives wrong result.
        if abs(theta0_tgt) < deg2rad(10)
            Xu_m_hat = PRIOR_Xu_m;
            fprintf('u-eq:  Xu_m fixed to prior=%.5f (|theta0|<10 deg — vx_b too small)\n', Xu_m_hat);
        else
            z_u   = udot - (r_e.*vy_b - q_e.*vz_b) + g.*sin(the_e);
            Phi_u = vx_b;
            Xu_m_hat = lsqlin(Phi_u, z_u, [], [], [], [], -Inf, -1e-6, [], lsq_opts);
            fprintf('u-eq:  Xu_m=%.5f\n', Xu_m_hat);
        end

        fprintf('fixed:  Tmax_m=%.5f  Zw_m=%.5f  Gamma2=%.5f\n', ...
                PRIOR_Tmax_m, PRIOR_Zw_m, Gamma2_hat);

        % At hover (|theta0| < 10 deg), vx_b ≈ 0 so OE cannot determine Xu_m.
        % Exclude Xu_m from p0_free entirely so the optimizer cannot move it.
        % Setting it as an initial value is not sufficient — OE will still
        % wander on the flat vx_b cost surface and produce nonsense values.
        if abs(theta0_tgt) < deg2rad(10)
            p0_free    = [Mq_hat; tau_q_hat];
            free_names = {'Mq','tau_q'};
        else
            p0_free    = [Mq_hat; tau_q_hat; Xu_m_hat];
            free_names = {'Mq','tau_q','Xu_m'};
        end

    % ── YAW: free = [Nr, Gamma3, tau_r] ─────────────────────────────────────
    case 'yaw'
        % r-equation: rdot = Nr*r + Gamma3*p*q + tau_r*cmd_yaw
        % Nr < 0 is yaw-rate aerodynamic damping.
        %
        % Gamma3 = (Ixx-Iyy)/Izz is NOT identifiable from yaw tests because
        % p*q is near-zero during a pure yaw excitation at any trim condition.
        % lesq produces wildly varying estimates (e.g. -8383 at P1).
        % Fix Gamma3 = 0 (symmetric quadrotor assumption) and only identify
        % Nr and tau_r. OE will be given a very tight prior on Gamma3 so it
        % cannot move from zero regardless of cost surface shape.
        z_r   = rdot;
        Phi_r = [r_e,  cy];          % only Nr and tau_r regressors
        lb_r  = [-Inf; -Inf];
        ub_r  = [-1e-4; Inf];        % Nr strictly negative
        pr    = lsqlin(Phi_r, z_r, [], [], [], [], lb_r, ub_r, [], lsq_opts);
        Nr_hat     = pr(1);
        Gamma3_hat = PRIOR_G3;       % fixed to 0 — not identifiable from yaw test
        tau_r_hat  = pr(2);
        fprintf('r-eq:  Nr=%.5f  Gamma3 fixed=%.5f  tau_r=%.5f\n', ...
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

    % ── LATERAL: free = [Yv_m, Yvv_m] ──────────────────────────────────────
    % Excited by bank-step sequence that generates lateral body velocity vy_b.
    % Yv_m < 0: linear lateral drag
    % Yvv_m < 0: quadratic lateral drag (only identifiable at vy_b > 3 m/s)
    %
    % SIGN CONVENTION: vy_b in this simulator is measured with OPPOSITE sign
    % to standard NED body-y (positive = leftward). The correct dynamics are:
    %   vdot = -(p*vz_b - r*vx_b) - g*cos(θ)*sin(φ) + Yv_m*v + Yvv_m*v*|v|
    % Residual after subtracting forcing terms:
    %   z_v = vdot + (p*vz_b - r*vx_b) + g*cos(θ)*sin(φ)
    case 'lateral'
        z_v   = vdot + (p_e.*vz_b - r_e.*vx_b) + g.*cos(the_e).*sin(phi_e);
        Phi_v = [vy_b,  vy_b.*abs(vy_b)];
        lb_v  = [-Inf; -Inf];
        ub_v  = [-1e-6; -1e-6];   % both must be strictly negative
        pv    = lsqlin(Phi_v, z_v, [], [], [], [], lb_v, ub_v, [], lsq_opts);
        Yv_m_hat   = pv(1);
        Yvv_m_hat  = pv(2);
        fprintf('v-eq:  Yv_m=%.5f  Yvv_m=%.5f\n', Yv_m_hat, Yvv_m_hat);

        p0_free    = [Yv_m_hat; Yvv_m_hat];
        free_names = {'Yv_m','Yvv_m'};

    otherwise
        error('Unknown excite_axis: %s  (valid: roll|pitch|yaw|heave|drop|lateral)', excite_axis);
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
        % Gamma1 removed from free params — p0_free is now [Lp, tau_p]
        x0_oe  = [p_e(1); phi_e(1)];
        z_oe   = [p_e,  phi_e];
        u_oe   = [cr,  q_e,  r_e,  the_e];   % [N x 4]
        axis_code = 1;

    case 'pitch'
        % States: [q_rate, theta, vx_b]
        % vz_b is NOT a state but IS passed as exogenous input (col 5) so the
        % Coriolis term -q*vz_b in udot is correctly captured.
        % vy_b is also passed (col 6) for the r*vy_b Coriolis term, which is
        % significant for B/C regimes where vy_b is nonzero at trim.
        x0_oe  = [q_e(1); the_e(1); vx_b(1)];
        z_oe   = [q_e,  the_e,  vx_b];
        u_oe   = [cp_,  p_e,  r_e,  phi_e,  vz_b,  vy_b];   % [N x 6]
        axis_code = 2;

    case 'yaw'
        % States: [r_rate, psi]
        % phi_e and the_e passed as exogenous inputs so psidot is computed
        % correctly in banked/pitched regimes (B/C) instead of using phi=theta=0.
        x0_oe  = [r_e(1); psi_e(1)];
        z_oe   = [r_e,  psi_e];
        u_oe   = [cy,  p_e,  q_e,  phi_e,  the_e];   % [N x 5]
        axis_code = 3;

    case {'heave','drop'}
        % States: [vz_b]
        x0_oe  = vz_b(1);
        z_oe   = vz_b(:);            % [N x 1] column
        u_oe   = ct(:);              % [N x 1]
        axis_code = 4;

    case 'lateral'
        % States: [vy_b]
        % All Coriolis and gravity forcing terms are exogenous — only Yv_m
        % and Yvv_m are free. phi_e provides the gravity excitation forcing.
        x0_oe  = vy_b(1);
        z_oe   = vy_b(:);            % [N x 1] column
        u_oe   = [phi_e,  the_e,  p_e,  r_e,  vx_b,  vz_b];   % [N x 6]
        axis_code = 5;
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

% c passed as row vector [g, axis_code, z_scale..., extra_constants]
% For pitch: append PRIOR_Tmax_m, PRIOR_Zw_m, and PRIOR_Xu_m as fixed constants
% so drone_statespace_reduced always has Xu_m available regardless of whether
% it is in p0_free (non-hover) or fixed (hover).
if strcmp(excite_axis, 'pitch')
    c_oe = [g, axis_code, z_scale, PRIOR_Tmax_m, PRIOR_Zw_m, PRIOR_Xu_m];
else
    c_oe = [g, axis_code, z_scale];
end

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
        % 2 free params: [Lp, tau_p] — Gamma1 fixed to 0
        sig_free = [abs(p0_free(1)) * 0.05;       % Lp     — 5%
                    abs(p0_free(2)) * 0.02];       % tau_p  — 2%

    case 'pitch'
        % 3 free params: [Mq, tau_q, Xu_m] — at non-hover
        % 2 free params: [Mq, tau_q]        — at hover (Xu_m excluded)
        sig_free = [abs(p0_free(1)) * 0.05;       % Mq     — 5%
                    abs(p0_free(2)) * 0.02];       % tau_q  — 2%
        if length(p0_free) == 3
            sig_free(3) = abs(p0_free(3)) * 0.10; % Xu_m   — 10%
        end

    case 'yaw'
        % Gamma3 fixed to 0 — tight prior prevents OE from moving it.
        % p0_free(2) = PRIOR_G3 = 0, so we use an absolute sigma of 0.01.
        sig_free = [abs(p0_free(1)) * 0.05;       % Nr      — 5%
                    0.01;                          % Gamma3  — TIGHT (not identifiable)
                    abs(p0_free(3)) * 0.02];       % tau_r   — 2%

    case {'heave','drop'}
        % Loosen Tmax_m to 10% — lesq estimate can be biased at extreme thrust
        sig_free = [abs(p0_free(1)) * 0.10;      % Tmax_m — 10% (loosened)
                    abs(p0_free(2)) * 0.10;       % Zw_m   — 10% (very small value)
                    max(abs(p0_free(3)), 0.01) * 0.10]; % Zww_m — 10%

    case 'lateral'
        sig_free = [abs(p0_free(1)) * 0.10;                  % Yv_m  — 10%
                    max(abs(p0_free(2)), 0.005) * 0.15];      % Yvv_m — 15% (small value)
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
        % 2 free params: [Lp, tau_p]
        p_full(14) = p_oe_free(1);   % Lp
        p_full(9)  = p_oe_free(2);   % tau_p
        p_full(8)  = PRIOR_G1;       % Gamma1 — fixed to 0
    case 'pitch'
        % 3 free params at non-hover: [Mq, tau_q, Xu_m]
        % 2 free params at hover:     [Mq, tau_q]  (Xu_m fixed to prior)
        p_full(15) = p_oe_free(1);   % Mq
        p_full(11) = p_oe_free(2);   % tau_q
        if length(p_oe_free) >= 3
            p_full(1) = p_oe_free(3);   % Xu_m — non-hover only
        end
        % else p_full(1) stays at PRIOR_Xu_m set at initialisation above
        p_full(10) = PRIOR_G2;       % Gamma2 — fixed to 0
        p_full(5)  = PRIOR_Tmax_m;   % fixed from heave
        p_full(6)  = PRIOR_Zw_m;     % fixed from heave/drop
    case 'yaw'
        p_full(16) = p_oe_free(1);   % Nr
        p_full(12) = p_oe_free(2);   % Gamma3
        p_full(13) = p_oe_free(3);   % tau_r
    case {'heave','drop'}
        p_full(5)  = p_oe_free(1);   % Tmax_m
        p_full(6)  = p_oe_free(2);   % Zw_m
        p_full(7)  = p_oe_free(3);   % Zww_m
    case 'lateral'
        p_full(3)  = p_oe_free(1);   % Yv_m
        p_full(4)  = p_oe_free(2);   % Yvv_m
end

% Well-identified flags: only free params can be well-identified
% Criterion: std/|value| < 0.5
well_id = false(16,1);   % 16 params — matches NP in drone_sysid_main.m
for i = 1:length(p0_free)
    val = p_oe_free(i);
    std_i = sqrt(crb_oe_free(i,i));
    if abs(val) > 1e-6 && std_i/abs(val) < 0.5
        % Map free-param index i back to full 16-element param vector index
        switch excite_axis
            case 'roll'
                full_idx = [14, 9];         % Lp, tau_p
            case 'pitch'
                if length(p0_free) == 3
                    full_idx = [15, 11, 1]; % Mq, tau_q, Xu_m (non-hover)
                else
                    full_idx = [15, 11];    % Mq, tau_q only (hover)
                end
            case 'yaw'
                full_idx = [16, 12, 13];
            case {'heave','drop'}
                full_idx = [5, 6, 7];
            case 'lateral'
                full_idx = [3, 4];   % Yv_m, Yvv_m
        end
        well_id(full_idx(i)) = true;
    end
end

% Build full crb — free params on diagonal, large values elsewhere
crb_full = eye(16) * 25.0;   % prior uncertainty for non-excited params
switch excite_axis
    case 'roll'
        idx = [14, 9];   % Lp, tau_p
        crb_full(idx,idx) = crb_oe_free;
    case 'pitch'
        if length(p0_free) == 3
            idx = [15, 11, 1];   % Mq, tau_q, Xu_m (non-hover)
        else
            idx = [15, 11];      % Mq, tau_q only (hover — Xu_m stays at prior)
        end
        crb_full(idx,idx) = crb_oe_free;
    case 'yaw'
        idx = [16, 12, 13];
        crb_full(idx,idx) = crb_oe_free;
    case {'heave','drop'}
        idx = [5, 6, 7];
        crb_full(idx,idx) = crb_oe_free;
    case 'lateral'
        idx = [3, 4];   % Yv_m, Yvv_m
        crb_full(idx,idx) = crb_oe_free;
end

% =============================================================================
% 11. Diagnostic plots
% =============================================================================
if DO_PLOT
    switch excite_axis
        case 'roll';   state_labels = {'p (rad/s)', '\phi (rad)'};
        case 'pitch';  state_labels = {'q (rad/s)', '\theta (rad)', 'u_b (m/s)'};
        case 'yaw';    state_labels = {'r (rad/s)', '\psi (rad)'};
        case {'heave','drop'};  state_labels = {'w_b (m/s)'};
        case 'lateral';         state_labels = {'v_b (m/s)'};
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