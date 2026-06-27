function y = drone_statespace_reduced(p, u, t, x0, c)
% drone_statespace_reduced  Minimal-state quadrotor model for SIDPAC oe.m
%
% Each axis test integrates ONLY the states directly driven by the excited
% axis. All other states are supplied as measured time-series columns of u,
% treating them as known exogenous inputs. This gives a small, well-
% conditioned OE problem (2-5 free parameters per test).
%
% AXIS CODES  (c(2)):
%
%   1 = ROLL
%       Free params  p = [Lp, tau_p]         (2 params — Gamma1 fixed to 0)
%       States       x = [p_rate, phi]        (2 states)
%       u columns    [cmd_roll, q_meas, r_meas, theta_meas]
%                     1         2       3       4
%
%   2 = PITCH
%       Free params  p = [Mq, tau_q] or [Mq, tau_q, Xu_m]   (2 or 3 params)
%       States       x = [q_rate, theta, vx_b]  (3 states)
%       u columns    [cmd_pitch, p_meas, r_meas, phi_meas, vz_b_meas, vy_b_meas]
%                     1          2       3       4          5           6
%       c extra      c(3+nz:end) = [Tmax_m, Zw_m, Xu_m]  (fixed from heave/prior)
%
%   3 = YAW
%       Free params  p = [Nr, Gamma3, tau_r]       (3 params)
%       States       x = [r_rate, psi]             (2 states)
%       u columns    [cmd_yaw, p_meas, q_meas, phi_meas, theta_meas]
%                     1        2       3       4          5
%
%   4 = HEAVE
%       Free params  p = [Tmax_m, Zw_m, Zww_m]    (3 params)
%       States       x = [vz_b]                    (1 state)
%       u columns    [cmd_thrust]
%                     1
%
%   5 = LATERAL
%       Free params  p = [Yv_m, Yvv_m]             (2 params)
%       States       x = [vy_b]                    (1 state)
%       u columns    [phi_meas, theta_meas, p_meas, r_meas, vx_b_meas, vz_b_meas]
%                     1         2           3       4       5           6
%
% c = [g, axis_code, z_scale...]   (row vector)
%
% Per-axis free parameter vectors p:
%   roll  : [Lp, Gamma1, tau_p]
%   pitch : [Mq, Gamma2, tau_q, Xu_m, Tmax_m, Zw_m]
%   yaw   : [Nr, Gamma3, tau_r]
%   heave : [Tmax_m, Zw_m, Zww_m]

g         = c(1);
axis_code = round(c(2));

% c layout: [g, axis_code, z_scale(1..nz), (optional extra constants)]
% For pitch (axis_code=2): c = [g, 2, z_scale(1..3), Tmax_m, Zw_m]
% z_scale has exactly ns elements (one per state).
% Extra constants follow after z_scale — extracted inside each case.
if length(c) > 2
    % z_scale length = ns (determined after axis_code is known)
    % We extract it after ns is known below
    c_extra = c(3:end);   % everything after [g, axis_code]
else
    c_extra = [];
end

N      = length(t);
ns     = length(x0);
x      = zeros(ns, N);
x(:,1) = x0(:);

% Now that ns is known, split c_extra into z_scale and any axis-specific extras
if length(c_extra) >= ns
    z_scale  = c_extra(1:ns);
    c_params = c_extra(ns+1:end);   % extra constants (e.g. Tmax_m, Zw_m for pitch)
else
    z_scale  = [];
    c_params = [];
end

for i = 1:N-1
    dt_i = t(i+1) - t(i);
    u_i  = u(i,:);

    k1 = dyn(x(:,i),               u_i, g, axis_code, p, c_params);
    k2 = dyn(x(:,i)+0.5*dt_i*k1,  u_i, g, axis_code, p, c_params);
    k3 = dyn(x(:,i)+0.5*dt_i*k2,  u_i, g, axis_code, p, c_params);
    k4 = dyn(x(:,i)+dt_i*k3,      u_i, g, axis_code, p, c_params);

    x(:,i+1) = x(:,i) + (dt_i/6)*(k1 + 2*k2 + 2*k3 + k4);

    if any(~isfinite(x(:,i+1)))
        x(:,i+1:end) = 1e6;
        break;
    end
end

y = x';                     % [N x ns], physical units
y(~isfinite(y)) = 1e6;

% Normalise output to match scaled z_oe_sc seen by OE
if ~isempty(z_scale) && length(z_scale) == size(y,2)
    y = y ./ z_scale(:)';   % ensure row vector broadcast
elseif ~isempty(z_scale)
    warning('drone_statespace_reduced: z_scale length %d != ns %d — skipping', ...
            length(z_scale), size(y,2));
end
end


% =============================================================================
function xdot = dyn(x, u_row, g, axis_code, p, c_params)
% Dispatch to axis-specific minimal dynamics.

switch axis_code

    % =========================================================================
    % ROLL  —  states: [p_rate, phi]
    %          params: [Lp, tau_p]   (2 params — Gamma1 fixed to 0)
    %          u_row:  [cmd_roll, q_meas, r_meas, theta_meas]
    % =========================================================================
    case 1
        p_rate = x(1);
        phi    = x(2);

        Lp    = p(1);   % roll-rate damping (< 0)
        tau_p = p(2);   % roll control effectiveness (> 0)

        cmd_roll   = u_row(1);
        q_meas     = u_row(2);
        r_meas     = u_row(3);
        theta_meas = u_row(4);

        % Roll rate: damping + control (Gamma1=0, Coriolis term omitted)
        pdot = Lp * p_rate + tau_p * cmd_roll;

        % Roll angle kinematic using measured q, r, theta
        cos_the = cos(theta_meas);
        if abs(cos_the) < 0.1
            cos_the = sign(cos_the + 1e-9) * 0.1;
        end
        phidot = p_rate ...
                 + (q_meas*sin(phi) + r_meas*cos(phi)) * (sin(theta_meas)/cos_the);

        xdot = [pdot; phidot];

    % =========================================================================
    % PITCH  —  states: [q_rate, theta, vx_b]
    %           params: [Mq, tau_q]          (2 params — hover, Xu_m fixed)
    %                or [Mq, tau_q, Xu_m]    (3 params — non-hover)
    %           u_row:  [cmd_pitch, p_meas, r_meas, phi_meas, vz_b_meas]
    %                    1          2       3       4          5
    %           c_params: [Tmax_m, Zw_m, Xu_m]   (fixed constants from heave/prior)
    %
    % vz_b is passed as measured exogenous input (col 5) so the Coriolis
    % term -q*vz_b in udot is correctly captured at high forward speed where
    % the drone is simultaneously pitching and descending (vz_b nonzero).
    % Without this term, u_b oscillations driven by pitch rate are missed.
    %
    % At hover (|theta0| < 10 deg), Xu_m is unidentifiable because vx_b ≈ 0.
    % It is excluded from p0_free and fixed via c_params(3) instead.
    %
    % Note: q_rate has been negated in analyze_segment to be consistent
    % with thetadot = q_rate * cos(phi).
    % =========================================================================
    case 2
        q_rate = x(1);
        theta  = x(2);
        vx_b   = x(3);

        Mq    = p(1);   % pitch-rate damping (< 0)
        tau_q = p(2);   % pitch control effectiveness (> 0, sign-corrected)
        % Xu_m: free param at non-hover (p has 3 elements); fixed at hover via c_params(3).
        % c_params = [Tmax_m, Zw_m, Xu_m] — Xu_m always passed as fallback constant.
        if length(p) >= 3
            Xu_m = p(3);
        else
            Xu_m = c_params(3);   % hover case — fixed to prior, not in p0_free
        end
        % Gamma2 fixed to 0 — not identifiable from pitch test

        cmd_pitch  = u_row(1);
        p_meas     = u_row(2);
        r_meas     = u_row(3);
        phi_meas   = u_row(4);
        vz_b_meas  = u_row(5);   % measured heave velocity — -q*w Coriolis term
        vy_b_meas  = u_row(6);   % measured lateral velocity — r*v Coriolis term

        % Pitch rate: damping + control (Gamma2=0, Coriolis omitted)
        qdot = Mq * q_rate + tau_q * cmd_pitch;

        % Pitch kinematic: thetadot = q*cos(phi) - r*sin(phi)
        thetadot = q_rate*cos(phi_meas) - r_meas*sin(phi_meas);

        % Forward velocity — full Coriolis coupling:
        % udot = (r*v_NED - q*w) - g*sin(theta) + Xu_m*u
        % Since vy_b_sim = -vy_b_NED, the r*v term becomes -r*vy_b_sim.
        udot = -r_meas*vy_b_meas - q_rate*vz_b_meas - g*sin(theta) + Xu_m*vx_b;

        xdot = [qdot; thetadot; udot];

    % =========================================================================
    % YAW  —  states: [r_rate, psi]
    %         params: [Nr, Gamma3, tau_r]
    %         u_row:  [cmd_yaw, p_meas, q_meas, phi_meas, theta_meas]
    % =========================================================================
    case 3
        r_rate = x(1);
        psi    = x(2);

        Nr     = p(1);   % yaw-rate damping (< 0)
        Gamma3 = p(2);
        tau_r  = p(3);

        cmd_yaw    = u_row(1);
        p_meas     = u_row(2);
        q_meas     = u_row(3);
        phi_meas   = u_row(4);   % measured roll — correct psidot in banked regimes
        theta_meas = u_row(5);   % measured pitch — correct psidot in pitched regimes

        % Yaw rate: Euler equation with damping + Coriolis + control
        rdot = Nr * r_rate + Gamma3 * p_meas * q_meas + tau_r * cmd_yaw;

        % Yaw kinematic with actual trim angles — critical for B/C regimes where
        % phi≈30-50° and theta≈-30 to -60°. The 1/cos(theta) factor can be ~2x
        % at theta=-60°, so the near-hover approximation (phi=theta=0) is wrong.
        cos_the = max(abs(cos(theta_meas)), 0.1);
        psidot = (q_meas*sin(phi_meas) + r_rate*cos(phi_meas)) / cos_the;

        xdot = [rdot; psidot];

    % =========================================================================
    % HEAVE  —  states: [vz_b]
    %           params: [Tmax_m, Zw_m, Zww_m]
    %           u_row:  [cmd_thrust]
    % =========================================================================
    case 4
        vz_b = x(1);

        Tmax_m = p(1);
        Zw_m   = p(2);
        Zww_m  = p(3);

        cmd_thrust = u_row(1);

        % Heave at level hover: wdot = g - Tmax_m*dT + Zw_m*w + Zww_m*w*|w|
        wdot = g - Tmax_m*cmd_thrust + Zw_m*vz_b + Zww_m*vz_b*abs(vz_b);

        xdot = wdot;   % scalar — MATLAB handles [1x1] correctly

    % =========================================================================
    % LATERAL  —  states: [vy_b]
    %             params: [Yv_m, Yvv_m]
    %             u_row:  [phi_meas, theta_meas, p_meas, r_meas, vx_b_meas, vz_b_meas]
    %                      1         2           3       4       5           6
    %
    % Identification excitation: the drone is banked to ±PHI_LAT_DEG in steps
    % to generate lateral body velocity vy_b. The Coriolis and gravity forcing
    % are all measured; only Yv_m and Yvv_m are free parameters.
    % =========================================================================
    case 5
        vy_b = x(1);

        Yv_m  = p(1);   % lateral drag per unit velocity (< 0)
        Yvv_m = p(2);   % quadratic lateral drag (< 0)

        phi_meas   = u_row(1);
        theta_meas = u_row(2);
        p_meas     = u_row(3);
        r_meas     = u_row(4);
        vx_b_meas  = u_row(5);
        vz_b_meas  = u_row(6);

        % Lateral velocity — simulator convention: vy_b_sim = -vy_b_NED
        % Correct dynamics:
        %   vdot = -(p*vz_b - r*vx_b) - g*cos(theta)*sin(phi) + Yv_m*vy_b + Yvv_m*vy_b*|vy_b|
        vdot = -(p_meas*vz_b_meas - r_meas*vx_b_meas) ...
               - g*cos(theta_meas)*sin(phi_meas) ...
               + Yv_m*vy_b + Yvv_m*vy_b*abs(vy_b);

        xdot = vdot;

    otherwise
        error('drone_statespace_reduced: unknown axis_code %d', axis_code);
end
end