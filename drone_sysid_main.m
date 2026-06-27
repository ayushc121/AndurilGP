% =============================================================================
% drone_sysid_main.m  (v4 — updated for 16-parameter model)
%
% Parameter vector p (16 elements):
%   1=Xu_m   2=Xuu_m  3=Yv_m   4=Yvv_m  5=Tmax_m  6=Zw_m   7=Zww_m
%   8=Gamma1 9=tau_p  10=Gamma2 11=tau_q 12=Gamma3 13=tau_r
%   14=Lp    15=Mq    16=Nr
%
% WORKFLOW
%   1. Point DATA_DIR at the folder containing CSV + meta JSON files.
%   2. Run this script. Calls analyze_segment() on every valid pair,
%      saving *_result.mat alongside each CSV.
%   3. Results aggregated by regime into gain_schedule.mat.
% =============================================================================

clear; clc; close all;

DATA_DIR = '.';
OUT_FILE = fullfile(DATA_DIR, 'gain_schedule.mat');
PLOT_ALL = false;
MIN_COST = inf;

NP = 16;   % total parameter count

% Thrust saturation threshold — regimes with T0 above this are flagged.
% At saturation, identified tau_p/tau_q/Lp/Mq are unreliable because the
% actuator cannot increase thrust further, making control authority appear
% lower than it actually is. These params are replaced by extrapolation.
T_SAT_THRESH = 0.90;

% Segments to exclude — add filenames (without path) of known bad segments.
% Useful for discarding flips, unstable excitations, etc. without deleting CSVs.
EXCLUDE_SEGMENTS = {
    'sysid_P5_pitch_1.csv', ...
    'sysid_P5_pitch_2.csv', ...
    'sysid_P0_heave_2.csv', ...
    'sysid_P0_heave_3.csv', ...
    'sysid_P0_heave_4.csv', ...
    'sysid_P0_lateral_1.csv', ...
    'sysid_P0_lateral_2.csv', ...
    'sysid_P0_lateral_3.csv', ...
};


param_names = {'Xu_m','Xuu_m','Yv_m','Yvv_m','Tmax_m','Zw_m','Zww_m', ...
               'Gamma1','tau_p','Gamma2','tau_q','Gamma3','tau_r', ...
               'Lp','Mq','Nr'};

% =============================================================================
% 1. Find all segment CSV files
% =============================================================================
csv_list = dir(fullfile(DATA_DIR, 'sysid_*.csv'));
csv_list = csv_list(~contains({csv_list.name}, '_result'));

if isempty(csv_list)
    error('No sysid_*.csv files found in %s', DATA_DIR);
end
fprintf('Found %d CSV files in %s\n', length(csv_list), DATA_DIR);

% =============================================================================
% 2. Process each segment
% =============================================================================
all_results = {};

for k = 1:length(csv_list)
    csv_path  = fullfile(DATA_DIR, csv_list(k).name);
    meta_path = strrep(csv_path, '.csv', '_meta.json');

    if ~isfile(meta_path)
        fprintf('[SKIP] No metadata JSON for %s\n', csv_list(k).name);
        continue;
    end

    % Skip excluded segments before loading cached results
    if any(strcmp(csv_list(k).name, EXCLUDE_SEGMENTS))
        fprintf('[EXCL] %s  (in exclusion list)\n', csv_list(k).name);
        continue;
    end

    result_path = strrep(csv_path, '.csv', '_result.mat');
    if isfile(result_path)
        fprintf('[LOAD] %s  (already processed)\n', csv_list(k).name);
        tmp = load(result_path, 'result');
        all_results{end+1} = tmp.result; %#ok<SAGROW>
        continue;
    end

    fprintf('[RUN]  %s\n', csv_list(k).name);
    try
        result = analyze_segment(csv_path, meta_path, ...
                                 'plot', PLOT_ALL, 'save', true);
        all_results{end+1} = result; %#ok<SAGROW>
    catch ME
        fprintf('[ERROR] %s: %s\n', csv_list(k).name, ME.message);
    end
end

if isempty(all_results)
    error('No segments successfully processed.');
end
fprintf('\nProcessed %d segments total.\n', length(all_results));

% =============================================================================
% 3. Quality filter
% =============================================================================
keep = true(length(all_results), 1);
for k = 1:length(all_results)
    r = all_results{k};
    if r.cost_oe > MIN_COST
        fprintf('[FILTER] %s rejected: cost=%.2f > %.2f\n', ...
                r.tag, r.cost_oe, MIN_COST);
        keep(k) = false;
    end
    if ~r.trim_ok
        fprintf('[WARN]   %s: trim not reached — keeping but flagged.\n', r.tag);
    end
end
all_results = all_results(keep);
fprintf('After quality filter: %d segments retained.\n', length(all_results));

% =============================================================================
% 4. Aggregate by regime
% =============================================================================
regime_ids = unique(cellfun(@(r) r.regime_id, all_results, 'UniformOutput', false));
fprintf('\nRegimes found: %s\n', strjoin(regime_ids, ', '));

% Re-order so all unsaturated regimes are processed before saturated ones.
% This ensures the extrapolation pool for saturated regimes (B3, C2, P4)
% contains every unsaturated anchor, including P3 at the same theta0 as B3/C2.
regime_T0 = zeros(1, length(regime_ids));
for ri_tmp = 1:length(regime_ids)
    [~, ~, T0_tmp] = get_trim_actuals(all_results, regime_ids{ri_tmp});
    regime_T0(ri_tmp) = T0_tmp;
end
is_sat_flag = regime_T0 > T_SAT_THRESH;   % logical: 0=unsaturated, 1=saturated
[~, sort_ord] = sort(is_sat_flag);         % false sorts before true
regime_ids = regime_ids(sort_ord);
fprintf('Processing order (unsaturated first): %s\n', strjoin(regime_ids, ', '));

% P0 hover fallback for NaN-filling
p0_hover = get_regime_aggregate(all_results, 'P0', NP);
if isempty(p0_hover)
    warning('No P0 segments found — using hardcoded prior fallback.');
    p0_hover = [-0.41; 0; -0.30; 0; 36.24; -0.18; -0.043; ...
                0; 47.8; 0; 55.5; 0; 29.7; -20.2; -23.3; -13.5];
end

gs.param_names  = param_names;
gs.regimes_cell = {};   % build as cell array, convert to struct array at end

% Hard-coded nominal phi0 for banked regimes.
% phi0_target from the metadata is unreliable if PHI0_DEG was set to 0 in the
% excitation script config. This table provides the ground truth from the test
% plan. Values are in degrees; only B and C regimes have nonzero phi0.
% Note: tests were run at positive phi0. The A matrix gravity coupling terms
% depend on |phi0|, not sign, so positive values are used here.
PHI0_OVERRIDE_DEG = struct('B1', 30, 'B2', 45, 'B3', 50, 'C1', 30, 'C2', 50);

% Hard-coded fallback priors for NaN-filling — matches analyze_segment.m priors
% These fill params that were never well-identified in any segment
%   [1] Xu_m    = -0.0117  (Xu_m_true at zero speed — sub-task 1 regression)
%   [2] Xuu_m   = -0.0493  (quadratic forward drag — sub-task 1 regression)
%   [3] Yv_m    = -0.08899 (lateral linear drag — fit_yv_m.m)
%   [4] Yvv_m   = -0.03909 (lateral quadratic drag — fit_yv_m.m)
%   [5] Tmax_m  = 36.24    (from P0_heave_1)
%   [6] Zw_m    = -0.18    (from P0_heave_1)
%   [7] Zww_m   = -0.043   (from D1_drop_1)
HARD_PRIOR = [-0.0117; -0.0493; -0.08899; -0.03909; 36.24; -0.18; -0.043; ...
               0; 44.0; 0; 57.0; 0; 29.7; -19.0; -24.0; -13.5];

for ri = 1:length(regime_ids)
    rid   = regime_ids{ri};
    p_agg = get_regime_aggregate(all_results, rid, NP);

    if isempty(p_agg)
        fprintf('[SKIP regime] %s — no valid segments.\n', rid);
        continue;
    end

    % Step 1: Fill NaN from P0 hover aggregate
    nan_mask = isnan(p_agg);
    if any(nan_mask) && ~isempty(p0_hover)
        p_agg(nan_mask) = p0_hover(nan_mask);
        nan_mask = isnan(p_agg);   % recheck after P0 fill
    end

    % Step 2: Fill any remaining NaN from hard-coded priors
    % (happens when P0 itself has NaN for a parameter)
    if any(nan_mask)
        p_agg(nan_mask) = HARD_PRIOR(nan_mask);
        fprintf('[%s] %d params filled from hard prior (P0 also NaN).\n', ...
                rid, sum(nan_mask));
    end

    [theta0_act, phi0_act, T0_act] = get_trim_actuals(all_results, rid);

    % Override phi0 from lookup table for banked regimes.
    % The metadata phi0_deg may be 0 if PHI0_DEG was left at default in the
    % excitation script. Use the known test-plan values instead.
    if isfield(PHI0_OVERRIDE_DEG, rid)
        phi0_act = deg2rad(PHI0_OVERRIDE_DEG.(rid));
        fprintf('[%s] phi0 overridden from lookup table: %.1f deg\n', rid, rad2deg(phi0_act));
    end

    % Step 3: Flag saturated-thrust regimes and replace unreliable params
    % At T0 > T_SAT_THRESH, identified tau_p/tau_q/Lp/Mq are not reliable
    % because actuator saturation makes control authority appear ~3x lower
    % than it actually is. Replace these with linear extrapolation from
    % the nearest two unsaturated regimes along the pitch-angle axis.
    saturated = (T0_act > T_SAT_THRESH);
    if saturated
        fprintf('[%s] SATURATED THRUST (T0=%.3f > %.2f) — replacing tau_p/tau_q/Lp/Mq\n', ...
                rid, T0_act, T_SAT_THRESH);

        % Find the two nearest unsaturated P-regime entries already processed
        % by scanning gs.regimes_cell for non-saturated entries
        unsat_theta = [];  unsat_tau_p = [];  unsat_tau_q = [];
        unsat_Lp    = [];  unsat_Mq    = [];  unsat_tau_r = [];
        for kk = 1:length(gs.regimes_cell)
            re = gs.regimes_cell{kk};
            if ~re.saturated
                unsat_theta(end+1) = re.theta0;       %#ok
                unsat_tau_p(end+1) = re.p(9);         %#ok
                unsat_tau_q(end+1) = re.p(11);        %#ok
                unsat_Lp(end+1)    = re.p(14);        %#ok
                unsat_Mq(end+1)    = re.p(15);        %#ok
                unsat_tau_r(end+1) = re.p(13);        %#ok
            end
        end

        if length(unsat_theta) >= 2
            % Linear interpolation/extrapolation using all unsaturated points
            p_agg(9)  = interp1(unsat_theta, unsat_tau_p, theta0_act, 'linear', 'extrap');
            p_agg(11) = interp1(unsat_theta, unsat_tau_q, theta0_act, 'linear', 'extrap');
            p_agg(14) = interp1(unsat_theta, unsat_Lp,    theta0_act, 'linear', 'extrap');
            p_agg(15) = interp1(unsat_theta, unsat_Mq,    theta0_act, 'linear', 'extrap');
            p_agg(13) = interp1(unsat_theta, unsat_tau_r, theta0_act, 'linear', 'extrap');
            fprintf('       tau_p extrapolated=%.4f  tau_q=%.4f  Lp=%.4f  Mq=%.4f  tau_r=%.4f\n', ...
                    p_agg(9), p_agg(11), p_agg(14), p_agg(15), p_agg(13));
        else
            fprintf('       [WARN] Not enough unsaturated regimes yet for extrapolation.\n');
        end
    end

    entry.regime_id   = rid;
    entry.theta0      = theta0_act;
    entry.phi0        = phi0_act;
    entry.T0          = T0_act;
    entry.saturated   = saturated;
    entry.p           = p_agg;
    entry.param_names = param_names;

    [A, B, v0_trim] = linearise_drone(p_agg, theta0_act, phi0_act);
    entry.A      = A;
    entry.B      = B;
    entry.v0_trim = v0_trim;

    gs.regimes_cell{end+1} = entry;

    sat_str = '';
    if saturated; sat_str = ' [SATURATED-EXTRAPOLATED]'; end
    fprintf('[%s]  theta0=%.1f deg  phi0=%.1f deg  T0=%.3f  v0=%.2f m/s%s\n', ...
            rid, rad2deg(theta0_act), rad2deg(phi0_act), T0_act, v0_trim, sat_str);
    fprintf('       Tmax_m=%.3f  tau_p=%.4f  tau_q=%.4f  tau_r=%.4f\n', ...
            p_agg(5), p_agg(9), p_agg(11), p_agg(13));
    fprintf('       Lp=%.4f  Mq=%.4f  Nr=%.4f\n', ...
            p_agg(14), p_agg(15), p_agg(16));
end

% =============================================================================
% 5. Convert regime cell array to struct array and save
% =============================================================================
% Convert cell array to struct array now that all entries are complete
% and have identical fields — avoids dissimilar-struct assignment errors.
if ~isempty(gs.regimes_cell)
    gs.regimes = [gs.regimes_cell{:}];
else
    gs.regimes = struct([]);
end
gs = rmfield(gs, 'regimes_cell');   % clean up the temporary cell field

save(OUT_FILE, 'gs');
fprintf('\nGain schedule saved → %s\n', OUT_FILE);
fprintf('Contains %d regimes.\n', length(gs.regimes));

% =============================================================================
% 6. Summary plots — parameter trends vs pitch trim
% =============================================================================
if length(gs.regimes) >= 2
    p_regime_ids = {'P0','P1','P2','P3','P4','P5'};
    theta0_vals  = [];
    param_matrix = [];

    for ri = 1:length(gs.regimes)
        if ismember(gs.regimes(ri).regime_id, p_regime_ids)
            theta0_vals(end+1)    = rad2deg(gs.regimes(ri).theta0); %#ok
            param_matrix(end+1,:) = gs.regimes(ri).p';              %#ok
        end
    end

    if ~isempty(theta0_vals)
        [theta0_vals, idx] = sort(theta0_vals);
        param_matrix = param_matrix(idx,:);

        % Key params to plot: indices into 16-element vector
        key_params = [5,  9,  11, 13,  1,   6,  14,  15,  16];
        key_labels = {'Tmax\_m','tau\_p','tau\_q','tau\_r', ...
                      'Xu\_m','Zw\_m','Lp','Mq','Nr'};

        figure('Name','Parameter Trends vs Pitch Trim','Position',[50 50 1400 800]);
        for ki = 1:length(key_params)
            subplot(3, 3, ki);
            plot(theta0_vals, param_matrix(:, key_params(ki)), 'bo-', 'LineWidth', 1.5);
            xlabel('\theta_0 (deg)');
            ylabel(key_labels{ki});
            title(key_labels{ki});
            grid on;
        end
        sgtitle('Identified Parameters vs Pitch Trim Angle');
        exportgraphics(gcf, fullfile(DATA_DIR,'param_vs_trim.png'), 'Resolution', 300);
        fprintf('Parameter trend plot saved.\n');
    end
end

% =============================================================================
% 7. Full parameter summary — all segments, easy to copy-paste
% =============================================================================
fprintf('\n');
fprintf('=================================================================\n');
fprintf('  FULL SEGMENT PARAMETER SUMMARY\n');
fprintf('=================================================================\n');

% Column header
fprintf('%-28s  %-10s', 'Segment', 'OE_cost');
for pi = 1:length(param_names)
    fprintf('  %-9s', param_names{pi});
end
fprintf('\n%s\n', repmat('-', 1, 200));

% Sort segments alphabetically by tag for readability
tags     = cellfun(@(r) r.tag, all_results, 'UniformOutput', false);
[~, ord] = sort(tags);

for k = ord(:)'
    r = all_results{k};

    % Guard: p_full and well_identified may have different lengths in old
    % result files saved before the 16-parameter expansion. Pad as needed.
    if isfield(r, 'p_full')
        p = r.p_full(:);
    else
        p = nan(NP, 1);
    end
    if isfield(r, 'well_identified')
        wi = r.well_identified(:);
    else
        wi = false(NP, 1);
    end
    % Pad to NP if shorter (old files had 13 params)
    if length(p)  < NP; p  = [p;  nan(NP-length(p),  1)]; end
    if length(wi) < NP; wi = [wi; false(NP-length(wi),1)]; end

    fprintf('%-28s  %-10.2f', r.tag, r.cost_oe);
    for pi = 1:NP
        if wi(pi)
            fprintf('  %-9.4f', p(pi));     % excited, well identified
        else
            fprintf('  (%-7.4f)', p(pi));   % prior / non-excited
        end
    end
    fprintf('\n');
end

fprintf('%s\n', repmat('-', 1, 200));
fprintf('Note: values in () are prior/non-excited — not identified from this segment.\n');
fprintf('      Excited params are printed without brackets.\n');

% Gain schedule summary
fprintf('\n');
fprintf('=================================================================\n');
fprintf('  GAIN SCHEDULE SUMMARY (aggregated per regime)\n');
fprintf('=================================================================\n');
fprintf('%-8s  %-8s  %-8s  %-8s', 'Regime', 'th0_deg', 'phi0_deg', 'T0');
for pi = 1:length(param_names)
    fprintf('  %-9s', param_names{pi});
end
fprintf('\n%s\n', repmat('-', 1, 200));

for ri = 1:length(gs.regimes)
    re = gs.regimes(ri);
    sat_marker = '';
    if isfield(re, 'saturated') && re.saturated
        sat_marker = ' *SAT*';
    end
    fprintf('%-8s  %-8.2f  %-8.2f  %-8.4f%-6s', ...
            re.regime_id, rad2deg(re.theta0), rad2deg(re.phi0), re.T0, sat_marker);
    for pi = 1:length(re.p)
        fprintf('  %-9.4f', re.p(pi));
    end
    fprintf('\n');
end
fprintf('%s\n', repmat('-', 1, 200));
fprintf('* SAT = saturated thrust regime; tau_p/tau_q/Lp/Mq replaced by extrapolation\n');

fprintf('\nDone.\n');


% =============================================================================
% Local helpers
% =============================================================================

function p_agg = get_regime_aggregate(all_results, rid, np)
% Inverse-covariance weighted average of p_full across all segments for
% regime rid. Uses crb_full (16x16) and well_identified flags from result.
% Returns NaN for parameters that no segment identified well.

    segs = all_results(cellfun(@(r) strcmp(r.regime_id, rid), all_results));
    if isempty(segs)
        p_agg = [];
        return;
    end

    sum_prec   = zeros(np, np);
    sum_prec_p = zeros(np, 1);

    for k = 1:length(segs)
        r  = segs{k};

        % Use p_full and crb_full — full 16-element vectors from analyze_segment
        if ~isfield(r, 'p_full') || ~isfield(r, 'crb_full') || ~isfield(r, 'well_identified')
            fprintf('[WARN] %s missing p_full/crb_full/well_identified — skipping.\n', r.tag);
            continue;
        end

        p_vec = r.p_full;
        crb   = r.crb_full;
        wi    = r.well_identified;

        if length(p_vec) ~= np || size(crb,1) ~= np
            fprintf('[WARN] %s parameter vector length mismatch (%d vs %d) — skipping.\n', ...
                    r.tag, length(p_vec), np);
            continue;
        end

        if ~any(wi); continue; end

        % Diagonal precision contribution from well-identified params only
        prec_diag        = zeros(np, 1);
        prec_diag(wi)    = 1 ./ max(diag(crb(wi, wi)), 1e-10);

        sum_prec   = sum_prec   + diag(prec_diag);
        sum_prec_p = sum_prec_p + diag(prec_diag) * p_vec;
    end

    % Weighted average; NaN where nothing contributed
    p_agg = nan(np, 1);
    for i = 1:np
        if sum_prec(i,i) > 1e-12
            p_agg(i) = sum_prec_p(i) / sum_prec(i,i);
        end
    end
end


function [theta0, phi0, T0] = get_trim_actuals(all_results, rid)
    segs       = all_results(cellfun(@(r) strcmp(r.regime_id, rid), all_results));
    theta_vals = cellfun(@(r) r.theta0_actual, segs);
    % phi0_actual is measured during the hold phase and is unreliable for banked
    % regimes — the drone has not yet settled to the target roll before the hold
    % window starts. Use phi0_target (from the metadata JSON) instead, which is
    % the commanded trim angle and is always correct.
    phi_vals   = cellfun(@(r) r.phi0_target,   segs);
    T_vals     = cellfun(@(r) r.T0_actual,      segs);
    theta0     = mean(theta_vals);
    phi0       = mean(phi_vals);
    T0         = mean(T_vals);
end


function [A, B, v0_trim] = linearise_drone(p, theta0, phi0)
% Linearise the 9-state nonlinear model at trim (theta0, phi0).
%
% State:  x = [u_b, v_b, w_b, p, q, r, phi, theta, psi]   (9 states)
% Input:  u = [cmd_roll, cmd_pitch, cmd_yaw, cmd_thrust]    (4 inputs)
%
% 16-element parameter vector:
%   1=Xu_m  2=Xuu_m  3=Yv_m  4=Yvv_m  5=Tmax_m  6=Zw_m  7=Zww_m
%   8=Gamma1  9=tau_p  10=Gamma2  11=tau_q  12=Gamma3  13=tau_r
%   14=Lp  15=Mq  16=Nr

    Xu_m   = p(1);   Xuu_m  = p(2);
    Yv_m   = p(3);   Yvv_m  = p(4);
    Tmax_m = p(5);   Zw_m   = p(6);
    % Zww_m p(7) — nonlinear term, zero in linearisation
    % Gamma terms p(8,10,12) — zero at trim (q0=r0=0=p0)
    tau_p  = p(9);
    tau_q  = p(11);
    tau_r  = p(13);
    Lp     = p(14);   % roll-rate damping
    Mq     = p(15);   % pitch-rate damping
    Nr     = p(16);   % yaw-rate damping

    g = 9.81;

    % Trim forward velocity from pitch angle.
    % udot=0: -g*sin(theta0) + Xu_m_eff*u0 = 0  (Xu_m here is effective, includes Xuu_m*u0)
    % u0 = g*sin(theta0)/Xu_m_eff  (Xu_m_eff<0, sin(theta0)<0 for nose-down => u0>0)
    if abs(Xu_m) > 1e-6
        u0 = g * sin(theta0) / Xu_m;
    else
        u0 = 0;
    end
    w0 = 0;   % altitude hold: zero heave velocity at trim

    % Trim lateral velocity v0 (standard NED convention, positive = rightward).
    % At banked trim (phi0 ≠ 0), gravity drives steady lateral drift.
    % Trim equation (NED): 0 = g*cos(theta0)*sin(phi0) + Yv_m*v0 + Yvv_m*v0*|v0|
    % vy_b in the simulator has OPPOSITE sign (vy_b_sim = -vy_b_NED), but v0 here
    % is in NED convention so it plugs directly into A(1,6) = v0 and A(3,4) = -v0.
    rhs_v = g * cos(theta0) * sin(phi0);   % > 0 for phi0 > 0
    if abs(phi0) > deg2rad(1) && abs(Yvv_m) > 1e-6
        % Solve: |Yvv_m|*v0^2 + |Yv_m|*|v0| = |rhs_v|, take positive root then apply sign
        a_v    = abs(Yvv_m);
        b_v    = abs(Yv_m);
        disc_v = max(b_v^2 + 4*a_v*abs(rhs_v), 0);
        v0     = sign(phi0) * (-b_v + sqrt(disc_v)) / (2*a_v);
    elseif abs(phi0) > deg2rad(1) && abs(Yv_m) > 1e-6
        v0 = -rhs_v / abs(Yv_m);   % linear-only fallback
    else
        v0 = 0;
    end
    v0_trim = v0;   % expose as output for printing and storage

    % ── A matrix [9x9] ───────────────────────────────────────────────────────
    A = zeros(9);

    % Row 1: u_b
    % NOTE: Xu_m stored in p_full is the EFFECTIVE drag coefficient at trim speed
    % (Xu_m_eff = Xu_m_true + Xuu_m*|u0|). The correct Jacobian is:
    %   d(udot)/d(u) = Xu_m_true + 2*Xuu_m*|u0| = Xu_m_eff + Xuu_m*|u0|
    % So add only ONE additional Xuu_m*|u0|, not two.
    A(1,1) = Xu_m + Xuu_m*abs(u0);   % correct Jacobian using effective Xu_m
    A(1,5) = -w0;                      % Coriolis q*w (zero at trim)
    A(1,6) =  v0;                      % Coriolis r*v — nonzero for banked B/C regimes
    A(1,8) = -g*cos(theta0);           % gravity coupling through theta

    % Row 2: v_b  (vy_b_sim = -vy_b_NED — gravity and Coriolis signs flip)
    % d(vy_b_sim_dot)/d(vy_b_sim) = Yv_m + 2*Yvv_m*|v0_sim|  = Yv_m + 2*Yvv_m*|v0_NED|
    % d(vy_b_sim_dot)/d(r)   = +u0   (was -u0 in standard NED)
    % d(vy_b_sim_dot)/d(phi) = -g*cos(theta0)*cos(phi0)
    % d(vy_b_sim_dot)/d(theta) = +g*sin(phi0)*sin(theta0)
    A(2,2) = Yv_m + 2*Yvv_m*abs(v0);   % |v0_sim| = |v0_NED| = |v0|
    A(2,4) = -w0;                        % Coriolis p*w — zero at trim (w0=0)
    A(2,6) =  u0;                        % Coriolis r*u — sign flipped vs NED
    A(2,7) = -g*cos(theta0)*cos(phi0);   % gravity through phi — sign flipped vs NED
    A(2,8) =  g*sin(phi0)*sin(theta0);   % gravity through theta — sign flipped vs NED

    % Row 3: w_b
    A(3,3) = Zw_m;
    A(3,4) = -v0;   % Coriolis p*v — nonzero for banked B/C regimes (v0 > 0 for phi0 > 0)
    A(3,5) =  u0;   % Coriolis q*u
    A(3,7) = -g*cos(theta0)*sin(phi0);
    A(3,8) = -g*sin(theta0)*cos(phi0);

    % Row 4: p (roll rate) — includes Lp damping
    A(4,4) = Lp;                         % roll-rate aerodynamic damping
    % Gamma1*q*r terms are zero at trim

    % Row 5: q (pitch rate) — includes Mq damping
    A(5,5) = Mq;                         % pitch-rate aerodynamic damping
    % Gamma2*p*r terms are zero at trim

    % Row 6: r (yaw rate) — includes Nr damping
    A(6,6) = Nr;                         % yaw-rate aerodynamic damping
    % Gamma3*p*q terms are zero at trim

    % Row 7: phi
    A(7,4) = 1;
    A(7,5) = sin(phi0)*tan(theta0);
    A(7,6) = cos(phi0)*tan(theta0);

    % Row 8: theta
    A(8,5) =  cos(phi0);
    A(8,6) = -sin(phi0);

    % Row 9: psi
    cos_the = max(abs(cos(theta0)), 0.1) * sign(cos(theta0) + 1e-9);
    A(9,5) = sin(phi0) / cos_the;
    A(9,6) = cos(phi0) / cos_the;

    % ── B matrix [9x4] ───────────────────────────────────────────────────────
    B = zeros(9, 4);
    B(3,4) = -Tmax_m;   % thrust -> w_b  (NED: thrust opposes downward motion)
    B(4,1) =  tau_p;    % cmd_roll  -> p
    B(5,2) =  tau_q;    % cmd_pitch -> q  (positive after sign correction)
    B(6,3) =  tau_r;    % cmd_yaw   -> r
end