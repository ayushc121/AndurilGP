"""
Scores the gate-relative estimator against logged ground truth.

    python -m analysis.ekf_accuracy [csv]

HEADLINE NUMBERS ARE FILTERED TO anchor_is_active=1.
The logger anchors each row to the true gate NEAREST the EKF-implied gate position.
That convention is what stops re-anchor/index-lag transition windows from
contaminating the score — but it also makes an UNFILTERED perr a min-over-gates
lower bound, which flatters the estimate (the EKF is always scored against
whichever gate it happens to be closest to). Restricting to rows where the
anchor IS the active race gate removes that flattery. Unfiltered numbers are
printed too, explicitly labelled as the optimistic bound — do not quote them alone.
"""

import csv
import glob
import math
import sys


def _f(row, key):
    v = row.get(key, '')
    if v is None or v == '':
        return None
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except ValueError:
        return None


def _stats(vals):
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    return {
        'n': n, 'mean': mean, 'median': s[n // 2],
        'p90': s[min(n - 1, int(0.90 * n))], 'max': s[-1],
    }


def _line(label, st, unit='m'):
    if not st:
        print(f'  {label:<26} —  (no data)')
        return
    print(f'  {label:<26} median {st["median"]:6.3f} {unit}   '
          f'mean {st["mean"]:6.3f}   p90 {st["p90"]:6.3f}   '
          f'max {st["max"]:7.3f}   n={st["n"]}')


def _verdict(label, ekf, base, unit):
    """EKF-vs-baseline comparison that degrades honestly: no divide-by-~0 blowups,
    and a tie is reported as a tie rather than as a loss."""
    if not ekf or not base:
        print(f'  {label} : —  (no data)')
        return
    e, b = ekf['median'], base['median']
    EPS = 1e-4
    if e < EPS and b < EPS:
        print(f'  {label} : both ~0 (degenerate/synthetic data — not meaningful)')
    elif abs(e - b) < EPS:
        print(f'  {label} : TIE ({e:.3f} {unit})')
    elif e < b:
        ratio = b / max(e, EPS)
        extra = '  (baseline ~0 — ratio not meaningful)' if b < EPS else ''
        print(f'  {label} : {ratio:6.2f}x better  ({e:.3f} vs {b:.3f} {unit}){extra}')
    else:
        print(f'  {label} : WORSE  ({e:.3f} vs {b:.3f} {unit})')


def report(rows, title):
    print(f'\n=== {title}  ({len(rows)} rows) ===')
    if not rows:
        print('  (none)')
        return

    # forward is split out: anisotropic R makes it the weak axis, and an
    # aggregate would hide that behind the tight lateral/vertical ones
    print('POSITION error (EKF vs truth, anchor gate):')
    _line('total', _stats([v for v in (_f(r, 'perr_total') for r in rows) if v is not None]))
    for key, lab in (('perr_fwd', 'forward  (WEAK AXIS)'), ('perr_right', 'right'),
                     ('perr_down', 'down')):
        _line(lab, _stats([abs(v) for v in (_f(r, key) for r in rows) if v is not None]))

    # VELOCITY — the milestone output.
    print('VELOCITY error (EKF vs truth):')
    _line('total', _stats([v for v in (_f(r, 'verr_total') for r in rows) if v is not None]), 'm/s')
    for key, lab in (('verr_fwd', 'forward  (WEAK AXIS)'), ('verr_right', 'right'),
                     ('verr_down', 'down')):
        _line(lab, _stats([abs(v) for v in (_f(r, key) for r in rows) if v is not None]), 'm/s')

    # BASELINES — the §8 success bar: must beat raw strapdown and raw CV.
    print('BASELINES (success bar = EKF beats both):')
    sd_p = _stats([v for v in (_f(r, 'sd_perr_total') for r in rows) if v is not None])
    sd_v = _stats([v for v in (_f(r, 'sd_verr_total') for r in rows) if v is not None])
    cv_v = _stats([v for v in (_f(r, 'cv_verr_total') for r in rows) if v is not None])
    _line('strapdown pos', sd_p)
    _line('strapdown vel', sd_v, 'm/s')
    _line('raw CV vel', cv_v, 'm/s')

    ekf_p = _stats([v for v in (_f(r, 'perr_total') for r in rows) if v is not None])
    ekf_v = _stats([v for v in (_f(r, 'verr_total') for r in rows) if v is not None])
    print('VERDICT:')
    _verdict('position vs strapdown', ekf_p, sd_p, 'm')
    _verdict('velocity vs strapdown', ekf_v, sd_v, 'm/s')
    _verdict('velocity vs raw CV   ', ekf_v, cv_v, 'm/s')


def range_scaling_check(rows):
    """First-run sanity check: perr must NOT grow proportional to range.
    If it does, that is a sign/transpose error in the R_wb chain, not noise."""
    print('\n=== SANITY: does position error scale with range? ===')
    buckets = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 999)]
    for lo, hi in buckets:
        vals = [_f(r, 'perr_total') for r in rows
                if (_f(r, 'range_m') or -1) >= lo and (_f(r, 'range_m') or -1) < hi]
        vals = [v for v in vals if v is not None]
        st = _stats(vals)
        _line(f'range {lo}-{hi} m', st)
    print('  ^ roughly FLAT across buckets = frame chain OK.')
    print('    Growing ~proportional to range = R_wb sign/transpose bug.')


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        cands = sorted(glob.glob('analysis/data/ekf_validation.csv'))
        if not cands:
            print('no ekf_val_*.csv found — run: python main.py --vq1')
            return 1
        path = cands[-1]
    print(f'reading {path}')
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print('empty log')
        return 1

    active = [r for r in rows if r.get('anchor_is_active') == '1']
    report(active, 'HEADLINE — anchor_is_active=1 (quote THESE numbers)')
    report(rows, 'UNFILTERED — optimistic lower bound, do NOT quote alone')
    range_scaling_check(active or rows)

    # operational counters
    n_re = sum(1 for r in rows if r.get('reanchored') == '1')
    n_acc = sum(1 for r in rows if r.get('accepted') == '1')
    n_dark = sum(1 for r in rows if r.get('vision_valid') == '0')
    print(f'\ncounters: rows={len(rows)}  active-anchor={len(active)}  '
          f'accepted={n_acc}  reanchors={n_re}  dark(blind-window)={n_dark}')
    # re-anchors should coincide with a CHANGING anchor gate; if the anchor id is
    # unchanged across them the handoff logic is firing on the same gate.
    ids = [r.get('anchor_gate_id') for r in rows if r.get('reanchored') == '1']
    if ids:
        print(f'  re-anchor gate ids (should progress): {ids[:12]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
