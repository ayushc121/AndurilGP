"""
Score the vision pipeline against logged ground truth. Reads the CSV from
`collect_ground_truth.py` and reports 3D position error by range, split by
estimator. Standard library only.

    python -m analysis.accuracy [path/to/cv_ground_truth.csv]

Read the p90, not the median — the tail is what flies into a gate post.
"""

import csv
import statistics
import sys

DEFAULT_CSV = 'analysis/data/cv_ground_truth.csv'
BUCKETS = [(0, 5, '<5 m'), (5, 15, '5-15 m'), (15, 25, '15-25 m'),
           (25, 40, '25-40 m'), (40, float('inf'), '>40 m')]


def load(path):
    """Rows with a usable position error, as (range, error, per-axis, pnp_ok)."""
    rows = []
    with open(path, newline='') as handle:
        for row in csv.DictReader(handle):
            if not row.get('err_total'):
                continue
            rows.append({
                'range': float(row['range_m']),
                'err': float(row['err_total']),
                'axes': (float(row['err_x']), float(row['err_y']), float(row['err_z'])),
                'pnp': row['pnp_ok'].strip().lower() == 'true',
            })
    return rows


def percentile(values, fraction):
    """Nearest-rank percentile; avoids a numpy dependency for one number."""
    if not values:
        return float('nan')
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarise(rows, label):
    errors = [r['err'] for r in rows]
    pnp_rate = 100.0 * sum(r['pnp'] for r in rows) / len(rows)
    return (f'{label:<10}{len(rows):>7}{statistics.median(errors):>12.2f}'
            f'{percentile(errors, 0.90):>10.2f}{pnp_rate:>10.0f}%')


def main(path=DEFAULT_CSV):
    rows = load(path)
    if not rows:
        print(f'No scored rows in {path}.')
        return 1

    ranges = [r['range'] for r in rows]
    print(f'{len(rows)} frames, range {min(ranges):.1f}-{max(ranges):.1f} m\n')

    print(f'{"range":<10}{"n":>7}{"median err":>12}{"p90":>10}{"PnP":>11}')
    print('-' * 50)
    for low, high, label in BUCKETS:
        bucket = [r for r in rows if low <= r['range'] < high]
        if bucket:
            print(summarise(bucket, label))

    print(f'\n{"source":<10}{"n":>7}{"median err":>12}{"p90":>10}')
    print('-' * 39)
    for label, subset in (('PnP', [r for r in rows if r['pnp']]),
                          ('bbox', [r for r in rows if not r['pnp']])):
        if subset:
            errors = [r['err'] for r in subset]
            print(f'{label:<10}{len(subset):>7}{statistics.median(errors):>12.2f}'
                  f'{percentile(errors, 0.90):>10.2f}')

    # A non-zero median on any axis is a bias, not noise — noise cancels.
    bias = [statistics.median([r['axes'][i] for r in rows]) for i in range(3)]
    print(f'\nmedian signed error   forward {bias[0]:+.2f} m   '
          f'right {bias[1]:+.2f} m   down {bias[2]:+.2f} m')
    return 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
