"""Preregistered conservative simulator-only selection; no cross-shape pooling."""
import math

def select(result, expected_ids):
    if result.get('status') != 'passed' or len(result.get('analysis',[])) != 3:
        return {'selected':'baseline','reason':'No complete qualifying comparison'}
    if len(expected_ids)!=3 or len(set(expected_ids))!=3 or {r.get('fixture') for r in result['analysis']}!=set(expected_ids):
        raise ValueError('Exact three fixture IDs required')
    rows=[]
    for row in result['analysis']:
        values=row['order_A_over_B']
        if set(values) != {'ABBA','BAAB'} or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in values.values()):
            raise ValueError('Invalid order ratio')
        rows.append({'fixture':row['fixture'],'order_A_over_B':values,
                     'variant_regression_percent':{o:(1/v-1)*100 for o,v in values.items()}})
    # A=baseline, B=variant. Require both orders at every shape to avoid a
    # material regression, and a >1% tick reduction at one shape in both orders.
    safe=all(v>=1/1.01 for row in rows for v in row['order_A_over_B'].values())
    agreement=all((row['order_A_over_B']['ABBA']>=1)==(row['order_A_over_B']['BAAB']>=1) for row in rows)
    improved=any(all(v>1/0.99 for v in row['order_A_over_B'].values()) for row in rows)
    return {'selected':'variant' if safe and improved and agreement else 'baseline',
            'reason':'All shapes within 1% and one shape more than 1% faster in both orders' if safe and improved and agreement else 'Tie, regression, or insufficient improvement: retain baseline',
            'per_shape':rows, 'evidence_level':'CaModel warmed ticks only; not hardware performance'}
