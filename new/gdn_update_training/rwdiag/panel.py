"""Frozen v1 controls plus matched v2 entity-binding panels.

Both orders render the SAME scene, not two independent samples. Only entity/value
assignments change between worlds; world, family and order are not in the question.
"""
from dataclasses import replace
from functools import lru_cache
import random
from . import legacy_panel as v1
from .legacy_panel import ALPHA, QUESTIONS, STAGES, FORM, Scene, Item, scenario, enc, padding, choices

FAMILIES = ('bind_x', 'bind_y', 'bind_swap')
ORDERS = ('xy', 'yx')
BINDING_PANELS = tuple(f'{kind}_{order}' for kind in FAMILIES for order in ORDERS)
PANELS = v1.PANELS + BINDING_PANELS

@lru_cache(maxsize=8192)
def _balanced_column(block_id, column):
    # Separate balanced streams: old values and the fixed OTHER value are NOT
    # complements of the changed labels. Cross-entity/repeat values are allowed.
    values = list(ALPHA * 6)
    random.Random(f'bind2/{block_id}/{column}').shuffle(values)
    return tuple(values)

def binding_values(s, family, world):
    if family not in FAMILIES or world not in (0,1):
        raise ValueError('invalid binding family/world')
    namespace, raw_i = s.id.rsplit('/', 1)
    block, j = divmod(int(raw_i), 24)
    cols = [_balanced_column(f'{namespace}/{block}', field)[j]
            for field in ('old_x', 'old_y', 'fixed_x', 'fixed_y')]
    old_x, old_y, fixed_x, fixed_y = cols
    # (s.b, s.c) enumerates all 12 distinct transitions twice in each block.
    # Deliberately do NOT expose the other two permutation labels as old values.
    varied = (s.b, s.c)[world]
    if family == 'bind_x':
        return old_x, old_y, varied, fixed_y
    if family == 'bind_y':
        return old_x, old_y, fixed_x, varied
    return old_x, old_y, varied, (s.c, s.b)[world]

def render(tok, s, panel, world, stage, distance, question, donor='self'):
    if panel not in BINDING_PANELS:
        return v1.render(tok, s, panel, world, stage, distance, question, donor)
    family, order = panel.rsplit('_', 1)
    old_x, old_y, new_x, new_y = binding_values(s, family, world)
    # v1's world-0 serializer takes explicit old_x=a, old_y=d, new_x=b,
    # new_y=c. This preserves update offsets, delayed stages, donor alignment,
    # native tokenization and no-lookahead answers when Y has not updated yet.
    explicit = replace(s, a=old_x, d=old_y, b=new_x, c=new_y)
    return v1.render(tok, explicit, 'swap' if order == 'xy' else 'swap_yfirst',
                     0, stage, distance, question, donor)
