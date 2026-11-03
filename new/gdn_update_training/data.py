"""Balanced counterfactual update groups, assembled in token space; no truncation."""
from __future__ import annotations
import hashlib
import json
from dataclasses import asdict, dataclass

LABELS = 'ABCD'
TRANSITIONS = tuple((a, b) for a in LABELS for b in LABELS if a != b)
TEMPLATES = {'train': (0, 1), 'val': (2,), 'test': (3,)}
KINDS = ('update_a', 'update_c', 'new_only', 'repeat_new',
         'protect_a', 'protect_c', 'earliest_a', 'earliest_c')

@dataclass(frozen=True)
class Group:
    group_id: str
    entity: str
    other_entity: str
    old: str
    new: str
    other_old: str
    protected: str
    template: int
    split: str

@dataclass(frozen=True)
class Row:
    group_id: str
    kind: str
    token_ids: tuple[int, ...]
    choice_ids: dict[str, int]
    gold: str
    old: str
    new: str
    update_position: int
    update_end: int
    actual_gap: int
    prompt_hash: str


def make_group(index: int, split: str, seed: int) -> Group:
    if split not in TEMPLATES or index < 0:
        raise ValueError('invalid split/index')
    old, new = TRANSITIONS[index % 12]
    block = index // 12
    possible_old = [x for x in LABELS if x != new]
    other_old = possible_old[(possible_old.index(old) + 1 + block % 2) % 3]
    # Unique split namespaces, with no target answer embedded in the entity.
    key = hashlib.sha256(f'{split}:{seed}:{index}'.encode()).hexdigest()[:16]
    templates = TEMPLATES[split]
    return Group(f'{split}-{seed}-{index}', f'X-{key}', f'Y-{key}',
                 old, new, other_old, LABELS[(index + block) % 4],
                 templates[block % len(templates)], split)


def _encode(tok, text, bos=False):
    result = list(tok.encode(text, add_special_tokens=bos))
    if not result:
        raise ValueError('empty token encoding')
    return result


def _fill(tok, n):
    if n < 0:
        raise ValueError('negative filler budget; increase context or reduce gap')
    if n == 0:
        return []
    unit = _encode(tok, 'Archive note: the routine checklist was completed.\n')
    return (unit * ((n + len(unit) - 1) // len(unit)))[:n]


def choices_after(tok, suffix):
    base = _encode(tok, suffix)
    for pattern in (' {}', '{}', '\n{}'):
        result = {}
        for letter in LABELS:
            joined = _encode(tok, suffix + pattern.format(letter))
            if len(joined) != len(base) + 1 or joined[:-1] != base:
                break
            result[letter] = joined[-1]
        if len(result) == 4 and len(set(result.values())) == 4:
            return result
    raise ValueError('answers must be four distinct next tokens after the real suffix')


def render_group(g: Group, tok, *, length: int, gap: int) -> list[Row]:
    if length <= 0 or gap < 0:
        raise ValueError('invalid length/gap')
    headers = (
        'Read the log. Values are A, B, C, D. For the current value use the last update.\n',
        'A register stores one of A, B, C, D. Later assignments replace earlier ones.\n',
        'Review the event history. Report the requested version, not an older conflicting value.\n',
        'Track changes to the records below; the latest assignment determines the present state.\n',
    )
    def record(entity, value):
        return f'\nRecord {entity}: {value}.\n'
    def query(entity, earliest=False):
        if earliest:
            return f'\nQuestion: What was the EARLIEST recorded value of {entity}?\nAnswer:'
        phrases = ('MOST RECENT recorded value', 'current value',
                   'value after the final update', 'latest recorded value')
        return f'\nQuestion: What is the {phrases[g.template]} of {entity}?\nAnswer:'
    header = _encode(tok, headers[g.template], bos=True)
    protected = _encode(tok, record(g.other_entity, g.protected))
    old_tokens = {v: _encode(tok, record(g.entity, v)) for v in (g.old,g.other_old,g.new)}
    slot = max(map(len, old_tokens.values()))
    # Value-token boundary is explicit. Full prompt is a token-space construction,
    # like the original census; no claim that decode/re-tokenize is identical.
    up_prefix = _encode(tok, f'\nRecord {g.entity}: updated to')
    up_value = _encode(tok, ' ' + g.new)
    up_end = _encode(tok, '.\n')
    queries = {k: query(g.other_entity if k.startswith('protect') else g.entity,
                        k.startswith('earliest')) for k in KINDS}
    qt = {k: _encode(tok,q) for k,q in queries.items()}
    maxq = max(map(len, qt.values()))
    fixed = len(header) + len(protected) + slot + len(up_prefix) + len(up_value) + len(up_end)
    before = _fill(tok, length - gap - maxq - fixed)
    common = header + protected + before
    update_position = len(common) + slot + len(up_prefix) + len(up_value) - 1
    update_end = update_position + 1 + len(up_end)
    rows=[]
    for kind in KINDS:
        old = g.other_old if kind.endswith('_c') else g.old
        if kind == 'repeat_new': old = g.new
        hist = _fill(tok,slot) if kind=='new_only' else old_tokens[old] + _fill(tok,slot-len(old_tokens[old]))
        suffix_fill = gap + maxq - len(qt[kind])
        ids = common + hist + up_prefix + up_value + up_end + _fill(tok,suffix_fill) + qt[kind]
        if len(ids) != length:
            raise AssertionError('exact token budget violated')
        gold = g.protected if kind.startswith('protect') else old if kind.startswith('earliest') else g.new
        rows.append(Row(g.group_id,kind,tuple(ids),choices_after(tok,queries[kind]),gold,
                        old,g.new,update_position,update_end,suffix_fill,
                        hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
    return rows


def build_groups(tok, *, split, seed, count, length, gap):
    if count <= 0 or count % 12:
        raise ValueError('group count must be a positive multiple of 12 for exact transition balance')
    return [render_group(make_group(i,split,seed),tok,length=length,gap=gap) for i in range(count)]


def dataset_hash(groups):
    content = [[asdict(r) for r in group] for group in groups]
    return hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()
