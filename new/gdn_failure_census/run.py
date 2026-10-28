#!/usr/bin/env python3
"""Stream paired, pinned pretrained GDN tests; explicit mock mode is CPU plumbing only."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import itertools
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from backend import MockEvaluator, MockTokenizer, PretrainedCausalLM, single_token_choices
from provenance import code_fingerprint, hub_revision, resolve_inputs, runtime_versions, sha256_file
from spec import CONDITIONS, make_probe, suite_size

SCHEMA = 'gdn-long-context-census.v2'


def csv_ints(text):
    out = [int(x) for x in text.split(',') if x.strip()]
    if not out or len(out) != len(set(out)) or any(x < 0 for x in out):
        raise argparse.ArgumentTypeError('expected distinct nonnegative integers')
    return out


def csv_floats(text):
    out = [float(x) for x in text.split(',') if x.strip()]
    if not out or len(out) != len(set(out)) or any(not .02 <= x <= .95 for x in out):
        raise argparse.ArgumentTypeError('expected distinct positions in [.02,.95]')
    return out


def csv_conditions(text):
    out = [x.strip() for x in text.split(',') if x.strip()]
    if not out or len(out) != len(set(out)) or any(x not in CONDITIONS for x in out):
        raise argparse.ArgumentTypeError('expected distinct conditions: ' + ','.join(CONDITIONS))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='m-a-p/340M-20B-GatedDeltaNet-pure-baseline')
    p.add_argument('--tokenizer')
    p.add_argument('--revision')
    p.add_argument('--tokenizer-revision')
    p.add_argument('--backend', choices=['hf', 'mock'], default='hf')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    p.add_argument('--lengths', type=csv_ints, default=[512,1024,2048,4096])
    p.add_argument('--seed-start', type=int, default=1000)
    p.add_argument('--seeds', type=int, default=20)
    p.add_argument('--positions', type=csv_floats, default=[.12,.5,.88])
    p.add_argument('--events', type=csv_ints, default=[8,32])
    p.add_argument('--conditions', type=csv_conditions, default=list(CONDITIONS))
    p.add_argument('--event-placement', choices=['early','late'], default='early')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--allow-extrapolation', action='store_true')
    p.add_argument('--preedit-probes', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='tokenizer + plan only, no weights')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-samples', type=int, default=0, help='whole-group cap; 0=all')
    p.add_argument('--max-input-tokens', type=int, default=0, help='planned forward-token cap including pre-edit probes; 0=uncapped')
    p.add_argument('--max-runtime-seconds', type=float, default=0, help='cumulative evaluation wall budget, checked between paired groups; 0=uncapped')
    p.add_argument('--error-policy', choices=['stop','record'], default='stop')
    p.add_argument('--log-every', type=int, default=25)
    a = p.parse_args(argv)
    if a.seeds < 1 or a.seed_start < 0 or a.log_every < 1:
        p.error('invalid seeds/log-every')
    if not math.isfinite(a.max_runtime_seconds) or min(a.max_samples, a.max_input_tokens, a.max_runtime_seconds) < 0:
        p.error('limits must be nonnegative')
    if any(x < 128 for x in a.lengths) or any(x > 8000 for x in a.events):
        p.error('lengths >=128 and events <=8000 required')
    if a.max_samples and a.max_samples < len(a.conditions):
        p.error('--max-samples must fit at least one complete condition group')
    return a


def _config(a):
    fields = ('model','tokenizer','revision','tokenizer_revision','backend','dtype','device',
              'lengths','seed_start','seeds','positions','events','conditions','event_placement',
              'allow_extrapolation','max_samples','preedit_probes','max_input_tokens',
              'max_runtime_seconds','error_policy')
    return {key: getattr(a,key) for key in fields}


def _write_json(path, obj):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def _groups(a, tok, stats=None):
    stats = stats if stats is not None else Counter()
    for length, seed, depth, n in itertools.product(a.lengths, range(a.seed_start,a.seed_start+a.seeds), a.positions,a.events):
        group = []
        reason = None
        for condition in a.conditions:
            try:
                case = make_probe(tok, condition=condition, length=length, seed=seed,
                                  needle_fraction=depth, num_events=n, event_placement=a.event_placement)
            except ValueError as exc:
                if 'required text' not in str(exc):
                    raise
                reason = 'required_text_does_not_fit'
                break
            if abs(case.needle_fraction-depth) > .10:
                reason = 'position_unattainable'
                break
            group.append(case)
        if not reason and len({x.needle_token for x in group}) != 1:
            reason = 'unequal_target_offsets'
        if reason:
            stats[reason] += 1
            continue
        yield group


def _cases(a, tok):
    for group in _groups(a,tok):
        yield from group


def forward_tokens(probe, tok, preedit):
    extra = probe.event_token + len(tok.encode(probe.question_suffix, add_special_tokens=False)) if preedit and probe.event_token is not None and probe.answerable else 0
    return probe.actual_length + extra


def plan_cases(a, tok, path):
    stats, seen, count, tokens = Counter(), set(), 0, 0
    stop_reason = None
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w',encoding='utf-8') as fh:
        for group in _groups(a,tok,stats):
            cost = sum(forward_tokens(p,tok,a.preedit_probes) for p in group)
            if a.max_samples and count+len(group) > a.max_samples:
                stop_reason = 'max_samples_whole_groups'
                break
            if a.max_input_tokens and tokens+cost > a.max_input_tokens:
                stop_reason = 'max_input_tokens_whole_groups'
                break
            for case in group:
                if case.probe_id in seen:
                    raise ValueError('duplicate probe ID')
                single_token_choices(tok, case.question_suffix)
                seen.add(case.probe_id)
                fh.write(json.dumps(case.to_meta(), ensure_ascii=False) + '\n')
            count += len(group)
            tokens += cost
    if not count:
        tmp.unlink(missing_ok=True)
        raise ValueError('empty census: no complete group fits length/position/token budget')
    os.replace(tmp,path)
    full = suite_size(a.lengths,list(range(a.seeds)),a.positions,a.events,a.conditions)
    invalid = sum(stats.values()) * len(a.conditions)
    return {'planned_cases':count,'planned_input_tokens':tokens,'full_grid_cases':full,
            'skipped_invalid_grid_cases':invalid,'invalid_group_reasons':dict(stats),
            'deferred_grid_cases':full-count-invalid,'planning_stop_reason':stop_reason,
            'plan_sha256':sha256_file(path)}


def read_jsonl(path):
    if not path.exists():
        return []
    rows=[]
    with path.open(encoding='utf-8') as fh:
        for line_no,line in enumerate(fh,1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_no}: incomplete/corrupt JSONL; preserve original before repair') from exc
    return rows


def validate_completed(plan, records, evidence_type=None):
    expected = {p['probe_id']:p for p in plan}
    if len(expected) != len(plan):
        raise ValueError('duplicate planned probe IDs')
    done = set()
    for r in records:
        pid=r.get('probe_id')
        if pid in done or pid not in expected or r.get('status') not in ('ok','error'):
            raise ValueError('resume refused: duplicate, unknown, or nonterminal result')
        if any(r.get(k) != v for k,v in expected[pid].items()):
            raise ValueError('resume refused: result metadata/prompt hash differs from frozen plan')
        if evidence_type is not None and r.get('evidence_type') != evidence_type:
            raise ValueError('resume refused: mismatched real/mock evidence type')
        if r['status']=='ok':
            if not isinstance(r.get('prediction'),str) or r['prediction'] not in tuple('ABCD') or r.get('correct') != int(r['prediction']==r['answer']):
                raise ValueError('resume refused: inconsistent prediction/score')
            for key in ('gold_conditional_nll','gold_full_nll','margin','latency_ms'):
                if not isinstance(r.get(key),(int,float)) or not math.isfinite(r[key]):
                    raise ValueError('resume refused: invalid model score')
        elif not isinstance(r.get('error_type'),str) or not isinstance(r.get('error'),str):
            raise ValueError('resume refused: error row lacks diagnostic information')
        done.add(pid)
    return done


def restore_probe(meta,tok):
    p=make_probe(tok,condition=meta['condition'],length=meta['target_length'],seed=meta['seed'],
                 needle_fraction=meta['requested_needle_fraction'],num_events=meta['num_events'],
                 event_placement=meta['event_placement'])
    if p.to_meta() != meta:
        raise ValueError('reconstructed tokens differ from frozen manifest')
    return p


def evaluate_case(p,tok,evaluator,a):
    choices=single_token_choices(tok,p.question_suffix)
    r=evaluator.score(p.token_ids,choices)
    before, pre_ids=None, None
    if a.preedit_probes and p.event_token is not None and p.answerable:
        pre_ids=p.token_ids[:p.event_token]+tuple(tok.encode(p.question_suffix,add_special_tokens=False))
        before=evaluator.score(pre_ids,choices)
    return {**p.to_meta(),'status':'ok','evidence_type':'plumbing_mock' if a.backend=='mock' else 'pretrained_lm',
            'prediction':r['prediction'],'correct':int(r['prediction']==p.answer),
            'margin':r['log_prob_choice'][p.answer]-max(v for k,v in r['log_prob_choice'].items() if k!=p.answer),
            'gold_conditional_nll':-r['log_prob_choice'][p.answer],
            'gold_full_nll':-r['log_prob_full'][p.answer],'choice_log_probs':r['log_prob_choice'],
            'latency_ms':r['latency_ms'],'peak_allocated_bytes':r['peak_allocated_bytes'],
            'peak_reserved_bytes':r.get('peak_reserved_bytes'),
            'is_context_extrapolated':bool(evaluator.context is not None and p.actual_length>evaluator.context),
            'preedit_correct':int(before['prediction']==p.initial_answer) if before else None,
            'preedit_gold_conditional_nll':-before['log_prob_choice'][p.initial_answer] if before else None,
            'preedit_prompt_tokens':len(pre_ids) if before else None,
            'preedit_latency_ms':before['latency_ms'] if before else None,
            'preedit_peak_allocated_bytes':before['peak_allocated_bytes'] if before else None}


def _execute(a):
    manifest_path, plan_path, results_path = (a.out/x for x in ('manifest.json','cases.jsonl','results.jsonl'))
    cfg, fingerprint, runtime = _config(a), code_fingerprint(), runtime_versions()
    saved = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if saved:
        if not a.resume:
            raise ValueError('output exists; use --resume or a new --out')
        if saved.get('schema') != SCHEMA or saved.get('merged_from') or saved.get('nonresumable') or saved.get('merge'):
            raise ValueError('resume requires a native v2 run, not v1 or merged output')
        if saved['config'] != cfg or saved['code_fingerprint'] != fingerprint or saved['runtime'] != runtime:
            raise ValueError('resume refused: config, inference code, or dependency versions changed')
        if saved['dry_run'] != a.dry_run or saved['plan_sha256'] != sha256_file(plan_path):
            raise ValueError('resume refused: dry-run mode or plan changed')
        resolved=saved['resolved']
        # Local checkpoints are mutable, even when their paths are unchanged.
        if a.backend=='hf' and (Path(a.model).is_dir() or (a.tokenizer and Path(a.tokenizer).is_dir())):
            if resolve_inputs(a) != resolved:
                raise ValueError('resume refused: local checkpoint/tokenizer bytes changed')
    else:
        if a.resume:
            raise ValueError('--resume requires an existing manifest')
        if plan_path.exists() or results_path.exists():
            raise ValueError('orphan case/result files exist; use a new output directory')
        resolved=resolve_inputs(a)
    if a.backend=='mock':
        print('MOCK: character units and fabricated plumbing responses; NO pretrained model evidence.',flush=True)
        tok=MockTokenizer()
    else:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(resolved['tokenizer_id'],
                revision=hub_revision(resolved['tokenizer_revision']),trust_remote_code=False)
    if not saved:
        planning=plan_cases(a,tok,plan_path)
        saved={'schema':SCHEMA,'created_utc':datetime.now(timezone.utc).isoformat(),
               'config':cfg,'backend':a.backend,'dry_run':a.dry_run,'resolved':resolved,
               'code_fingerprint':fingerprint,'runtime':runtime,**planning,
               'execution':{'status':'planned','consumed_evaluation_seconds':0.,'completed_cases':0},
               'notes':['Configured model context is not proof of training exposure.',
                        'Synthetic diagnostics are not official benchmark scores.',
                        'No-target gold labels measure leakage/chance, not answerable-task ability.',
                        'Filler is assembled in token space; task/query are never truncated.']}
        _write_json(manifest_path,saved)
    print(f"Plan: {saved['planned_cases']} cases, {saved['planned_input_tokens']:,} forward input tokens including requested pre-edit probes",flush=True)
    if a.dry_run:
        saved['execution']['status']='dry_run'
        _write_json(manifest_path,saved)
        return 0
    plan=read_jsonl(plan_path)  # only metadata, never long token sequences
    records=read_jsonl(results_path)
    completed=validate_completed(plan,records,'plumbing_mock' if a.backend=='mock' else 'pretrained_lm')
    if len(completed)==len(plan):
        saved['execution'].update(status='completed',completed_cases=len(completed))
        saved['execution'].pop('stop_reason',None)
        _write_json(manifest_path,saved)
        print('All frozen cases already attempted; errors remain recorded, not silently retried.',flush=True)
        return 0
    try:
        evaluator=(MockEvaluator(tokenizer=tok) if a.backend=='mock' else PretrainedCausalLM(
            a.model,tokenizer_id=resolved['tokenizer_id'],revision=hub_revision(resolved['model_revision']),
            tokenizer_revision=hub_revision(resolved['tokenizer_revision']),tokenizer=tok,
            device=a.device,dtype=a.dtype,allow_extrapolation=a.allow_extrapolation))
    except BaseException as exc:
        saved['execution'].update(status='failed',stop_reason='model_load_failed',error_type=type(exc).__name__,error=str(exc))
        _write_json(manifest_path,saved)
        raise
    info=evaluator.info.__dict__
    if saved.get('loaded_model') and saved['loaded_model'] != info:
        raise ValueError('resume refused: loaded model configuration changed')
    saved['loaded_model']=info
    started=time.perf_counter()
    previous=float(saved['execution'].get('consumed_evaluation_seconds',0))
    saved['execution'].update(status='running',completed_cases=len(completed))
    _write_json(manifest_path,saved)
    stop_error=None
    try:
        with results_path.open('a',encoding='utf-8') as fh:
            # Check wall budget only between COMPLETE matched groups.
            for _, metas in itertools.groupby(plan,key=lambda m:m['pair_id']):
                group=list(metas)
                if all(m['probe_id'] in completed for m in group):
                    continue
                if a.max_runtime_seconds and previous+time.perf_counter()-started >= a.max_runtime_seconds:
                    saved['execution']['status']='partial'
                    saved['execution']['stop_reason']='cumulative_wall_budget'
                    break
                for meta in group:
                    if meta['probe_id'] in completed:
                        continue
                    p=restore_probe(meta,tok)
                    try:
                        record=evaluate_case(p,tok,evaluator,a)
                    except Exception as exc:
                        record={**p.to_meta(),'status':'error','error_type':type(exc).__name__,
                                'error':str(exc),'evidence_type':'plumbing_mock' if a.backend=='mock' else 'pretrained_lm'}
                        if a.error_policy=='stop':
                            stop_error=exc
                    fh.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                    fh.flush()
                    completed.add(p.probe_id)
                    if len(completed)%a.log_every==0 or len(completed)==len(plan):
                        print(f"{len(completed)}/{len(plan)} attempted; {p.condition} {p.actual_length} tokens; {record['status']}",flush=True)
                    if stop_error:
                        raise stop_error
            else:
                saved['execution']['status']='completed'
    except BaseException:
        saved['execution']['status']='failed'
        raise
    finally:
        saved['execution']['completed_cases']=len(completed)
        saved['execution']['consumed_evaluation_seconds']=previous+time.perf_counter()-started
        _write_json(manifest_path,saved)
    return 0 if saved['execution']['status']=='completed' else 75


def main(argv=None):
    a=parse_args(argv)
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another process is writing this output; use separate shard directories') from exc
        return _execute(a)


if __name__=='__main__':
    sys.exit(main())
