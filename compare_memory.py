import numpy as np
import csv
import sys

A = 'memory_snapshot.npz'
B = 'memory_snapshot_B.npz'

def load_npz(fn):
    d = np.load(fn, allow_pickle=True)
    print(f'Loaded {fn}, keys={list(d.keys())}')
    # try common names
    for k in ['long_memory','mem','memory','data']:
        if k in d:
            return d[k]
    # fallback: pick first array-like key
    for k in d.files:
        return d[k]
    raise RuntimeError('no arrays in npz')


def summarize_mem(mem):
    rows = []
    n_regions = len(mem)
    for i in range(n_regions):
        lm = mem[i]
        if lm is None:
            continue
        # ensure list
        try:
            entries = list(lm)
        except Exception:
            entries = [lm]
        for j, e in enumerate(entries):
            etype = e.get('type') if isinstance(e, dict) else ''
            err = e.get('error', '') if isinstance(e, dict) else ''
            weight = e.get('weight', '') if isinstance(e, dict) else ''
            ts = e.get('ts', '') if isinstance(e, dict) else ''
            tpl = e.get('template', None) if isinstance(e, dict) else None
            tpl_mean = tpl_std = ''
            tpl_sample = ''
            if tpl is not None:
                try:
                    arr = np.asarray(tpl).ravel()
                    tpl_mean = float(np.mean(arr))
                    tpl_std = float(np.std(arr))
                    tpl_sample = ','.join(map(str, arr.flatten()[:5].tolist()))
                except Exception:
                    tpl_sample = str(type(tpl))
            rows.append((i, j, etype, err, weight, ts, tpl_mean, tpl_std, tpl_sample))
    return rows


def write_csv(rows, outfn):
    hdr = ['region','entry_index','type','error','weight','ts','tpl_mean','tpl_std','tpl_sample']
    with open(outfn, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for r in rows:
            w.writerow(r)


def compare(A_mem, B_mem, out='memory_diff.txt'):
    n = min(len(A_mem), len(B_mem))
    lines = []
    for i in range(n):
        a = A_mem[i]
        b = B_mem[i]
        a_entries = list(a) if a is not None else []
        b_entries = list(b) if b is not None else []
        a_pos = next((e for e in a_entries if isinstance(e, dict) and e.get('type')=='positive'), None)
        b_pos = next((e for e in b_entries if isinstance(e, dict) and e.get('type')=='positive'), None)
        def tpl_mse(x,y):
            try:
                xa = np.asarray(x).ravel().astype(float)
                yb = np.asarray(y).ravel().astype(float)
                L = min(len(xa), len(yb))
                return float(np.mean((xa[:L]-yb[:L])**2))
            except Exception:
                return float('inf')
        if a_pos is None and b_pos is None:
            lines.append(f'Region {i}: no positive in A or B')
            continue
        if a_pos is None and b_pos is not None:
            lines.append(f'Region {i}: positive created by B (A had none)')
            continue
        if a_pos is not None and b_pos is None:
            lines.append(f'Region {i}: positive from A disappeared in B')
            continue
        # both exist
        mse = tpl_mse(a_pos.get('template'), b_pos.get('template'))
        a_err = a_pos.get('error', None)
        b_err = b_pos.get('error', None)
        a_w = a_pos.get('weight', None)
        b_w = b_pos.get('weight', None)
        status = 'unchanged'
        if mse > 1e-6:
            status = f'changed (mse={mse:.6g})'
        lines.append(f'Region {i}: positive present in both — {status}; A_err={a_err} A_w={a_w} | B_err={b_err} B_w={b_w}')
        # negatives count
        a_neg = len([e for e in a_entries if isinstance(e, dict) and e.get('type')!='positive'])
        b_neg = len([e for e in b_entries if isinstance(e, dict) and e.get('type')!='positive'])
        lines.append(f'  negatives: A={a_neg} B={b_neg}')
    with open(out,'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return lines

if __name__ == '__main__':
    A_mem = load_npz(A)
    B_mem = load_npz(B)
    A_rows = summarize_mem(A_mem)
    B_rows = summarize_mem(B_mem)
    write_csv(A_rows, 'memory_A.csv')
    write_csv(B_rows, 'memory_B.csv')
    print('Exported memory_A.csv and memory_B.csv')
    diff = compare(A_mem, B_mem, out='memory_diff.txt')
    print('\n'.join(diff))
    print('Wrote memory_diff.txt')
