import os
import pandas as pd
import numpy as np


DESIRED_COLUMNS = [
    'MIDI','dB','Total','Clarity','Crest','SpecBal','CPP','Entropy',
    'dEGGmax','Qcontact','Icontact','HRFegg','maxCluster',
    'Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5',
    'maxCPhon','cPhon 1','cPhon 2','cPhon 3','cPhon 4','cPhon 5'
]


def load_base_vrp(base_csv: str) -> pd.DataFrame:
    df = pd.read_csv(base_csv)
    # normalize column names
    rename_map = {
        'frequencies': 'MIDI',
        'SPLs': 'dB',
        'Total': 'Total',
        'clarities': 'Clarity',
        'crests': 'Crest',
        'SBs': 'SpecBal',
        'CPPs': 'CPP',
        'qdeltas': 'dEGGmax',
        'qcis': 'Qcontact',
    }
    df = df.rename(columns=rename_map)
    # ensure numeric types
    for c in ['MIDI','dB','Total','Clarity','Crest','SpecBal','CPP','dEGGmax','Qcontact']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # add missing Entropy as NaN placeholder (not implemented yet)
    if 'Entropy' not in df.columns:
        df['Entropy'] = np.nan
    # order partial columns we have
    present = [c for c in ['MIDI','dB','Total','Clarity','Crest','SpecBal','CPP','Entropy','dEGGmax','Qcontact'] if c in df.columns]
    df = df[present]
    return df


def load_template_tail(template_csv: str) -> pd.DataFrame:
    # try semicolon first (FonaDyn format), fall back to comma
    try:
        tdf = pd.read_csv(template_csv, sep=';')
    except Exception:
        tdf = pd.read_csv(template_csv)
    # normalize column names (strip spaces)
    tdf.columns = [c.strip() for c in tdf.columns]
    # unifying key names
    col_lower = {c.lower(): c for c in tdf.columns}
    midi_col = col_lower.get('midi') or col_lower.get('midi ') or 'MIDI'
    db_col = None
    for k in ['db','dB','Db']:
        if k.lower() in col_lower:
            db_col = col_lower[k.lower()]
            break
    if db_col is None:
        db_col = 'dB'
    # select from Icontact onward if present
    if 'Icontact' in tdf.columns:
        start_idx = list(tdf.columns).index('Icontact')
        tail_cols = tdf.columns[start_idx:]
    else:
        # if not present, return empty
        tail_cols = []
    keys = [midi_col, db_col]
    cols_to_keep = keys + list(tail_cols)
    tdf = tdf[cols_to_keep].copy()
    # rename keys to standard
    tdf = tdf.rename(columns={midi_col: 'MIDI', db_col: 'dB'})
    # ensure numeric keys
    tdf['MIDI'] = pd.to_numeric(tdf['MIDI'], errors='coerce')
    tdf['dB'] = pd.to_numeric(tdf['dB'], errors='coerce')
    return tdf


def build_final_vrp(base_csv: str,
                    template_csv: str,
                    output_csv: str) -> pd.DataFrame:
    base = load_base_vrp(base_csv)
    templ_tail = load_template_tail(template_csv)

    # merge on (MIDI, dB)
    out = base.merge(templ_tail, on=['MIDI','dB'], how='left')

    # ensure all desired columns exist
    for c in DESIRED_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan

    out = out[DESIRED_COLUMNS]
    # 将空值统一写成 0（包括空字符串和 NaN）
    for c in DESIRED_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out = out.fillna(0)

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    out.to_csv(output_csv, index=False, sep=';', float_format='%.6g')
    return out


def compare_keysets(output_csv: str, template_csv: str, out_dir: str) -> None:
    """Compare (MIDI,dB) pairs between our output and template; write three lists."""
    # load output (semicolon)
    out_df = pd.read_csv(output_csv, sep=';')
    out_keys = out_df[['MIDI','dB']].dropna()
    out_keys['MIDI'] = pd.to_numeric(out_keys['MIDI'], errors='coerce')
    out_keys['dB'] = pd.to_numeric(out_keys['dB'], errors='coerce')
    out_set = set(map(tuple, out_keys[['MIDI','dB']].astype(int).values))

    # load template tail (or full) and normalize
    try:
        tpl_df = pd.read_csv(template_csv, sep=';')
    except Exception:
        tpl_df = pd.read_csv(template_csv)
    tpl_df.columns = [c.strip() for c in tpl_df.columns]
    midi_col = 'MIDI'
    db_col = 'dB' if 'dB' in tpl_df.columns else ('db' if 'db' in tpl_df.columns else tpl_df.columns[1])
    tpl_keys = tpl_df[[midi_col, db_col]].copy()
    tpl_keys.columns = ['MIDI','dB']
    tpl_keys['MIDI'] = pd.to_numeric(tpl_keys['MIDI'], errors='coerce')
    tpl_keys['dB'] = pd.to_numeric(tpl_keys['dB'], errors='coerce')
    tpl_set = set(map(tuple, tpl_keys[['MIDI','dB']].astype(int).values))

    overlap = sorted(out_set & tpl_set)
    only_out = sorted(out_set - tpl_set)
    only_tpl = sorted(tpl_set - out_set)

    os.makedirs(out_dir or '.', exist_ok=True)
    pd.DataFrame(overlap, columns=['MIDI','dB']).to_csv(
        os.path.join(out_dir, 'overlap_keys.csv'), index=False, sep=';')
    pd.DataFrame(only_out, columns=['MIDI','dB']).to_csv(
        os.path.join(out_dir, 'only_in_output.csv'), index=False, sep=';')
    pd.DataFrame(only_tpl, columns=['MIDI','dB']).to_csv(
        os.path.join(out_dir, 'only_in_template.csv'), index=False, sep=';')


if __name__ == '__main__':
    base_csv = 'results/VRP_long.csv'
    template_csv = 'audio/test_Voice_EGG_VRP.csv'
    output_csv = 'results/VRP_long_VRP.csv'
    df = build_final_vrp(base_csv, template_csv, output_csv)
    print(f'Wrote: {output_csv} rows={len(df)}')
    # write comparison lists for (MIDI,dB)
    compare_keysets(output_csv, template_csv, os.path.dirname(output_csv))
    print('Wrote: results/overlap_keys.csv, only_in_output.csv, only_in_template.csv')


