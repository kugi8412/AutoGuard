#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dataset download and verification for AutoGuard."""

import argparse, logging, shutil, urllib.request
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

AUTO_DOWNLOADS = {
    'grampa': {'url': 'https://github.com/zswitten/Antimicrobial-Peptides/raw/master/data/grampa.csv', 'dest': 'raw/grampa/grampa.csv', 'description': 'GRAMPA: ~5,980 peptide-MIC pairs', 'min_size': 500000},
    'apd_natural': {'url': 'https://aps.unmc.edu/assets/sequences/naturalAMPs_APD2024a.fasta', 'dest': 'raw/apd/natural.fasta', 'description': 'APD6: 3,306 natural AMPs', 'min_size': 50000},
    'apd_human': {'url': 'https://aps.unmc.edu/assets/sequences/humanAMPs_APD2024.fasta', 'dest': 'raw/apd/human.fasta', 'description': 'APD6: 154 human host-defense peptides', 'min_size': 5000},
    'apd_animal': {'url': 'https://aps.unmc.edu/assets/sequences/animalAMPs_APD2024a.fasta', 'dest': 'raw/apd/animal.fasta', 'description': 'APD6: 2,580 animal AMPs', 'min_size': 50000},
}

MANUAL_DOWNLOADS = {
    'dramp': {'dest': 'raw/dramp/general_amps.fasta', 'description': 'DRAMP 3.0: 22,259 AMPs', 'instructions': ['curl -o data/raw/dramp/general_amps.fasta "https://dramp.cpu-bioinfor.org/down_open.php?filename=download/general_amps.fasta"']},
    
    # Zaktualizowane pliki DBAASP (CSV)
    'dbaasp_1': {'dest': 'raw/dbaasp/peptides1.csv', 'description': 'DBAASP v3: peptides1.csv', 'instructions': ['# Manually download and copy the file peptides1.csv to data/raw/dbaasp/peptides1.csv']},
    'dbaasp_2': {'dest': 'raw/dbaasp/peptides2.csv', 'description': 'DBAASP v3: peptides2.csv', 'instructions': ['# Manually download and copy the file peptides2.csv to data/raw/dbaasp/peptides2.csv']},
    'dbaasp_3': {'dest': 'raw/dbaasp/peptides3.csv', 'description': 'DBAASP v3: peptides3.csv', 'instructions': ['# Manually download and copy the file peptides3.csv to data/raw/dbaasp/peptides3.csv']},
    'dbaasp_4': {'dest': 'raw/dbaasp/peptides4.csv', 'description': 'DBAASP v3: peptides4.csv', 'instructions': ['# Manually download and copy the file peptides4.csv to data/raw/dbaasp/peptides4.csv']},
    'dbaasp_5': {'dest': 'raw/dbaasp/peptides5.csv', 'description': 'DBAASP v3: peptides5.csv', 'instructions': ['# Manually download and copy the file peptides5.csv to data/raw/dbaasp/peptides5.csv']},
    
    'toxinpred_pos': {'dest': 'raw/toxinpred/pos.txt', 'description': 'ToxinPred: 1,805 toxic peptides', 'instructions': ['curl -o data/raw/toxinpred/pos.txt https://webs.iiitd.edu.in/raghava/toxinpred/dataset_main_positive.txt']},
    'toxinpred_neg': {'dest': 'raw/toxinpred/neg.txt', 'description': 'ToxinPred: 3,593 non-toxic peptides', 'instructions': ['curl -o data/raw/toxinpred/neg.txt https://webs.iiitd.edu.in/raghava/toxinpred/dataset_main_negative.txt']},
    
    'hemopi_pos_train': {'dest': 'raw/hemopi/pos_train.fa.txt', 'description': 'HemoPI: 708 hemolytic peptides (train)', 'instructions': ['curl -o data/raw/hemopi/pos_train.fa.txt https://webs.iiitd.edu.in/raghava/hemopi/pos_train.fa.txt']},
    'hemopi_pos_test': {'dest': 'raw/hemopi/pos_test.fa.txt', 'description': 'HemoPI: 177 hemolytic peptides (test)', 'instructions': ['curl -o data/raw/hemopi/pos_test.fa.txt https://webs.iiitd.edu.in/raghava/hemopi/pos_test.fa.txt']},
    'hemopi_neg_train': {'dest': 'raw/hemopi/neg_train.fa.txt', 'description': 'HemoPI: 590 non-hemolytic peptides (train)', 'instructions': ['curl -o data/raw/hemopi/neg_train.fa.txt https://webs.iiitd.edu.in/raghava/hemopi/neg_train.fa.txt']},
    'hemopi_neg_test': {'dest': 'raw/hemopi/neg_test.fa.txt', 'description': 'HemoPI: 148 non-hemolytic peptides (test)', 'instructions': ['curl -o data/raw/hemopi/neg_test.fa.txt https://webs.iiitd.edu.in/raghava/hemopi/neg_test.fa.txt']},
    
    'iedb': {'dest': 'raw/iedb/epitope_full_v3.csv', 'description': 'IEDB: ~1.5M epitopes (for SLE experiment)', 'instructions': ['curl -o data/raw/iedb/epitope_full_v3.zip "https://www.iedb.org/downloader.php?file_name=doc/epitope_full_v3.zip"', 'unzip data/raw/iedb/epitope_full_v3.zip -d data/raw/iedb/']},
}


def download_file(url, dest, min_size=0):
    dest = Path(dest)

    if dest.exists() and dest.stat().st_size > min_size:
        logger.info(f'  Already exists: {dest} ({dest.stat().st_size:,} bytes)')
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f'  Downloading: {url}')

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'AutoGuard/0.1'})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, 'wb') as f:
            shutil.copyfileobj(resp, f)
        logger.info(f'  OK: {dest.name} ({dest.stat().st_size:,} bytes)')
        return True
    except Exception as e:
        logger.warning(f'  FAILED: {e}')

        if dest.exists():
            dest.unlink()

        return False


def setup_directories(base_dir):
    base_dir = Path(base_dir)

    for d in ['raw/grampa', 'raw/apd', 'raw/dramp', 'raw/dbaasp',
              'raw/toxinpred', 'raw/hemopi', 'raw/iedb',
              'processed', 'species_trees', 'embeddings']:
        (base_dir / d).mkdir(parents=True, exist_ok=True)


def copy_saap_trees(base_dir):
    base_dir = Path(base_dir)
    trees_dest = base_dir / 'species_trees'
    trees_dest.mkdir(parents=True, exist_ok=True)
    saap = base_dir.parent / 'SnakeAnalysisPhylogenomicsPipeline-main' / 'trees_from_sample'
    count = 0

    if saap.exists():
        for f in saap.iterdir():
            if f.suffix in ('.newick', '.nwk', '.treefile'):
                d = trees_dest / f.name
                if not d.exists():
                    shutil.copy2(f, d)
                count += 1

    samples = base_dir.parent / 'SnakeAnalysisPhylogenomicsPipeline-main' / 'config' / 'samples.csv'

    if samples.exists():
        d = trees_dest / 'species_metadata.csv'
        if not d.exists():
            shutil.copy2(samples, d)

    return count


def verify_all(base_dir):
    base_dir = Path(base_dir)
    status = {}

    for name, info in AUTO_DOWNLOADS.items():
        dest = base_dir / info['dest']
        if dest.exists() and dest.stat().st_size > info.get('min_size', 0):
            status[name] = f'OK ({dest.stat().st_size:,}B)'
        else:
            status[name] = 'MISSING'

    for name, info in MANUAL_DOWNLOADS.items():
        dest = base_dir / info['dest']
        if dest.exists() and dest.stat().st_size > 100:
            status[name] = f'OK ({dest.stat().st_size:,}B)'
        else:
            status[name] = 'MISSING (manual)'

    trees = base_dir / 'species_trees'
    n = sum(1 for p in trees.glob('*') if p.suffix in ('.newick', '.nwk', '.treefile')) if trees.exists() else 0
    status['phylo_trees'] = f'{n} trees' if n else 'MISSING'

    for f in ['amp_train.csv', 'amp_val.csv', 'amp_test.csv']:
        p = base_dir / 'processed' / f
        if p.exists():
            with open(p) as fh:
                n = sum(1 for _ in fh) - 1
            status[f] = f'{n:,} entries'
        else:
            status[f] = 'run prepare_data.py'

    return status


def main():
    parser = argparse.ArgumentParser(description='Download and verify AutoGuard datasets')
    parser.add_argument('--output_dir', default='data/')
    parser.add_argument('--auto', action='store_true', help='Auto-download GRAMPA+APD6')
    parser.add_argument('--verify', action='store_true', help='Only verify')
    parser.add_argument('--setup_only', action='store_true')
    args = parser.parse_args()
    base_dir = Path(args.output_dir)
    setup_directories(base_dir)

    if args.setup_only:
        print('Directories created.')
        return

    if args.verify:
        status = verify_all(base_dir)
        print(f"{'Dataset':<22} Status")
        print(f"{'---':<22} ---")
        for k, v in status.items():
            print(f'{k:<22} {v}')
        ok = sum(1 for v in status.values() if 'MISSING' not in v and 'run' not in v)
        print(f'\n{ok}/{len(status)} ready')
        return

    if args.auto:
        logger.info('Downloading auto datasets...')
        for name, info in AUTO_DOWNLOADS.items():
            download_file(info['url'], base_dir / info['dest'], info.get('min_size', 0))

    logger.info('Copying SAAP trees...')
    n = copy_saap_trees(base_dir)
    logger.info(f'  {n} trees copied')

    challenge = base_dir.parent / 'amp-challenge-2027-main' / 'data' / 'antibacterial.fasta'
    ref = base_dir / 'processed' / 'antibacterial_reference.fasta'
    if challenge.exists() and not ref.exists():
        shutil.copy2(challenge, ref)

    missing = [(n, i) for n, i in MANUAL_DOWNLOADS.items() if not (base_dir / i['dest']).exists()]
    if missing:
        print('\n' + '=' * 60)
        print('MANUAL DOWNLOADS NEEDED:')
        print('=' * 60)
        for name, info in missing:
            print(f'\n  [{name}] {info["description"]}')
            print(f'  Target: {info["dest"]}')
            for line in info['instructions']:
                print(f'    {line}')

    print('\nVerification:')
    status = verify_all(base_dir)
    for k, v in status.items():
        m = '+' if 'MISSING' not in v and 'run' not in v else '-'
        print(f'  [{m}] {k:<22} {v}')


if __name__ == '__main__':
    main()
