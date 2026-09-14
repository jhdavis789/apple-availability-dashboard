"""Collect, preserve history, validate, and publish. No raw-data deletion."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime
from recovery_merge import merge_data, merge_map

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNTIME = ROOT / 'apple-runtime'
GIT = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback/git'

def command(args, timeout=180):
    subprocess.run([str(a) for a in args], cwd=HERE, check=True, timeout=timeout)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--publish',action='store_true');ap.add_argument('--no-collect',action='store_true');args=ap.parse_args()
    lock=open(RUNTIME/'update.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ.update(GIT_ASKPASS=str(ROOT/'git-askpass.py'),GIT_TERMINAL_PROMPT='0')
    if args.publish:
        command([GIT,'fetch','origin','main'])
        command([GIT,'merge','--ff-only','origin/main'])
    # Published Git data is authoritative even after a failed local build.
    baseline={name:json.loads(subprocess.check_output([str(GIT),'show','origin/main:'+name],cwd=HERE)) for name in ['data.json','store_map.json']}
    backup=RUNTIME/'published-backups'/datetime.now().strftime('%Y%m%d_%H%M%S');backup.mkdir(parents=True)
    for name,data in baseline.items():(backup/name).write_text(json.dumps(data,separators=(',',':')))
    if not args.no_collect:command([sys.executable,HERE/'availability_matrix_csv_rest.py'],1200)
    ebay=RUNTIME/'Ebay Scrape'
    database=ebay/'ebay_data.db'
    if not database.exists() or datetime.now().timestamp()-database.stat().st_mtime>7200:
        from dotenv import dotenv_values
        env=os.environ.copy()
        env.update({k:v for k,v in dotenv_values(Path.home()/'.brain-secrets/ebay-credentials.env').items() if v is not None})
        subprocess.run([sys.executable,str(ebay/'ebay_scraper.py')],cwd=ebay,env=env,check=True,timeout=900)
    command([sys.executable,HERE/'build_data.py'],900)
    current=json.loads((HERE/'data.json').read_text())
    if not current['snapshots'] or (datetime.now()-datetime.fromisoformat(current['snapshots'][-1]['timestamp'])).total_seconds()>3600:
        raise RuntimeError('No fresh collection; refusing to publish a misleading update timestamp')
    merged=merge_data(baseline['data.json'],current)
    mapped=merge_map(baseline['store_map.json'],json.loads((HERE/'store_map.json').read_text()))
    (HERE/'data.json').write_text(json.dumps(merged,separators=(',',':')))
    (HERE/'store_map.json').write_text(json.dumps(mapped,separators=(',',':')))
    (HERE/'index.html').write_bytes((HERE/'dashboard.html').read_bytes())
    # Syntax checks run on the actual browser script; chart rendering is checked separately.
    import re
    node=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
    for i,script in enumerate(re.findall(r'<script(?:\s[^>]*)?>(.*?)</script>',(HERE/'index.html').read_text(),re.S)):
        p=RUNTIME/f'check-script-{i}.js';p.write_text(script);command([node,'--check',p])
    command([GIT,'diff','--stat'])
    if args.publish:
        command([GIT,'add','data.json','store_map.json','index.html'])
        changed=subprocess.run([str(GIT),'diff','--cached','--quiet'],cwd=HERE).returncode
        if changed==1:
            command([GIT,'-c','user.name=Jackson Davis','-c','user.email=jhdavis789@gmail.com','commit','-m','Refresh Apple observations; preserve device history'])
            command([GIT,'push','origin','HEAD:main'])
        elif changed!=0:raise RuntimeError('Unable to inspect staged changes')
    (RUNTIME/'last-success.json').write_text(json.dumps({'completed':datetime.now().isoformat(),'published':args.publish,'latest_observation':current['snapshots'][-1]['timestamp']}))

if __name__=='__main__':main()
