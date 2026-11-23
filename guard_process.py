#!/usr/bin/env python3
"""Run a process group with 250ms unified-memory protection and a JSON audit."""
import argparse, datetime, json, os, signal, subprocess, time
from pathlib import Path

def available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError('MemAvailable missing')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--min-available-gib',type=float,default=20)
    p.add_argument('--audit',default='logs/memory_guard.json')
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args(); cmd=a.command
    if cmd and cmd[0]=='--': cmd=cmd[1:]
    if not cmd: p.error('command required after --')
    initial=available_gib()
    if initial<a.min_available_gib: raise SystemExit(f'Insufficient available memory: {initial:.2f} GiB')
    t=time.perf_counter(); child=subprocess.Popen(cmd,start_new_session=True)
    minimum=initial; killed=False
    try:
        while child.poll() is None:
            current=available_gib(); minimum=min(minimum,current)
            if current<a.min_available_gib:
                killed=True
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL)
                break
            time.sleep(.25)
        code=child.wait()
    except BaseException:
        if child.poll() is None: os.killpg(child.pid,signal.SIGTERM)
        raise
    finally:
        out=Path(a.audit);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(dict(timestamp=datetime.datetime.now().astimezone().isoformat(),command=cmd,elapsed_s=time.perf_counter()-t,min_available_gib=minimum,threshold_gib=a.min_available_gib,killed=killed,returncode=child.poll()),indent=2)+'\n')
    raise SystemExit(code if not killed else 137)
if __name__=='__main__': main()
