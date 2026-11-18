"""Download the exact public sources used in the submitted data audit."""
from pathlib import Path
import json,subprocess
ROOT=Path(__file__).resolve().parents[1]
audit=json.loads((ROOT/'data_info/audit.json').read_text())
for url,path,commit in [('https://github.com/DMIRLAB-Group/Dr.ECI.git',ROOT/'upstream',audit['upstream_commit']),
                        ('https://github.com/tommasoc80/EventStoryLine.git',ROOT/'data_raw/EventStoryLine',audit['esc_commit'])]:
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone',url,str(path)],check=True)
        subprocess.run(['git','-C',str(path),'checkout',commit],check=True)
    actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if actual!=commit:raise SystemExit(f'{path} has different commit; use a clean directory to restore pinned source')
    print(path.name,actual)
