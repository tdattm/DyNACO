"""Archive the actual user approval message from this Codex session log."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
from srr_p2.core import ROOT,save,sha
OUT=ROOT/'artifacts/m2_principal/v1'
def main():
    p=Path('/mnt/c/Users/TIEN DAT/.codex/sessions/2026/10/31/rollout-2026-10-31T21-37-53-01a19265-45ec-75f2-bf7b-ad905d297865.jsonl')
    found=[]
    with p.open() as f:
        for line_number,line in enumerate(f,1):
            obj=json.loads(line);v=obj.get('payload',{})
            if obj.get('type')=='response_item' and v.get('role')=='user':
                text='\n'.join(c.get('text','') for c in v.get('content',[]))
                if text.startswith('DyNACO — M2 Autonomous End-to-End Principal Experiment'):
                    found.append((obj,text,line_number,hashlib.sha256(line.encode()).hexdigest()))
    if len(found)!=1:raise ValueError('unique actual approval message not found; no fabricated substitute')
    obj,text,line_number,h=found[0]
    (OUT/'authorization_basis.txt').open('x').write(text)
    save(OUT/'authorization_basis_message.json',obj)
    save(OUT/'approval_provenance.json',dict(source='User-submitted instruction in this Codex conversation',session_id='01a19265-45ec-75f2-bf7b-ad905d297865',session_log=str(p),source_line=line_number,source_record_sha256=h,archived_at=datetime.now(timezone.utc).isoformat(),approval_basis_sha256=sha(OUT/'authorization_basis.txt'),human_identity='Not cryptographically verified; no personal name asserted'))
if __name__=='__main__':main()
