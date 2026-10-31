import json
from pathlib import Path
from r3.common import ROOT,sha256,save_json
PATH=ROOT/'r3/protocols/stage_a_v1.json'
def load():
    record=json.loads((ROOT/'artifacts/r3/manifests/protocol_frozen.json').read_text())
    if sha256(PATH)!=record['sha256']:raise AssertionError('Frozen protocol changed; new version required')
    return json.loads(PATH.read_text())
if __name__=='__main__':
    p=json.loads(PATH.read_text())
    save_json(ROOT/'artifacts/r3/manifests/protocol_frozen.json',dict(path=str(PATH),sha256=sha256(PATH),
        frozen_before_principal=True,principal_data_generated=False,version=p['version']))
    print(sha256(PATH))
