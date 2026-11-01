"""Validate externally supplied approval metadata. No approval creation API."""
from datetime import datetime,timezone
from pathlib import Path
from .common import *

def validate_records(op,generation,execution,*,attempt,amendment_hash,implementation_hash,candidate_hash,seal_hash):
    if not attempt or attempt.startswith('old-'):raise ValueError('new attempt identity required')
    for record,action in ((op,'ACCEPT_M2_OPERATIONAL_AMENDMENT'),(generation,'PERMIT_M2_DATASET_GENERATION'),(execution,'APPROVE_M2_PRINCIPAL_192')):
        if not isinstance(record,dict) or record.get('action')!=action:raise ValueError('missing/malformed approval action')
        expected=dict(attempt=attempt,amendment_sha256=amendment_hash,implementation_sha256=implementation_hash,scientific_registration_sha256=SCIENTIFIC,registration_sha256=candidate_hash)
        if any(record.get(k)!=v for k,v in expected.items()):raise ValueError('approval identity/hash mismatch')
        if not record.get('approved_by') or not record.get('basis_sha256'):raise ValueError('approval provenance missing')
        date=datetime.fromisoformat(record['approved_at'])
        if date.tzinfo is None or date>datetime.now(timezone.utc):raise ValueError('invalid approval timestamp')
    if execution.get('dataset_manifest_sha256')!=seal_hash:raise ValueError('sealed manifest approval mismatch')
    dates=[datetime.fromisoformat(v['approved_at']) for v in (op,generation,execution)]
    if dates!=sorted(dates):raise ValueError('approval gate order')
    return True

def validate_bundle(spec):
    # Invoked before output creation/claim/dataset-content loading.
    op,generation,execution=(read(spec[k]) for k in ('operational_approval','generation_permission','execution_authorization'))
    seal=read(spec['dataset_manifest'])
    validate_records(op,generation,execution,attempt=spec['attempt'],amendment_hash=sha(spec['amendment']),implementation_hash=implementation(),candidate_hash=sha(spec['registration']),seal_hash=sha(spec['dataset_manifest']))
    for record in (op,generation,execution):
        if sha(record['basis_file'])!=record['basis_sha256']:raise ValueError('actual approval basis drift')
    if seal.get('generation_permission_sha256')!=sha(spec['generation_permission']):raise ValueError('generation permission not bound to seal')
    if not (datetime.fromisoformat(generation['approved_at'])<=datetime.fromisoformat(seal['sealed_at'])<=datetime.fromisoformat(execution['approved_at'])):raise ValueError('generation/seal/execute gate timing')
    return True
