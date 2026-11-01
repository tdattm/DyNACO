# R3 final end-to-end

**Completed: 192/192 valid principal runs. Scientific and investment decision:
NO-GO under the prespecified 8-second screening criteria.**

Read [the Vietnamese final research report](../artifacts/r3_final_end_to_end/v1/FINAL_RESEARCH_REPORT.md)
and [executive decision](../artifacts/r3_final_end_to_end/v1/EXECUTIVE_DECISION.md).
Raw trajectories, the sealed independent dataset, protocol, hashes, eight PNG/SVG
figures and complete instance-level statistics are in that versioned directory.

The native change affects only SRR candidate access. Sampling, neural input and
pheromone stay K32; every arm has exactly one original SRR pass. All historical
Stage A/M1/M2 evidence and the original working checkout are preserved.

From PowerShell, reproduce analysis and figures without new solver execution:

```powershell
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO-r3-final/r3/run.sh -m r3_final.analyze
```

Full commands and provenance are in the archived `reproducibility/commands.md`.
The archive includes both exact tested native binaries compressed with gzip.
The sealed principal dataset must not be regenerated or overwritten.
