# Zyron Engineering Documentation

Internal reference for Zyron developers. How the format and upgrade substrate is implemented, what the on-disk formats look like byte by byte, and the workflows for changing them.

Every document here has a user-facing counterpart under [../business/](../business/).

## Storage

- [storage/format-agility.md](storage/format-agility.md), the envelope every persistent file carries, per-record version tags and their reserved values, the format registry columns, the workflow for bumping a format version, and the CI checks that enforce migration coverage.
- [storage/formats/zyr-spec.md](storage/formats/zyr-spec.md), the `.zyr` columnar file format.
- [storage/formats/zyridx-spec.md](storage/formats/zyridx-spec.md), the `.zyridx` index checkpoint format.
- [storage/online-ddl.md](storage/online-ddl.md), schema epochs and the slot field they live in, absent values and dropped placeholders, the publish-wait-scan-load-flip sequence and why the wait is what makes it correct, the shadow rewrite's dual-write hook and catch-up, the bulk tree build, and epoch retirement.

## Operations

- [operations/auto-upgrade.md](operations/auto-upgrade.md), the compatibility gate, upgrade coordination across nodes, and health baseline mechanics.

## Security

- [security/signature-agility.md](security/signature-agility.md), scheme registration, verifier dispatch, and the retention sweep for long-lived signed artifacts.
