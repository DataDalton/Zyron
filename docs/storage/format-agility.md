# Format Agility

Zyron guarantees that on-disk file formats can evolve without data loss and without manual migration on your part.

## What this means for you

- Upgrading Zyron never requires you to hand-run a migration script against your data.
- Files written by an older version of Zyron continue to be readable by the current version, within the supported version window.
- When a file needs to be moved to a newer format, Zyron handles it automatically according to a per-format policy.
- If a file is too old to read directly, Zyron refuses to open it with a clear error naming the intermediate version you must upgrade through first. Never silent data loss.

## Migration policies

Every persistent format Zyron writes has one of three migration policies. You can inspect the policy for any format via `zyron_sys.storage.format_registry`.

- **Eager.** After an upgrade, a background sweep converts files of the old version to the new version. Budgeted per configuration so it does not saturate your cluster. Pausable and resumable.
- **Lazy.** Files are converted the first time they are modified after upgrade. Zero background cost, but old-version files persist on disk until they are touched.
- **Coexist.** Both versions persist indefinitely. Used for immutable historical data such as snapshots, backups, and audit chain entries.

## Reader compatibility window

Zyron carries readers for the current format version and a bounded number of prior versions. Files written by any version inside that window can be read directly by the current binary. Files older than the window cannot be read directly. You must upgrade through an intermediate release first, and the upgrade path is named in the error message when this happens.

## Observability

Query these views to see format state:

- `zyron_sys.storage.format_registry`, current per-format state including which versions the running binary can read and write.
- `zyron_sys.storage.format_migrations`, in-progress format migrations with progress percentages and estimated completion time.

Live progress is also available via WebSocket subscription at `/api/upgrade/format_migrations`.

## CLI

An admin can inspect and manage format state directly:

- `zyron-ctl format inspect <file>`, reports the format kind, version, and integrity of any Zyron file.
- `zyron-ctl format migrate --format <kind> --to <version> --path <dir>`, triggers a batch migration for a specific directory.
- `zyron-ctl format verify --path <dir>`, verifies every file in a directory matches its registered format expectations.

## Related

- [../operations/auto-upgrade.md](../operations/auto-upgrade.md), how format migrations are scheduled and coordinated across an upgrade.
- [../security/signature-agility.md](../security/signature-agility.md), the parallel guarantee for cryptographic scheme rotation.
