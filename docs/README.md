# Zyron Documentation

End-user and operator documentation for Zyron.

## Categorical structure

Every doc lives inside a category, never at the root. Categories are added as they emerge.

| Category | Contents |
|----------|----------|
| `storage/` | Storage behaviors visible to operators |
| `operations/` | Running the cluster, upgrades, backups, monitoring, admin surface |
| `security/` | Auth, crypto rotation, and other operator-facing security controls |
| `sql/` | SQL surface reference (functions, DDL, dialect) |
| `api/` | Zyron API and wire protocol reference |
| `drivers/` | Client driver and SDK reference |
| `admin/` | Tenant, workspace, group, and RBAC administration guides |

## Contents

### storage

- [storage/format-agility.md](storage/format-agility.md), what Zyron guarantees about on-disk file format upgrades and how migrations are surfaced.

### operations

- [operations/auto-upgrade.md](operations/auto-upgrade.md), release channels, rolling upgrade behavior, emergency controls, configuration, observability, troubleshooting.

### security

- [security/signature-agility.md](security/signature-agility.md), signature scheme rotation, currently registered schemes, DDL, overlap semantics, retirement lifecycle.
