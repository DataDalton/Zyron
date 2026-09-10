# LIST REGISTRY

Reads a registry: which signing algorithms exist, which is bound to each artifact kind, what upgrades this node has been through, which on-disk formats it can read and write, and what has been deprecated. These answer the questions an upgrade raises before it is run, which is why they are readable as statements rather than only from a file an operator would have to find.

## Syntax

```sql
LIST SIGNATURE SCHEMES | ARTIFACT SCHEMES | UPGRADE HISTORY [LIMIT n] | FORMAT REGISTRY | DEPRECATIONS
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `LIMIT n` | Reads the most recent n entries of the upgrade history rather than all of them. | The whole history is read. |

## Examples

```sql
LIST FORMAT REGISTRY
```

Every on-disk format this build knows, with the versions it reads and writes.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [SHOW UPGRADE](show-upgrade.md)
- [TRIGGER UPGRADE](trigger-upgrade.md)
