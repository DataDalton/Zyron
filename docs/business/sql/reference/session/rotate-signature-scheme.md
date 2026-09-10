# ROTATE SIGNATURE SCHEME

Moves an artifact kind to a new signing algorithm. The previous algorithm continues to verify for the overlap, so artifacts already issued remain valid. Set the overlap to exceed the lifetime of the longest-lived artifact of that kind.

## Syntax

```sql
ROTATE SIGNATURE SCHEME kind TO 'scheme' OVERLAP 'duration'
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `OVERLAP 'duration'` | How long the previous algorithm stays able to verify, so artifacts already issued keep working. | Not applicable. |

## Examples

```sql
ROTATE SIGNATURE SCHEME JWT TO 'ML-DSA-65' OVERLAP '24h'
```

New tokens use the new algorithm and tokens already issued verify for another day.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [SET SIGNATURE SCHEME](set-signature-scheme.md)
- [ROTATE SERVICE PRINCIPAL KEY](rotate-service-principal-key.md)
