# ROTATE SERVICE PRINCIPAL KEY

Issues a new signing key for a principal and keeps the previous one valid for an overlap, so anything holding the old key keeps working while it is replaced. Rotating without an overlap is what turns a key change into an outage: every holder of the old key fails at once, and they do not all update at the same moment.

## Syntax

```sql
ROTATE SERVICE PRINCIPAL KEY name SCHEME 'scheme' OVERLAP 'duration'
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `OVERLAP 'duration'` | How long the previous key stays valid, so holders have time to take the new one. | Not applicable. |

## Examples

```sql
ROTATE SERVICE PRINCIPAL KEY sp1 SCHEME 'ML-DSA-65' OVERLAP '24h'
```

A new key is issued and the previous one stays valid for another day.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [ROTATE SIGNATURE SCHEME](rotate-signature-scheme.md)
- [SET SIGNATURE SCHEME](set-signature-scheme.md)
