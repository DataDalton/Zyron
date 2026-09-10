# masking_ip

Keeps the leading bits and zeroes the rest, so addresses from one network still group together while the individual host is gone. An IPv6 address accepts any prefix from 0 to 128.

## Syntax

```sql
masking_ip(ip [, keep_prefix_bits])
```

## Returns

VARCHAR. NULL when the address is NULL or does not parse.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `keep_prefix_bits` | How many leading bits to keep. | 24 for an IPv4 address. |

## Examples

```sql
SELECT masking_ip('192.168.1.55')
```

192.168.1.0.

## Refused

- Called with more than two arguments.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [masking_email](masking-email.md)
- [inet_network](inet-network.md)
- [inet_prefix](inet-prefix.md)
