# inet_format

Writes the address as text. IPv6 is written in its compressed form, with the longest run of zero groups replaced by a double colon, so one address has one spelling.

## Syntax

```sql
inet_format(inet)
```

## Returns

TEXT. NULL when the argument is NULL.

## Examples

```sql
SELECT inet_format(inet_parse('10.0.0.7/24'))
```

The text 10.0.0.7/24.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [inet_parse](inet-parse.md)
- [macaddr_format](macaddr-format.md)
