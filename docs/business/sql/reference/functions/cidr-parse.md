# cidr_parse

Reads text as a CIDR network. Unlike inet_parse, bits set outside the prefix length are an error rather than being kept, so 10.0.0.1/24 is refused and 10.0.0.0/24 is accepted. Use it where a value has to be a network and not a host within one.

## Syntax

```sql
cidr_parse(text)
```

## Returns

CIDR. NULL when the argument is NULL.

## Examples

```sql
SELECT cidr_parse('10.0.0.0/24')
```

A CIDR network covering 10.0.0.0 to 10.0.0.255.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [inet_parse](inet-parse.md)
- [inet_contains](inet-contains.md)
