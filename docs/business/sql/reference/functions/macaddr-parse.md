# macaddr_parse

Reads a six-byte or eight-byte hardware address, accepting colon, hyphen or dot separators and either case. Text that is not an address is an error.

## Syntax

```sql
macaddr_parse(text)
```

## Returns

MACADDR. NULL when the argument is NULL.

## Examples

```sql
SELECT macaddr_parse('08:00:2b:01:02:03')
```

A MACADDR value.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [macaddr_format](macaddr-format.md)
- [macaddr_oui](macaddr-oui.md)
