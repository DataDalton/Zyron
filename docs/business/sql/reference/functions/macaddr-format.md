# macaddr_format

Writes the address as lower-case hexadecimal pairs separated by colons, so addresses read from sources using different separators compare equal as text.

## Syntax

```sql
macaddr_format(macaddr)
```

## Returns

TEXT. NULL when the argument is NULL.

## Examples

```sql
SELECT macaddr_format(macaddr_parse('08-00-2B-01-02-03'))
```

The text 08:00:2b:01:02:03.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [macaddr_parse](macaddr-parse.md)
