# macaddr_oui

Returns the organisationally unique identifier, the first three bytes that the IEEE assigns to a manufacturer. Addresses sharing an OUI came from the same vendor, which groups devices without a device inventory.

## Syntax

```sql
macaddr_oui(macaddr)
```

## Returns

TEXT. NULL when the argument is NULL.

## Examples

```sql
SELECT macaddr_oui(macaddr_parse('08:00:2b:01:02:03'))
```

The text 08:00:2b.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [macaddr_parse](macaddr-parse.md)
- [macaddr_format](macaddr-format.md)
