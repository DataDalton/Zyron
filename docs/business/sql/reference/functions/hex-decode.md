# hex_decode

Reads pairs of hexadecimal digits back into bytes, accepting upper and lower case. Text of odd length, or holding a character outside 0 to 9 and a to f, is an error rather than being read as far as it parses.

## Syntax

```sql
hex_decode(text)
```

## Returns

BYTEA, half the input length. NULL when the argument is NULL.

## Examples

```sql
SELECT hex_decode('4142')
```

The two bytes 0x41 and 0x42.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [hex_encode](hex-encode.md)
