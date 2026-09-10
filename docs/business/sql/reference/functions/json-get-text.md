# json_get_text

Reads the same member json_get reads and returns it unquoted, so a string member comes back as its contents. A member that is an object or an array has no text form of its own and comes back as its JSON.

## Syntax

```sql
json_get_text(json, key)
```

## Returns

TEXT. NULL when the value is NULL or holds no such member.

## Examples

```sql
SELECT json_get_text('{"a":"x"}', 'a')
```

x, without the quotes json_get would keep.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [json_get](json-get.md)
- [json_get_path_text](json-get-path-text.md)
