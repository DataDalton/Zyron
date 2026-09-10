# json_get_path_text

Walks the same path json_get_path walks and returns the value unquoted. A path reaching an object or an array comes back as its JSON, because neither has a text form of its own.

## Syntax

```sql
json_get_path_text(json, path)
```

## Returns

TEXT. NULL when the value is NULL or the path reaches nothing.

## Examples

```sql
SELECT json_get_path_text('{"a":{"b":"x"}}', 'a', 'b')
```

x.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [json_get_path](json-get-path.md)
- [json_get_text](json-get-text.md)
