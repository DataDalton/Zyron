# FLATTEN

Produces one row per member of a VARIANT document. Six columns: `seq` numbers the members of one document from 1, `key` holds an object member's name and is NULL for an array element, `path` holds the full route to the member as text, `index` holds an array element's position and is NULL for an object member, `value` holds the member, and `this` holds the container the member was found in. A document whose root is a scalar has no members and produces no rows.

## Syntax

```sql
FLATTEN(variant_expr [, path => 'a.b[*]'] [, outer => bool] [, recursive => bool])
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `path => 'a.b[*]'` | Starts the walk at a position inside the document rather than at its root. | The walk starts at the root. |
| `outer => bool` | Yields one row with a null value for a document that has no members, instead of no rows. | A document with no members yields nothing. |
| `recursive => bool` | Walks nested arrays and objects depth first rather than stopping at the first level. | Only the members of the starting container are reached. |

## Examples

```sql
SELECT f.path, f.value FROM FLATTEN(PARSE_JSON('{"a": 1}')) AS f
```

One row per member of the document, with the route to it and the value it holds.

## Refused

- The argument is not a document.
- It reads a column of a relation written earlier in FROM without LATERAL.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [UNNEST](unnest.md)
- [LATERAL](lateral.md)
