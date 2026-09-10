# isbn_format

Validates the ISBN, converts it to the requested form and inserts hyphens at the standard group boundaries. A version other than 10 or 13 gives NULL.

## Syntax

```sql
isbn_format(isbn, version)
```

## Returns

VARCHAR. NULL when either argument is NULL, the ISBN is invalid, or the version is not 10 or 13.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `version` | 10 or 13. | Not applicable. |

## Examples

```sql
SELECT isbn_format('9780306406157', 13)
```

978-0-30640-615-7.

## Refused

- The isbn argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [isbn_to_13](isbn-to-13.md)
