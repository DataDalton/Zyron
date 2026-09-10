# CREATE ABAC POLICY

Attaches a predicate that decides which rows a reader sees. A grant controls access to the table, and this controls access to its rows. The predicate may read the row and the reader's attributes. The planner applies it as part of the query, so it narrows the scan and cannot be bypassed by rewriting the query.

## Syntax

```sql
CREATE ABAC POLICY name ON TABLE | PUBLICATION target WHERE predicate
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `ON PUBLICATION target` | Applies the rule to what a publication delivers rather than to a table's reads. | The rule applies to reads of the named table. |

## Examples

```sql
CREATE ABAC POLICY own_rows ON TABLE orders WHERE region = 'west'
```

Readers of the table see only the rows the predicate admits.

## Refused

- The target is a temporary table.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [GRANT](grant.md)
- [ALTER SECURITY MAP](alter-security-map.md)
- [ALTER COLUMN CLASSIFICATION](alter-column-classification.md)
