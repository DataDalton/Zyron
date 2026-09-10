# COPY

Moves rows between a table and the connection's own stream, in bulk and without a statement per row. FROM STDIN loads and TO STDOUT unloads. The format says how the bytes are read or written, and a column list narrows which columns take part.

## Syntax

```sql
COPY name [(col, ...)] FROM | TO STDIN | STDOUT | path | backend uri FORMAT fmt [CREDENTIALS (k=v, ...)] | EXTERNAL SOURCE | SINK name
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `(col, ...)` | Names the columns taking part, in the order the stream holds them. | Every column takes part, in the table's own order. |
| `FORMAT fmt` | Says how the bytes of an external destination are read or written. | A stdio endpoint uses the connection's own format. |

## Examples

```sql
COPY orders TO STDOUT
```

Every row of the table written to the connection.

## On a cluster

The rows this statement writes are captured and replicated, so every member ends up with what it produced rather than running it again. A value like now() or a sequence draw therefore reads the same on every member.

## See also

- [INSERT](insert.md)
