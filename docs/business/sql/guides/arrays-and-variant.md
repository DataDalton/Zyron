# Arrays and VARIANT

An array column holds many values in one row. A VARIANT column holds a document whose shape the table never declared. `UNNEST` turns the first into rows, `FLATTEN` turns the second into rows, and the array functions turn rows back into arrays.

## What this means for you

- One array of ten elements becomes ten rows without a join.
- A VARIANT document becomes one row per member, with the path to each.
- Both read the stored bytes in place. A ten-element array costs one walk over its offsets, not ten row copies.
- Anything that reads a column of the rows before it in `FROM` is written with `LATERAL`, and a statement that leaves `LATERAL` out is refused rather than run.

## UNNEST

```sql
UNNEST(array_expr [, array_expr ...]) [WITH ORDINALITY] [AS alias (col [, col ...])]
```

One array yields one column.

```sql
SELECT e FROM UNNEST(ARRAY[10, 20, 30]) AS t (e);
```

```text
 e
----
 10
 20
 30
```

Several arrays yield several columns, zipped position by position. The longest decides how many rows come out, and a shorter array pads with NULL.

```sql
SELECT a, b FROM UNNEST(ARRAY[1, 2, 3], ARRAY[10, 20, 30, 40, 50]) AS t (a, b);
```

```text
 a    | b
------+----
 1    | 10
 2    | 20
 3    | 30
 NULL | 40
 NULL | 50
```

`WITH ORDINALITY` appends a `BIGINT` column numbering the produced rows from 1. It is a column like any other, so the alias list names it too.

```sql
SELECT v, n FROM UNNEST(ARRAY['a', 'b']) WITH ORDINALITY AS t (v, n);
```

An empty array yields no rows. A NULL array yields no rows.

The produced column takes the array's element type, whatever that is: `INT[]` yields `INT`, `TEXT[]` yields `TEXT`, and `STRUCT<...>[]` yields `STRUCT` rows whose fields are addressable the way a stored struct's are. `UNNEST` needs to know that element type, so it is written over a column declared `T[]` or over an `ARRAY[...]` constructor. An expression that yields an array of no declared element type is refused, naming the argument.

## UNNEST over a column

An array held in a table is unnested per row, which is what `LATERAL` means.

```sql
SELECT o.id, item
FROM orders AS o, LATERAL UNNEST(o.items) AS u (item);
```

Each order contributes one row per element of its own `items`, carrying the order's columns beside it. An order with an empty array contributes none. To keep it anyway, write the join out and make it a `LEFT JOIN LATERAL`:

```sql
SELECT o.id, item
FROM orders AS o LEFT JOIN LATERAL UNNEST(o.items) AS u (item) ON TRUE;
```

Without `LATERAL`, an `UNNEST` that names a column of a preceding relation is refused, and the refusal names `LATERAL` and the relation:

```text
UNNEST reads a column of o, which comes before it in FROM; write LATERAL UNNEST
so it is executed once per row of o
```

That refusal exists so a plan is never silently correlated. A correlated expansion runs once per row of what precedes it, which is a different query and a different cost from one that does not, and the statement says which it is.

## FLATTEN

```sql
FLATTEN(variant_expr [, path => 'a.b[*]'] [, outer => bool] [, recursive => bool])
```

`FLATTEN` walks a document and yields one row per member it reaches. Every row carries six columns.

| Column | Type | What it holds |
| --- | --- | --- |
| `seq` | `BIGINT` | The member's position in the walk, from 1 |
| `key` | `TEXT` | The object member's name, NULL for an array element |
| `path` | `TEXT` | The path from the walk's start to this member |
| `index` | `BIGINT` | The array element's position, NULL for an object member |
| `value` | `VARIANT` | The member's own value |
| `this` | `VARIANT` | The container the member was found in |

```sql
SELECT f.key, f.path, f.value
FROM docs AS d, LATERAL FLATTEN(d.doc) AS f;
```

`path => 'a.b[*]'` starts the walk somewhere other than the document's root. A segment is an object member name, `[n]` is an array position, and `[*]` is the array itself, so `a.b[*]` walks the elements of `b`. A path the document has nothing at yields no rows.

`recursive => TRUE` descends into nested objects and arrays, depth first, so a leaf three levels down comes out with the full path to it:

```text
 path  | value
-------+-------
 a     | {"b":{"c":7}}
 a.b   | {"c":7}
 a.b.c | 7
```

Without it the walk reports the first level and stops.

`outer => TRUE` yields one row carrying a NULL `value` when the input is empty or NULL, instead of no rows. It is what keeps the input row in the result when there is nothing inside it to report, the same thing `LEFT JOIN LATERAL` does for `UNNEST`.

The `LATERAL` rule is the same as `UNNEST`'s: a `FLATTEN` naming a column of a preceding relation is written under `LATERAL`, and is refused naming `LATERAL` otherwise.

## Array functions

These take arrays apart and put them back together, so a result that came out of `UNNEST` can go back into a column.

| Function | Result |
| --- | --- |
| `array_length(arr)` | The number of elements. NULL for a NULL array, 0 for an empty one. |
| `array_position(arr, element)` | The one-based position of the first matching element, NULL when it is absent. |
| `array_contains(arr, element)` | True when the array holds the element. |
| `array_distinct(arr)` | The array with duplicates removed, keeping first-appearance order. |
| `array_sort(arr)` | The array sorted ascending, nulls last. |
| `array_slice(arr, start, length)` | A run of elements from a one-based start. |
| `array_concat(arr, arr [, ...])` | The arrays joined end to end. |
| `array_to_string(arr, delimiter [, null_text])` | The elements rendered and joined. A null element is skipped unless `null_text` is given. |
| `string_to_array(text, delimiter [, null_text])` | The text split into a `TEXT` array. A part equal to `null_text` becomes a null element. |
| `array_filter(arr, x -> predicate)` | The elements the lambda holds true for, in order. |
| `array_transform(arr, x -> expr)` | Each element replaced by the lambda's value for it. |

A NULL array gives NULL from every one of them. An empty array gives an empty array from the ones that return arrays, 0 from `array_length`, false from `array_contains`, and NULL from `array_position`.

## Lambdas

`array_filter` and `array_transform` take an expression over one element, written `x -> expr`. The name is yours; it stands for one element and is typed as the array's element type.

```sql
SELECT array_transform(prices, p -> p * 2) FROM baskets;
SELECT array_filter(scores, s -> s >= 50) FROM results;
```

`array_filter`'s lambda yields a boolean, and one that yields anything else is refused naming the type it yields.

`->` reads as a lambda arrow only in the argument list of a function that takes one. Everywhere else it is the JSON access operator it has always been, so `doc -> 'key'` is unaffected.

A lambda is not evaluated once per element. Every element of every row in a batch is gathered into one column and the body runs over that column in a single pass, so ten million rows of ten elements is one expression evaluation over a hundred million values.

## Where the work happens

`EXPLAIN` names the expansion and the column it reads:

```text
Unnest  source=t0.c3
  SeqScan table_id=7
```

```text
Flatten  source=t0.c2  path=a.b[*]  recursive=yes
  SeqScan table_id=9
```

The planner estimates how many rows an expansion produces from the source column's recorded mean width, which `ANALYZE` collects. Without statistics it assumes four elements per row.
