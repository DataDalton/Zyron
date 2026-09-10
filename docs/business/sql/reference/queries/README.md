# Shaping a Result

8 statements and constructs.

- [ASOF JOIN](asof-join.md), Joins each left row to the nearest right row in one direction along an ordered column.
- [FLATTEN](flatten.md), Walks a VARIANT document and yields seq, key, path, index, value and this per member reached.
- [LATERAL](lateral.md), Runs a FROM item once per row of the relations before it, so it may read their columns.
- [MATCH_CONDITION](match-condition.md), The inequality an ASOF JOIN matches along, and the tolerance that bounds how far a match reaches.
- [PIVOT](pivot.md), Turns the values of one column into columns, aggregating each remaining group into them.
- [UNNEST](unnest.md), Yields one row per array element. Several arrays zip to the longest, the shorter padded with NULL, and WITH ORDINALITY appends a BIGINT position starting at 1.
- [UNPIVOT](unpivot.md), Turns columns into rows, one row per named column per input row.
- [WITH ORDINALITY](with-ordinality.md), Appends a BIGINT column numbering each produced row from 1.
