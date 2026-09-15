# VERIFY TABLE

Walks the commit hash chain of a table that carries one. Every entry's link is recomputed from what the entry states, so an entry rewritten anywhere in the chain is reported. The rows a commit covered are read back and rehashed for a sample of the commits by default and for every commit under rows => 'all'. The chain is then held against the anchors taken over it, so a chain that was truncated and relinked is reported alongside one that was edited. The result names the mode that ran and the number of rows it read, so a sampled pass is never read as a full one. A run takes no lock, runs at background priority and is cancellable, and every run is recorded in zyron_sys.verify.runs. Columns: table, commits_checked, rows_checked, anchors_checked, mode, intact, detail.

## Syntax

```sql
VERIFY TABLE table [FROM VERSION n] [TO VERSION n] [WITH (rows => 'all' | 'sampled', sample => n)]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `FROM VERSION n` | Starts the walk at the first commit at or above that version. | The walk starts at the chain's first entry. |
| `TO VERSION n` | Ends the walk at the last commit at or below that version. | The walk ends at the chain's head. |
| `WITH (rows => 'all')` | Reads back and rehashes the rows of every commit in range and the set a genesis entry covers, which requires MANAGE_VERIFICATION because it reads the whole table. | A sample of the commits have their rows read back, and the set a genesis entry covers is not read. |
| `WITH (sample => n)` | Reads back the rows of that many commits, spread across the range with both ends included. | The node's verify.default_sample setting decides how many. |

## Examples

```sql
VERIFY TABLE ledger
```

One row stating the commits walked, the rows the sampled commits covered, the anchors held against, and whether the chain is intact.

```sql
VERIFY TABLE ledger FROM VERSION 100 TO VERSION 200 WITH (rows => 'all')
```

One row stating that every commit between those versions was walked and every row it covered rehashed.

## Refused

- The table carries no commit chain.
- rows => 'all' is asked for without MANAGE_VERIFICATION.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [ALTER TABLE SET OPTIONS](../lake/alter-table-set-options.md)
- [LEGAL HOLD](legal-hold.md)
