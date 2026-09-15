# Verifiable Tables

A verifiable table keeps a hash chain over its own commits. Each commit that
writes to the table adds one entry naming the rows it wrote, and each entry
links to the one before it. The sequence of entries states what the table
held at every commit, so an edit to a stored row, a removed row, a removed
commit, or a chain cut short all contradict something.

`VERIFY TABLE` walks the chain and reports what it found.

## What immutable already guarantees

`immutable` refuses UPDATE and DELETE on a table's rows, and refuses DROP and
TRUNCATE of the table itself. A row that is in an immutable table stays there,
with the bytes it was written with, for as long as the table stands.

That guarantee is the database's. It holds against every path through Zyron
and against nothing else. Someone with the data directory and a hex editor
changes a byte in a heap page and the database reads the changed row without
any sign that it changed, because nothing recorded what the row used to be.

## What the chain adds

The chain records what each commit wrote, as a hash. Recomputing that hash
from the rows the table holds now answers a question `immutable` cannot: are
these the same bytes.

One entry per committing transaction:

| Field | What it states |
| --- | --- |
| `commit_version` | The log position the rows were written at |
| `txn_id` | The transaction that wrote them, which is how they are read back |
| `rows_hash` | SHA-256 over the rows the commit wrote |
| `row_count` | How many rows that was |
| `commit_ts` | When the commit happened |
| `prev_hash` | The link of the entry before it |
| `entry_hash` | SHA-256 over all of the above except `txn_id` |

`txn_id` is left out of the link because it is the stamp this node's own rows
carry, and every member of a cluster stamps its rows with an id of its own;
what the link covers is what the rows are, not where one member keeps them.
An edited id still fails a verification, because the rows found under it are
not the rows the entry recorded.

Because `entry_hash` covers `prev_hash`, changing anything in one entry
changes every entry after it. A person who edits one row and stops there is
caught by the first recomputation. A person who edits one row and then
recomputes every hash after it produces a chain that is internally consistent.
That is what anchors are for.

## Why the anchor is what makes truncation detectable

Delete the last three commits from the table and delete their chain entries
too. What is left is a shorter chain whose every link still holds. Walking it
reports nothing wrong, because nothing in it is wrong: it is a complete,
correct chain over a table that now has fewer commits. A chain read on its own
cannot tell "this table has five commits" from "this table had eight and three
were removed".

An anchor is a record of the head the chain stood at, at a version, at an
instant. Zyron takes one on an interval, writes it into the audit log as a
`ChainAnchored` event, and holds it beside the chain. A chain that no longer
reaches the anchored commit, or that stands at a different head there,
contradicts the anchor, and `VERIFY TABLE` names the anchor it contradicts.

The same is what catches a fully rewritten chain. A forger can produce a new
head. A forger cannot produce the head the anchor already names.

## Exporting an anchor

An anchor held inside the cluster is only as good as the cluster. Whoever can
rewrite the chain can rewrite the anchor beside it.

An exported anchor is a small artifact naming a table, a version, a head hash
and an instant, signed under the release signing scheme. Held outside the
cluster, it is evidence that survives the cluster being compromised: a chain
rewritten from end to end still fails against it, because the signature covers
the head the table used to have and cannot be produced again.

For financial records, regulated retention, and anything an auditor will ask
about years later, export an anchor on a schedule and keep it somewhere Zyron
cannot reach.

## Turning it on

```sql
ALTER TABLE ledger SET (immutable = true, verified = true);
```

Both in one statement. A chain states which rows each commit wrote, so four
operations have to be refused before one is worth anything:

| Operation | Why |
| --- | --- |
| DELETE | A removed row makes the commit's count wrong |
| UPDATE | Changed bytes make the commit's hash wrong |
| TRUNCATE | Both, for every commit at once |
| A row-rewriting schema change | Re-encoded rows make every hash wrong |

The chain cannot tell any of those from tampering, so `verified` without
`immutable` is refused naming them. `immutable` refuses all four, which is
what makes it the way a table becomes verifiable.

Adding a column is **not** among them. It mints a schema epoch, every row
keeps the epoch it was written under, and the epoch is part of what the
commit hashed, so both sides of the change stay verifiable. So does widening
a column in place, which re-encodes nothing.

Turning verification off is refused once the chain holds a commit. A chain
that can be switched off is not evidence, and the chain stands as long as
the table does.

Dropping the table is a separate question. The chain states what happened
while the table existed; it does not claim the table exists forever. A drop
removes the table and its chain together, records a `VerifiedTableDropped`
audit event, and leaves any exported anchor behind as the record that the
table was there. Whether a drop is allowed at all is the table's own
protection, not something verification decides: `immutable` refuses it,
and a tenant can require two-person approval for any `DROP TABLE` through
its governance rules.

Enabling on a table that already holds rows writes a **genesis entry** over
them. The chain covers the table from the moment it became verifiable, and the
genesis entry covers the rows that were already there as one set rather than
one commit at a time. `zyron_sys.verify.tables` says so in its `covers`
column, so nobody reads a genesis as per-commit coverage.

A transaction that was already writing to the table when it became verifiable
is refused at its commit, with a message that says so, and run again it is
chained. Its rows would otherwise be covered by neither the genesis entry nor
an entry of their own. Nothing waits for it, so the statement holds up no
other commit.

## Running a verification

```sql
VERIFY TABLE ledger;
VERIFY TABLE ledger FROM VERSION 1000 TO VERSION 2000 WITH (rows => 'all');
```

The result names the table, the commits checked, the rows checked, the anchors
checked, the mode that ran, whether the chain is intact, and on a failure the
first commit that did not match and what differed: a row content mismatch, a
row count mismatch, a broken link, a missing commit, or an anchor
contradiction.

### What a sampled verification does and does not tell you

Every entry's link is recomputed whatever the mode. A chain whose entries were
edited is caught by the walk alone, and the walk is cheap: ninety-six bytes per
commit.

Reading the rows back is the expensive half, and the default reads back a
sample of the commits rather than all of them. So a sampled pass states:

- Every entry in range links correctly to the one before it.
- Every entry hashes to the link it carries.
- The chain does not contradict any anchor over it.
- The rows of the sampled commits hash to what their entries recorded.

It does **not** state that every row of every commit is unchanged. A row
altered inside a commit the sample did not read is not reported by a sampled
pass. Nor does it read back the rows a genesis entry covers, because reading
them back is reading the whole of what the table held when it became
verifiable; the entry's own link is checked, and the summary says the set was
not read.

The result always names the mode and the number of rows it read, so a sampled
pass cannot be mistaken for a full one. Raise the sample with
`WITH (sample => n)`, or read every row with `WITH (rows => 'all')`.

`rows => 'all'` reads the whole table and requires `MANAGE_VERIFICATION` on
top of `SELECT`, because a full verification of a large table is real work and
deciding to spend it is a deliberate act.

### What a verification costs the rest of the node

A verification runs at background priority under the pressure substrate. It
takes no locks on the table and reads its pages the way any other reader does,
so a writer keeps writing while one runs. It is cancellable at every page: a
cancel request from the client or `CANCEL BACKEND` stops it.

Every run is recorded in `zyron_sys.verify.runs` with who ran it, the range,
the mode, the outcome and the duration. A verification is itself evidence, so
it is on the record whether it passed or failed.

## Storage cost

A chain entry is a fixed **96 bytes** on disk. There is no per-entry framing:
a chain of n commits is exactly n records long.

| Commits per day | Per year |
| --- | --- |
| 100 | 3.5 MB |
| 1,000 | 35 MB |
| 100,000 | 3.5 GB |

The cost is per **commit**, not per row. A commit that writes one row and a
commit that writes a million rows both cost 96 bytes, because the entry holds
one hash over the rows rather than a hash per row. A table loaded in bulk
costs almost nothing to keep verifiable; a table taking one row per
transaction costs the table of figures above.

Chain bytes attribute to the table's owner in `zyron_sys.storage.attribution`
under `owner_kind` `chain`.

## Watching it

- `zyron_sys.verify.tables`: every table carrying a chain, with the head it
  stands at, when it was last anchored, when it was last verified and what
  that found, how many commits it holds and what they cost.
- `zyron_sys.verify.runs`: the verifications this node ran.
- `zyron_sys.compliance.report('audit')`: the compliance log's own chain,
  walked through the same verb.

Two alerts fire through the contact channels:

- `verify_failed`, when a verification returns not intact.
- `chain_not_anchored`, when a verified table's head has gone unanchored for
  longer than twice the anchoring interval, because a truncation of it would
  not be detected until it is anchored again.

## The audit log

`zyron_sys.compliance.log` is itself an immutable, verified table. Every
governed event Zyron records lands there, and the commit that appends one
extends the log's chain. Verifying the log is `VERIFY TABLE` against it, the
same walk over the same mechanism as any other verified table.

## Settings

| Setting | Default | What it decides |
| --- | --- | --- |
| `verify.anchor_interval_secs` | 3600 | How often a chain's head is anchored, and half the window `chain_not_anchored` fires on |
| `verify.default_sample` | 64 | Commits a verification reads the rows of when the statement names no number |
