# CREATE CDC STREAM

Sends a table's changes to a sink as they are committed, rather than on a schedule. What arrives is the change itself, so a consumer applies rows rather than comparing snapshots. Delivery reads a change stream and moves its position once the sink has taken the records, so where delivery has got to is the stream's position in zyron_sys.cdc.change_streams. Named with ON, the stream is created with the outbound stream, named after it, at the table's current version, and dropped with it. Named with FROM CHANGE STREAM, an existing stream is consumed from where it stands and stays when the outbound stream is dropped.

## Syntax

```sql
CREATE CDC STREAM name ON [TABLE] table | FROM CHANGE STREAM stream TO sink [WITH (option = value, ...)]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `ON [TABLE] table` | Delivers the table's changes through a change stream created for this outbound stream, named __cdc_ followed by the outbound stream's name. | Not applicable. |
| `FROM CHANGE STREAM stream` | Delivers from a change stream that exists, starting at its position. The stream must read one table. | Not applicable. |
| `WITH (option = value, ...)` | Options the sink reads, such as where to write and how to batch. | The sink's own defaults apply. |

## Examples

```sql
CREATE CDC STREAM s ON orders TO kafka
```

The table's committed changes arrive at that sink, delivered from a change stream named __cdc_s.

```sql
CREATE CDC STREAM s FROM CHANGE STREAM order_changes TO webhook WITH (url = 'https://example.test/changes')
```

The changes past the stream's position arrive at that sink, and the stream's position moves as they do.

## Refused

- The table has no change data feed.
- The change stream named reads more than one table.
- A change stream already holds the name the outbound stream would create.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP CDC STREAM](drop-cdc-stream.md)
- [CREATE CDC INGEST](create-cdc-ingest.md)
- [CREATE CHANGE STREAM](create-change-stream.md)
