# ZyronLake

19 statements and constructs.

- [ALTER TABLE CLUSTER BY](alter-table-cluster-by.md), Sets the keys a lake table's rows are arranged by.
- [ALTER TABLE CLUSTERING SCHEDULE](alter-table-clustering-schedule.md), Says when a table's clustering pass runs.
- [ALTER TABLE FOLLOW](alter-table-follow.md), Keeps a table current with a table on another cluster.
- [ALTER TABLE MOVE](alter-table-move.md), Moves rows to another storage tier.
- [ALTER TABLE SET OPTIONS](alter-table-set-options.md), Changes the storage options recorded against a table, including its change data feed.
- [ALTER TABLE SET TTL](alter-table-set-ttl.md), Says how long a row lives, measured from one of its columns.
- [ALTER TABLE SET USING](alter-table-set-using.md), Moves a table between the row store and the lake.
- [ARCHIVE TABLE](archive-table.md), Writes rows out to a destination and removes them from the table.
- [CREATE BRANCH](create-branch.md), Forks a version of the data that can be written without affecting the branch it came from.
- [CREATE VERSION](create-version.md), Names a table version, so it can be read back by name.
- [DROP BRANCH](drop-branch.md), Removes a branch and the writes made on it.
- [DROP VERSION](drop-version.md), Removes a named version tag.
- [MERGE BRANCH](merge-branch.md), Replays a branch's writes onto another branch.
- [OPTIMIZE TABLE](optimize-table.md), Rewrites a table's layout, and applies the deletes it has recorded.
- [RESTORE SOFT DELETE](restore-soft-delete.md), Brings back rows a soft delete marked, while they are still there.
- [RESTORE TABLE](restore-table.md), Reads archived rows back into a table.
- [RESTORE TABLE VERSION](restore-table-version.md), Puts a table back to the contents it had at a version or a time.
- [RUN RETENTION JOB](run-retention-job.md), Acts on the rows a TTL has expired, now rather than on schedule.
- [USE BRANCH](use-branch.md), Points this session's reads and writes at a branch.
