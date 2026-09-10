# ALTER USER

Changes what a principal authenticates with or the options recorded against it. The roles it holds are unchanged, because those are granted rather than declared here.

## Syntax

```sql
ALTER USER name SET PASSWORD 'text' | RENAME TO name | WITH LOGIN | NOLOGIN | VALID UNTIL 'text'
```

## Examples

```sql
ALTER USER alice SET PASSWORD 'secret'
```

The principal authenticates with the new password.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE USER](create-user.md)
