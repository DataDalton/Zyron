# Transactions

5 statements and constructs.

- [BEGIN](begin.md), Opens a transaction, so the statements after it commit or roll back together.
- [COMMIT](commit.md), Ends a transaction, keeping everything it wrote.
- [RELEASE SAVEPOINT](release-savepoint.md), Forgets a savepoint, keeping the work done since it.
- [ROLLBACK](rollback.md), Ends a transaction, discarding everything it wrote.
- [SAVEPOINT](savepoint.md), Marks a point inside a transaction that a rollback can return to.
