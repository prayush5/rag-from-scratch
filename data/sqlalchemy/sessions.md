# SQLAlchemy Session Management

The `Session` establishes and manages all conversations between your Python program and the underlying database.

## Essential Operations

- **`session.add(obj)`**: Stages an object for insertion into the database.
- **`session.commit()`**: Writes all staged changes permanently to the database transaction.
- **`session.rollback()`**: Cancels all pending changes in the current transaction if an error occurs.
