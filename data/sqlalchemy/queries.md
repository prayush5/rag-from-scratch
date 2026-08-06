# Querying Data in SQLAlchemy

Modern SQLAlchemy uses the `select()` statement to construct database queries cleanly.

## Query Building

- Filtering results: `select(User).where(User.name == "Alice")`
- Joining tables: `select(User).join(User.posts)`
- Fetching results: Using `.all()`, `.first()`, or `.one_or_none()` on the session execution result.
