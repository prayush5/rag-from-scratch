# Defining SQLAlchemy Relationships

Relationships link two ORM models together, allowing seamless navigation between related database tables.

## Types of Relationships

- **One-to-Many**: Built using `relationship()` combined with a `ForeignKey` constraint on the child model.
- **Many-to-Many**: Uses an intermediate association table to link two primary entities together.
