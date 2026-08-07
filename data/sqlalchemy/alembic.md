What is Alembic?
Imagine you start your app with a simple users table. Later, you want to add an age column. How do you change your database without deleting all your data?

Alembic is a database migration tool designed for this:

It tracks changes in your models.
It creates migration scripts that apply schema changes safely.
It lets you upgrade or downgrade your database schema version by version.
Think of Alembic as a time machine for your database schema.

Setting Up Alembic and SQLAlchemy Step-by-Step
Let’s say you want to create a Python project with SQLAlchemy models and manage your database schema using Alembic.

Step 1: Install required packages
Step 2: Create your SQLAlchemy models
Step 3: Initialize Alembic in your project
Step 4: Configure Alembic to use your models
Step 5: Create your first migration
Step 6: Apply the migration to your database
Step 7: Make changes and create more migrations

Common Alembic Commands to Remember
alembic init alembic — Initialize Alembic in your project
alembic revision --autogenerate -m "message" — Generate migration script automatically
alembic upgrade head — Apply latest migrations
alembic downgrade -1 — Undo the last migration (go back one version)
alembic history — See migration history
