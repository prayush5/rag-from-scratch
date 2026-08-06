# FastAPI Dependency Injection

FastAPI features a powerful Dependency Injection system that allows developers to share logic, database sessions, and security checks across multiple routes easily.

## Key Concepts

- **`Depends`**: A function parameter default that declares a dependency.
- **Reusability**: Shared code like DB connections or authentication logic can be written once and reused anywhere.
- **Yield Dependencies**: Used for setup and teardown tasks, such as opening and closing database sessions automatically.
