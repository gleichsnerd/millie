# Millie, an ORM-esque library for Milvus and Python

[![Tests](https://github.com/gleichsnerd/millie/actions/workflows/test.yml/badge.svg)](https://github.com/gleichsnerd/millie/actions/workflows/test.yml)

*Users beware: Millie is currently in a very early stage of development. It is not yet ready for production use.*

Millie is a library that aims to provide a helpful interface for interacting with Milvus. It is built on top of the [Milvus Python SDK](https://github.com/milvus-io/pymilvus), but aims to add an ORM layer with migration, seeding, and (eventually) cached embedding support in order to reduce the amount of boilerplate code required over time.

## Layout

Millie is broken up into the following areas:

### `cli`

The `cli` module contains the command line interface for Millie. It is built using [Click](https://click.palletsprojects.com/) and is used to interact with the database. Invoke the cli with `millie` to get started, or add it to your existing Click commands with the following:

```python
from millie.cli import add_millie_commands
...
add_millie_commands(cli)
```

### `db`

The `db` module is in charge of making connecting to and updating the database easier, along with generating and running migrations.

### `orm`

The `orm` module contains utilities for the ORM layer for Millie. The core piece is the `MilvusModel` decorator which is used to extend your classes with ORM functionality and be picked up by the migration system.
