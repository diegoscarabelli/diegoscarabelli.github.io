---
date: "2025-10-24"
lastmod: "2025-10-24"
author: "Diego Scarabelli"
title: "garmin-health-data"
description: "A CLI tool to extract your complete Garmin Connect health and activity data to a local SQLite database."
comments: true
categories:
- data engineering
tags:
- openetl
- garmin
---

<!--more-->

This is a short follow-up to my previous post [OpenETL: The Data Engineering Bicycle](../openetl/). Some readers suggested it would be valuable to have the [Garmin Connect data pipeline](https://github.com/diegoscarabelli/openetl/tree/main/dags/pipelines/garmin), which was built within the OpenETL framework, available as a standalone tool with minimal setup requirements. Since most of the data processing code was already written, this seemed like an achievable addition with moderate effort. The result is a Python package called [garmin-health-data](https://pypi.org/project/garmin-health-data/), which you can install directly into your Python environment:

```bash
pip install garmin-health-data
```

The package includes a command-line interface (CLI) tool called `garmin`, which extracts your complete Garmin Connect health and activity data to a local [SQLite](https://sqlite.org/) database file. The [database schema](https://github.com/diegoscarabelli/garmin-health-data/blob/main/garmin_health_data/tables.ddl) is a close adaptation to SQLite of the PostgreSQL/TimescaleDB schema from the OpenETL pipeline. It includes inline comments for all tables and columns, making it easy to understand your data structure (whether you're exploring it yourself or using an LLM agent). You can retrieve the schema directly from the SQLite `sqlite_master` table:

```python
import sqlite3

conn = sqlite3.connect("garmin_health_data.db")
cursor = conn.cursor()

# Get the CREATE TABLE statement with inline comments
cursor.execute("""
    SELECT sql
    FROM sqlite_master
    WHERE type='table' AND name='activity'
""")

create_sql = cursor.fetchone()[0]
print(create_sql)

conn.close()
```

The [GitHub repository](https://github.com/diegoscarabelli/garmin-health-data) includes a quick start guide, detailed usage instructions, and examples to help you get started.

Here's a short demonstration of the tool in action:

<video width="60%" controls>
  <source src="index_files/garmin.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

SQLite is a serverless, self-contained database engine that is easy to set up and use, making it an excellent choice for personal data storage and analysis. With `garmin-health-data`, you can have your Garmin Connect data readily available in a structured format for analysis, visualization, or integration with other tools.

Other projects exist with similar goals, as discussed in [this section of the docs](https://github.com/diegoscarabelli/garmin-health-data?tab=readme-ov-file#comparison-with-other-tools). However, `garmin-health-data` stands out for its comprehensive and well-documented schema, automatic data deduplication, and the ability to process detailed FIT files for time-series activity data, with enhanced support for running, cycling, and swimming activities while maintaining broad compatibility with other activity types.