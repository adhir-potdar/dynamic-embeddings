#!/usr/bin/env python
"""
Create Collection Metadata Table (Python version)

Creates namespace-specific collection metadata table using Python/SQLAlchemy.

Usage:
    python create_collection_metadata_table.py --namespace NAMESPACE
"""

import argparse
import sys
from sqlalchemy import create_engine, text, inspect
from dynamic_embeddings.database.connection import DatabaseConnection
from dynamic_embeddings.database.schema import get_collection_metadata_model, Base


def create_metadata_table(namespace: str):
    """Create collection metadata table for namespace."""

    print(f"🔄 Creating collection metadata table for namespace: {namespace}")
    print("=" * 60)

    table_name = f'embeddings_collection_metadata_{namespace}'
    print(f"   Table name: {table_name}")

    try:
        # Initialize database connection
        db_conn = DatabaseConnection()

        # Get the metadata model for this namespace
        MetadataModel = get_collection_metadata_model(namespace)

        # Create table using SQLAlchemy
        print(f"\n   ℹ️  Creating table...")
        with db_conn.get_session() as session:
            # Create table
            Base.metadata.create_all(
                bind=session.connection(),
                tables=[MetadataModel.__table__],
                checkfirst=True
            )

        print(f"   ✅ Table created successfully")

        # Verify table exists
        print(f"\n   ℹ️  Verifying table...")
        with db_conn.get_session() as session:
            inspector = inspect(session.bind)
            if table_name in inspector.get_table_names():
                print(f"   ✅ Table verified: {table_name}")

                # Show table structure
                print(f"\n   📋 Table structure:")
                columns = inspector.get_columns(table_name)
                for col in columns:
                    col_type = str(col['type'])
                    nullable = "NULL" if col['nullable'] else "NOT NULL"
                    print(f"      - {col['name']}: {col_type} {nullable}")

                # Show indexes
                indexes = inspector.get_indexes(table_name)
                if indexes:
                    print(f"\n   📋 Indexes:")
                    for idx in indexes:
                        cols = ', '.join(idx['column_names'])
                        print(f"      - {idx['name']}: ({cols})")

                print(f"\n{'=' * 60}")
                print(f"✅ Collection metadata table setup complete!")
                print(f"\n   Next steps:")
                print(f"      1. Run metadata builder: python build_collection_metadata.py --namespace {namespace}")
                print(f"      2. Test V4: python ../yield_mgmt_qa_ex_charts_v4.py")
                print(f"{'=' * 60}")
            else:
                print(f"   ❌ Table verification failed")
                sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create collection metadata table for namespace'
    )
    parser.add_argument(
        '--namespace',
        required=True,
        help='Namespace for the table (e.g., default, revenue_mgmt)'
    )

    args = parser.parse_args()

    # Validate namespace format
    import re
    if not re.match(r'^[a-z0-9_]+$', args.namespace):
        print(f"❌ Invalid namespace: {args.namespace}")
        print(f"   Namespace must contain only lowercase letters, numbers, and underscores")
        sys.exit(1)

    create_metadata_table(args.namespace)
