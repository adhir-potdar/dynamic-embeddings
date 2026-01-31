#!/usr/bin/env python
"""
Collection Metadata Builder

Rebuilds the collection_metadata table by:
1. Listing all collections from vector store
2. Parsing collection names to extract metadata
3. Updating metadata table with parsed information

Run daily via cron job to keep metadata fresh.

Usage:
    python build_collection_metadata.py [--namespace NAMESPACE]
"""

import re
from datetime import datetime
from sqlalchemy import text
from dynamic_embeddings.database.connection import DatabaseConnection
from dynamic_embeddings.services.vector_store import VectorStore


def parse_collection_name(collection_name: str) -> dict:
    """
    Parse collection name to extract metadata.

    Format: {dimension}_{granularity}_{start1}_{end1}_vs_{start2}_{end2}
    Example: property_geo_device_qoq_20250601_20250831_vs_20250901_20251130

    Returns:
        dict with dimension, time_granularity, period dates
    """
    # Known time granularities
    time_granularities = ['qoq', 'qtd', 'mom', 'mtd', 'wow', 'wtd', 'dod']

    parts = collection_name.split('_')

    # Find granularity position
    gran_idx = -1
    granularity = None
    for i, part in enumerate(parts):
        if part.lower() in time_granularities:
            gran_idx = i
            granularity = part.lower()
            break

    if gran_idx == -1:
        raise ValueError(f"No time granularity found in collection: {collection_name}")

    # Extract dimension (everything before granularity)
    dimension = '_'.join(parts[:gran_idx])

    # Extract dates after granularity
    # Format: {gran}_{start1}_{end1}_vs_{start2}_{end2}
    date_parts = parts[gran_idx + 1:]

    # Find 'vs' separator
    try:
        vs_idx = date_parts.index('vs')
        period1_dates = date_parts[:vs_idx]
        period2_dates = date_parts[vs_idx + 1:]

        # Validate date format (YYYYMMDD = 8 digits)
        if len(period1_dates) >= 2 and len(period2_dates) >= 2:
            return {
                'dimension': dimension,
                'time_granularity': granularity,
                'period1_start_date': int(period1_dates[0]),
                'period1_end_date': int(period1_dates[1]),
                'period2_start_date': int(period2_dates[0]),
                'period2_end_date': int(period2_dates[1])
            }
    except (ValueError, IndexError):
        pass

    raise ValueError(f"Cannot parse date format from collection: {collection_name}")


def build_metadata_table(namespace: str = 'default'):
    """Rebuild collection metadata table."""

    print(f"🔄 Starting collection metadata rebuild for namespace: {namespace}")
    print("="*60)

    # Generate table name
    table_name = f'embeddings_collection_metadata_{namespace}'
    print(f"   Target table: {table_name}")

    # Initialize services
    db_conn = DatabaseConnection()
    vector_store = VectorStore(db_conn)

    # Step 1: Get all collections (lightweight - only names)
    print("\n📋 Fetching collection names...")

    with db_conn.get_session() as session:
        # Get distinct collection names (no expensive stats)
        RecordModel = vector_store.table_factory.get_or_create_model(namespace)
        collections = session.query(RecordModel.collection_name).distinct().all()
        collection_names = [col[0] for col in collections]

    print(f"   Found {len(collection_names)} collections")

    # Step 2: Parse collection names
    print("\n🔍 Parsing collection names...")

    parsed_metadata = []
    parse_errors = []

    for collection_name in collection_names:
        try:
            metadata = parse_collection_name(collection_name)
            metadata['collection_name'] = collection_name

            # Get embedding count (single query)
            with db_conn.get_session() as session:
                count = session.query(RecordModel).filter(
                    RecordModel.collection_name == collection_name
                ).count()
                metadata['total_embeddings'] = count

            metadata['last_updated_at'] = datetime.utcnow()
            parsed_metadata.append(metadata)

        except ValueError as e:
            parse_errors.append({'collection': collection_name, 'error': str(e)})

    print(f"   Successfully parsed: {len(parsed_metadata)}")
    if parse_errors:
        print(f"   ⚠️  Parse errors: {len(parse_errors)}")
        for err in parse_errors[:5]:  # Show first 5 errors
            print(f"      - {err['collection']}: {err['error']}")

    # Step 3: Update metadata table (bulk insert/update)
    print("\n💾 Updating metadata table...")

    with db_conn.get_session() as session:
        # Clear existing metadata
        session.execute(text(f"DELETE FROM {table_name}"))

        # Bulk insert new metadata
        session.execute(
            text(f"""
                INSERT INTO {table_name}
                (collection_name, dimension, time_granularity,
                 period1_start_date, period1_end_date,
                 period2_start_date, period2_end_date,
                 total_embeddings, last_updated_at)
                VALUES
                (:collection_name, :dimension, :time_granularity,
                 :period1_start_date, :period1_end_date,
                 :period2_start_date, :period2_end_date,
                 :total_embeddings, :last_updated_at)
            """),
            parsed_metadata
        )

        session.commit()

    print(f"   ✅ Inserted {len(parsed_metadata)} metadata records")

    # Step 4: Summary
    print(f"\n{'='*60}")
    print(f"✅ Metadata rebuild complete!")
    print(f"   Collections processed: {len(collection_names)}")
    print(f"   Metadata records: {len(parsed_metadata)}")
    print(f"   Parse errors: {len(parse_errors)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Build collection metadata table')
    parser.add_argument('--namespace', default='default', help='Namespace to process')
    args = parser.parse_args()

    try:
        build_metadata_table(namespace=args.namespace)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
