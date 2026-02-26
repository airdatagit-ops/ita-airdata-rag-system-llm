"""
Version Manager for Aviation RAG System.

Handles regulation versioning and supersession logic.
"""

from datetime import datetime
from typing import Dict, Optional
from loguru import logger

from database.qdrant_manager import QdrantManager


class VersionManager:
    """Manages regulation versions and supersession."""

    def __init__(self, qdrant_manager: QdrantManager = None):
        """
        Initialize version manager.

        Args:
            qdrant_manager: QdrantManager instance
        """
        self.db = qdrant_manager or QdrantManager()
        logger.info("VersionManager initialized")

    def supersede_regulation(
        self,
        old_regulation_id: str,
        old_version: str,
        new_version_data: Dict
    ) -> bool:
        """
        Mark old version as superseded and add new version.

        Args:
            old_regulation_id: ID of regulation to supersede
            old_version: Version string of old regulation
            new_version_data: Data for new version (with vector, payload)

        Returns:
            True if successful
        """
        try:
            # Find old version point(s)
            old_points = self._find_regulation_points(old_regulation_id, old_version)

            if not old_points:
                logger.warning(f"Old version not found: {old_regulation_id} v{old_version}")
                return False

            # Update old version: mark as superseded
            new_effective_date = new_version_data["payload"].get("effective_date")

            for point in old_points:
                updated_payload = point.payload.copy()
                updated_payload["status"] = "superseded"
                updated_payload["expiry_date"] = new_effective_date
                updated_payload["superseded_by_version"] = new_version_data["payload"].get("version")

                self.db.client.set_payload(
                    collection_name=self.db.collection_name,
                    payload=updated_payload,
                    points=[point.id]
                )

            logger.info(f"Marked {len(old_points)} points as superseded")

            # Add new version
            new_version_data["payload"]["supersedes_version"] = old_version
            self.db.upsert_points([new_version_data])

            logger.success(f"Superseded {old_regulation_id} v{old_version} with new version")
            return True

        except Exception as e:
            logger.error(f"Error superseding regulation: {e}")
            return False

    def _find_regulation_points(self, regulation_id: str, version: str = None):
        """Find points for a specific regulation and version."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        scroll_filter = Filter(
            must=[
                FieldCondition(key="regulation_id", match=MatchValue(value=regulation_id))
            ]
        )

        if version:
            scroll_filter.must.append(
                FieldCondition(key="version", match=MatchValue(value=version))
            )

        results, _ = self.db.client.scroll(
            collection_name=self.db.collection_name,
            scroll_filter=scroll_filter,
            limit=1000
        )

        return results


if __name__ == "__main__":
    print("VersionManager ready")
