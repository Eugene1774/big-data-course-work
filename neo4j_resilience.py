from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from threading import Lock
from typing import Final
from urllib.parse import urlparse

from neo4j import Driver, GraphDatabase


LOGGER = logging.getLogger(__name__)
REQUIRED_SCHEMA_LABELS: Final[frozenset[str]] = frozenset({"Concept", "Method", "Task", "Case", "Metric", "Mistake", "Evidence", "KnowledgeCard", "Project"})
MAX_CONNECT_RETRIES: Final[int] = 3
BASE_BACKOFF_SECONDS: Final[float] = 0.5
POOL_CONFIG: Final[dict[str, int | float]] = {
    "max_connection_pool_size": 25,
    "connection_timeout": 60,
    "max_connection_lifetime": 3600,
}
SCHEMA_LABELS_QUERY: Final[str] = """
CALL db.schema.visualization()
YIELD nodes
UNWIND nodes AS schema_node
UNWIND labels(schema_node) AS label
RETURN collect(DISTINCT label) AS labels
"""


def _credential_status_payload(uri: str, user: str, password: str) -> dict[str, bool | str]:
    sanitized_uri = str(uri or "").strip()
    sanitized_user = str(user or "").strip()
    sanitized_password = str(password or "")
    return {
        "uri_present": bool(sanitized_uri),
        "user_present": bool(sanitized_user),
        "password_present": bool(sanitized_password),
        "user_is_default": sanitized_user == "neo4j",
        "password_is_default": sanitized_password == "neo4j",
    }


def _credential_status_text(uri: str, user: str, password: str) -> str:
    return json.dumps(_credential_status_payload(uri, user, password), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class Neo4jConnectionConfig:
    uri: str
    user: str
    password: str

    @property
    def credential_status(self) -> str:
        return _credential_status_text(self.uri, self.user, self.password)


def _safe_close(driver: Driver | None) -> None:
    if driver is None:
        return
    try:
        driver.close()
    except Exception:
        LOGGER.debug("Closing Neo4j driver raised a secondary exception.", exc_info=True)


def _build_optional_resolver(uri: str):
    resolver_ip = str(os.getenv("NEO4J_RESOLVER_IP") or "").strip()
    if not resolver_ip:
        return None

    parsed = urlparse(str(uri or "").strip())
    uri_host = (parsed.hostname or "").strip().lower()
    resolver_host = str(os.getenv("NEO4J_RESOLVER_HOST") or "").strip().lower() or uri_host

    try:
        resolver_port = int(str(os.getenv("NEO4J_RESOLVER_PORT") or "7687").strip())
    except ValueError:
        resolver_port = 7687

    def _resolver(address):
        host = ""
        port = None
        if hasattr(address, "host"):
            host = str(getattr(address, "host") or "").strip().lower()
            port = getattr(address, "port", None)
        elif isinstance(address, tuple) and len(address) >= 2:
            host = str(address[0]).strip().lower()
            port = address[1]
        if host == resolver_host:
            return [(resolver_ip, resolver_port or int(port or 7687))]
        return [address]

    return _resolver


def _schema_labels(driver: Driver, database: str) -> set[str]:
    with driver.session(database=database) as session:
        record = session.run(SCHEMA_LABELS_QUERY).single()
    labels = record.get("labels", []) if record else []
    return {str(label).strip() for label in labels if str(label).strip()}


def validate_schema_or_raise(
    driver: Driver,
    database: str,
    *,
    uri: str,
    user: str,
    password: str,
) -> None:
    credential_status = _credential_status_text(uri, user, password)
    try:
        labels = _schema_labels(driver, database)
    except Exception as exc:
        LOGGER.exception(
            "Neo4j schema inspection failed | uri=%s | database=%s | credential_status=%s",
            uri,
            database,
            credential_status,
        )
        raise RuntimeError(
            f"Neo4j schema inspection failed for database '{database}' at '{uri}'."
        ) from exc

    missing_labels = sorted(REQUIRED_SCHEMA_LABELS - labels)
    if missing_labels:
        LOGGER.error(
            "Neo4j schema validation failed | uri=%s | database=%s | labels=%s | missing_labels=%s | credential_status=%s",
            uri,
            database,
            sorted(labels),
            missing_labels,
            credential_status,
        )
        raise RuntimeError(
            "Neo4j schema validation failed: missing required labels "
            f"{missing_labels} in database '{database}' at '{uri}'."
        )

    LOGGER.info(
        "Neo4j schema validated | uri=%s | database=%s | labels=%s | credential_status=%s",
        uri,
        database,
        sorted(labels),
        credential_status,
    )


class Neo4jConnectionManager:
    _instances: dict[tuple[str, str, str], "Neo4jConnectionManager"] = {}
    _instances_lock: Lock = Lock()

    def __init__(self, config: Neo4jConnectionConfig) -> None:
        self._config = config
        self._driver: Driver | None = None
        self._lock = Lock()
        self._validated_databases: set[str] = set()

    @classmethod
    def get_instance(cls, uri: str, user: str, password: str) -> "Neo4jConnectionManager":
        key = (str(uri or "").strip(), str(user or "").strip(), str(password or ""))
        with cls._instances_lock:
            manager = cls._instances.get(key)
            if manager is None:
                manager = cls(Neo4jConnectionConfig(*key))
                cls._instances[key] = manager
            return manager

    def _assert_config(self) -> None:
        status = self._config.credential_status
        if not self._config.uri.strip() or not self._config.user.strip() or not self._config.password:
            LOGGER.error(
                "Neo4j configuration is incomplete | uri=%s | credential_status=%s",
                self._config.uri,
                status,
            )
            raise ConnectionError(
                f"Neo4j configuration is incomplete for uri '{self._config.uri}'. "
                f"credential_status={status}"
            )

    def _build_driver_with_retry(self) -> Driver:
        self._assert_config()
        last_error: Exception | None = None

        for attempt in range(1, MAX_CONNECT_RETRIES + 1):
            driver: Driver | None = None
            try:
                resolver = _build_optional_resolver(self._config.uri)
                driver_kwargs = dict(POOL_CONFIG)
                if resolver is not None:
                    driver_kwargs["resolver"] = resolver
                driver = GraphDatabase.driver(
                    self._config.uri,
                    auth=(self._config.user, self._config.password),
                    **driver_kwargs,
                )
                driver.verify_connectivity()
                LOGGER.info(
                    "Neo4j connectivity verified | uri=%s | attempt=%s/%s | credential_status=%s",
                    self._config.uri,
                    attempt,
                    MAX_CONNECT_RETRIES,
                    self._config.credential_status,
                )
                return driver
            except Exception as exc:
                last_error = exc
                _safe_close(driver)
                LOGGER.warning(
                    "Neo4j connection attempt failed | uri=%s | attempt=%s/%s | credential_status=%s | error=%s",
                    self._config.uri,
                    attempt,
                    MAX_CONNECT_RETRIES,
                    self._config.credential_status,
                    exc,
                )
                if attempt < MAX_CONNECT_RETRIES:
                    time.sleep(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)))

        raise ConnectionError(
            f"Failed to connect to Neo4j at '{self._config.uri}' after {MAX_CONNECT_RETRIES} attempts. "
            f"credential_status={self._config.credential_status}"
        ) from last_error

    def get_driver(self, database: str, *, validate_schema: bool = True) -> Driver:
        with self._lock:
            if self._driver is not None:
                try:
                    self._driver.verify_connectivity()
                except Exception as exc:
                    LOGGER.warning(
                        "Cached Neo4j driver became unhealthy and will be recreated | uri=%s | database=%s | credential_status=%s | error=%s",
                        self._config.uri,
                        database,
                        self._config.credential_status,
                        exc,
                    )
                    _safe_close(self._driver)
                    self._driver = None
                    self._validated_databases.clear()

            if self._driver is None:
                self._driver = self._build_driver_with_retry()

            if validate_schema and database not in self._validated_databases:
                validate_schema_or_raise(
                    self._driver,
                    database,
                    uri=self._config.uri,
                    user=self._config.user,
                    password=self._config.password,
                )
                self._validated_databases.add(database)

            return self._driver

    def close(self) -> None:
        with self._lock:
            _safe_close(self._driver)
            self._driver = None
            self._validated_databases.clear()


_EXTERNAL_DRIVER_VALIDATIONS: set[tuple[int, str]] = set()
_EXTERNAL_DRIVER_VALIDATIONS_LOCK = Lock()


def get_managed_driver(
    uri: str,
    user: str,
    password: str,
    database: str,
    *,
    validate_schema: bool = True,
) -> Driver:
    manager = Neo4jConnectionManager.get_instance(uri, user, password)
    return manager.get_driver(database, validate_schema=validate_schema)


def ensure_external_driver_schema(
    driver: Driver,
    database: str,
    *,
    uri: str,
    user: str,
    password: str,
) -> Driver:
    cache_key = (id(driver), database)
    with _EXTERNAL_DRIVER_VALIDATIONS_LOCK:
        if cache_key in _EXTERNAL_DRIVER_VALIDATIONS:
            return driver

    validate_schema_or_raise(driver, database, uri=uri, user=user, password=password)
    with _EXTERNAL_DRIVER_VALIDATIONS_LOCK:
        _EXTERNAL_DRIVER_VALIDATIONS.add(cache_key)
    return driver


def close_managed_driver(uri: str, user: str, password: str) -> None:
    manager = Neo4jConnectionManager.get_instance(uri, user, password)
    manager.close()
