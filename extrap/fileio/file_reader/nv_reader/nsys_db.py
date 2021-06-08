# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import sqlite3
from pathlib import Path
from sqlite3 import Cursor
from typing import Tuple, List

from extrap.util.exceptions import FileFormatError


class NsysReport:
    def __init__(self, filename):
        filename = Path(filename)
        self._check_file(filename)
        self.db = sqlite3.connect(filename)
        self._prepare_shared()

    def _check_file(self, filename):
        if filename.suffix == ".qdstrm":
            raise FileFormatError(
                f"You need to convert the file into a .qdrep-report and than into a sqlite database: {filename}")
        elif filename.suffix == ".qdrep":
            raise FileFormatError(f"You need to convert the file into a sqlite database: {filename}")
        elif filename.suffix == ".nsight-cuprof-report":
            raise FileFormatError(f"You need to use the ncu reader to open the file: {filename}")
        else:
            return filename

    def _check_table_exists(self, name):
        return self.db.execute(
            "SELECT name FROM sqlite_master WHERE (type='table' OR type='view') AND name=?",
            [name]).fetchone() is not None

    def _prepare_shared(self):
        self.db.execute("""CREATE VIEW IF NOT EXISTS EXTRAP_RESOLVED_CALLPATHS AS
WITH resolved_callchains AS (
    SELECT CUDA_CALLCHAINS.id, symbol, stackDepth, value AS name
    FROM CUDA_CALLCHAINS
             LEFT JOIN StringIds ON symbol = StringIds.id
    WHERE unresolved IS NULL
    ORDER BY CUDA_CALLCHAINS.id, stackDepth DESC
)
SELECT id,
       GROUP_CONCAT(symbol, ',') AS path,
       MAX(stackDepth),
       GROUP_CONCAT(name, '->')  AS callpath
FROM resolved_callchains
GROUP BY id
         """)

    def get_mem_copies(self) -> List[Tuple[int, str, str, str, float, int, str, float]]:
        if not self._check_table_exists('CUPTI_ACTIVITY_KIND_MEMCPY'):
            return []
        c = self.db.cursor()
        return list(c.execute("""WITH cupti_memory AS (
    SELECT correlationId,
           (end - start) AS duration,
           CASE copyKind
               WHEN 0 THEN 'UNKNOWN'
               WHEN 1 THEN 'HTOD'
               WHEN 2 THEN 'DTOH'
               WHEN 3 THEN 'HTOA'
               WHEN 4 THEN 'ATOH'
               WHEN 5 THEN 'ATOA'
               WHEN 6 THEN 'ATOD'
               WHEN 7 THEN 'DTOA'
               WHEN 8 THEN 'DTOD'
               WHEN 9 THEN 'HTOH'
               WHEN 10 THEN 'PTOP'
               END       AS copyKind,
           bytes,
           NULL          AS shortName
    FROM CUPTI_ACTIVITY_KIND_MEMCPY),
     cupti_memory_overlap AS (
         SELECT correlationId,
                duration,
                CASE copyKind
                    WHEN 0 THEN 'UNKNOWN'
                    WHEN 1 THEN 'HTOD'
                    WHEN 2 THEN 'DTOH'
                    WHEN 3 THEN 'HTOA'
                    WHEN 4 THEN 'ATOH'
                    WHEN 5 THEN 'ATOA'
                    WHEN 6 THEN 'ATOD'
                    WHEN 7 THEN 'DTOA'
                    WHEN 8 THEN 'DTOD'
                    WHEN 9 THEN 'HTOH'
                    WHEN 10 THEN 'PTOP'
                    END AS copyKind,
                NULL    AS bytes,
                shortName
         FROM (SELECT correlationId, CAKK.end - CAKM.end AS duration, copyKind, shortName
               FROM CUPTI_ACTIVITY_KIND_MEMCPY AS CAKM
                        INNER JOIN (SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL) AS CAKK
                                   ON CAKK.start BETWEEN CAKM.start AND CAKM.end AND
                                      CAKK.end NOT BETWEEN CAKM.start AND CAKM.end
               UNION ALL
               SELECT correlationId, CAKM.start - CAKK.start AS duration, copyKind, shortName
               FROM CUPTI_ACTIVITY_KIND_MEMCPY AS CAKM
                        INNER JOIN (SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL) AS CAKK
                                   ON CAKK.end BETWEEN CAKM.start AND CAKM.end AND
                                      CAKK.start NOT BETWEEN CAKM.start AND CAKM.end
               UNION ALL
               SELECT correlationId, CAKM.end - CAKM.start - (CAKK.start - CAKK.end) AS duration, copyKind, shortName
               FROM CUPTI_ACTIVITY_KIND_MEMCPY AS CAKM
                        INNER JOIN (SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL) AS CAKK
                                   ON CAKK.end BETWEEN CAKM.start AND CAKM.end AND
                                      CAKK.start BETWEEN CAKM.start AND CAKM.end
               UNION ALL
               SELECT correlationId, CAKM.end - CAKM.start AS duration, copyKind, shortName
               FROM CUPTI_ACTIVITY_KIND_MEMCPY AS CAKM
                        INNER JOIN (SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL) AS CAKK
                                   ON CAKK.end NOT BETWEEN CAKM.start AND CAKM.end AND
                                      CAKK.start NOT BETWEEN CAKM.start AND CAKM.end
              )
     ),
     cupti_activity AS (
         SELECT callchainId,
                value                                                               AS demangledName,
                copyKind,
                CUPTI_ACTIVITY_KIND_RUNTIME.end - CUPTI_ACTIVITY_KIND_RUNTIME.start AS duration,
                bytes,
                CA.duration                                                         AS other_duration
         FROM CUPTI_ACTIVITY_KIND_RUNTIME
                  INNER JOIN (SELECT *
                              FROM cupti_memory
                              --UNION ALL
                              --SELECT *
                              --FROM cupti_memory_overlap
                              ) AS CA
                             ON CUPTI_ACTIVITY_KIND_RUNTIME.correlationId = CA.correlationId
                  LEFT JOIN StringIds ON nameId = StringIds.id
     )
SELECT paths.id,
       path,
       callpath,
       demangledName       AS name,
       SUM(duration)       AS duration,
       SUM(bytes)          AS bytes,
       copyKind,
       SUM(other_duration) AS other_duration
FROM (SELECT id,
             path,
             demangledName,
             callpath,
             duration,
             copyKind,
             other_duration,
             bytes
      FROM EXTRAP_RESOLVED_CALLPATHS
               INNER JOIN cupti_activity ON (id - 1) = CUPTI_ACTIVITY.callchainId) AS paths
GROUP BY path, demangledName, copyKind 
    """))

    def get_synchronization(self) -> List[Tuple[int, str, str, str, float, float, str, float]]:
        if not self._check_table_exists('CUPTI_ACTIVITY_KIND_SYNCHRONIZATION'):
            return []
        c = self.db.cursor()
        return list(c.execute("""WITH cupti_synchronization AS (
    SELECT correlationId,
           (end - start) AS duration,
           NULL          AS durationGPU,
           CASE syncType
               WHEN 0 THEN 'UNKNOWN'
               WHEN 1 THEN 'EVENT_SYNCHRONIZE'
               WHEN 2 THEN 'STREAM_WAIT_EVENT'
               WHEN 3 THEN 'STREAM_SYNCHRONIZE'
               WHEN 4 THEN 'CONTEXT_SYNCHRONIZE'
               END       AS syncType,
           NULL          AS shortName
    FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION),
     cupti_synchronization_kernels AS (
         SELECT correlationId,
                NULL                                                    AS duration,
                (MIN(CAKS.end, CAKK.end) - MAX(CAKS.start, CAKK.start)) AS durationGPU,
                CASE syncType
                    WHEN 0 THEN 'UNKNOWN'
                    WHEN 1 THEN 'EVENT_SYNCHRONIZE'
                    WHEN 2 THEN 'STREAM_WAIT_EVENT'
                    WHEN 3 THEN 'STREAM_SYNCHRONIZE'
                    WHEN 4 THEN 'CONTEXT_SYNCHRONIZE'
                    END                                                 AS syncType,
                shortName
         FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION AS CAKS
                  LEFT JOIN (SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL) AS CAKK
                            ON CAKK.start BETWEEN CAKS.start AND CAKS.end OR CAKK.end BETWEEN CAKS.start AND CAKS.end
     ),
     cupti_activity AS (
         SELECT callchainId,
                value                                                               AS demangledName,
                syncType,
                CUPTI_ACTIVITY_KIND_RUNTIME.end - CUPTI_ACTIVITY_KIND_RUNTIME.start AS duration,
                durationGPU,
                CAKK.duration                                                       AS other_duration
         FROM CUPTI_ACTIVITY_KIND_RUNTIME
                  INNER JOIN (SELECT *
                             FROM cupti_synchronization
                             UNION ALL
                             SELECT *
                             FROM cupti_synchronization_kernels) AS CAKK
                            ON CUPTI_ACTIVITY_KIND_RUNTIME.correlationId = CAKK.correlationId
                  LEFT JOIN StringIds ON shortName = StringIds.id
     )
SELECT paths.id,
       path,
       callpath,
       demangledName       AS kernelName,
       SUM(duration)       AS duration,
       SUM(durationGPU)    AS durationGPU,
       syncType,
       SUM(other_duration) AS other_duration
FROM (SELECT id,
             path,
             demangledName,
             callpath,
             duration,
             durationGPU,
             syncType,
             other_duration
      FROM EXTRAP_RESOLVED_CALLPATHS
               INNER JOIN cupti_activity ON (id - 1) = CUPTI_ACTIVITY.callchainId) AS paths
GROUP BY path, demangledName, syncType 
    """))

    def get_kernel_runtimes(self) -> List[Tuple[int, str, str, str, float, float, float]]:
        if not self._check_table_exists('CUPTI_ACTIVITY_KIND_KERNEL'):
            return []
        c = self.db.cursor()
        return list(c.execute("""WITH cupti_kernel AS (
    SELECT correlationId,
           NULL                                                     AS duration,
           (end - start)                                            AS durationGPU,
           shortName,
           ('(' || gridX || ',' || gridY || ',' || gridZ || ')')    AS grid,
           ('(' || blockX || ',' || blockY || ',' || blockZ || ')') AS block,
           sharedMemoryExecuted
    FROM CUPTI_ACTIVITY_KIND_KERNEL
),
     cupti_activity AS (
         SELECT callchainId,
                value                                                               AS demangledName,
                grid,
                block,
                sharedMemoryExecuted,
                CUPTI_ACTIVITY_KIND_RUNTIME.end - CUPTI_ACTIVITY_KIND_RUNTIME.start AS duration,
                durationGPU,
                CAKK.duration                                                       AS other_duration
         FROM CUPTI_ACTIVITY_KIND_RUNTIME
                  LEFT JOIN cupti_kernel AS CAKK
                            ON CUPTI_ACTIVITY_KIND_RUNTIME.correlationId = CAKK.correlationId
                  LEFT JOIN StringIds ON shortName = StringIds.id
     )
SELECT paths.id,
       path,
       callpath,
       demangledName       AS kernelName,
       SUM(duration)       AS duration,
       SUM(durationGPU)    AS durationGPU,
       SUM(other_duration) AS other_duration
FROM (SELECT id,
             path,
             demangledName,
             --(demangledName || '<<<' || grid || ',' || block || ',' || sharedMemoryExecuted || '>>>') AS demangledName,
             callpath,
             duration,
             durationGPU,
             other_duration
      FROM EXTRAP_RESOLVED_CALLPATHS
               LEFT JOIN cupti_activity ON (id - 1) = CUPTI_ACTIVITY.callchainId) AS paths
GROUP BY path, demangledName 
"""))

    def get_kernelid_paths(self) -> Cursor:

        c = self.db.cursor()
        return c.execute("""WITH cupti_kernel AS (
    SELECT correlationId,
           gridId,
           (end - start)                                            AS durationGPU,
           shortName,
           ('(' || gridX || ',' || gridY || ',' || gridZ || ')')    AS grid,
           ('(' || blockX || ',' || blockY || ',' || blockZ || ')') AS block,
           sharedMemoryExecuted
    FROM CUPTI_ACTIVITY_KIND_KERNEL
),
     cupti_activity AS (
         SELECT callchainId,
                gridId,
                value                                                               AS demangledName,
                grid,
                block,
                sharedMemoryExecuted,
                CUPTI_ACTIVITY_KIND_RUNTIME.end - CUPTI_ACTIVITY_KIND_RUNTIME.start AS duration,
                durationGPU
         FROM cupti_kernel AS CAKK
                  LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME
                            ON CUPTI_ACTIVITY_KIND_RUNTIME.correlationId = CAKK.correlationId
                  LEFT JOIN StringIds ON shortName = StringIds.id
         WHERE callchainId IS NOT NULL
     )
SELECT demangledName AS name,
       grid,
       block,
       sharedMemoryExecuted,
       callpath
FROM cupti_activity
         LEFT JOIN EXTRAP_RESOLVED_CALLPATHS ON (id - 1) = CUPTI_ACTIVITY.callchainId

    """)

    def get_os_runtimes(self) -> List[Tuple[int, str, str, str, float, float, str, float]]:
        if not self._check_table_exists('OSRT_API'):
            return []
        c = self.db.cursor()
        return list(c.execute("""WITH osrt_activity AS (
    SELECT callchainId,
           end - start AS duration,
           value       AS name
    FROM OSRT_API
             LEFT JOIN StringIds ON nameId = StringIds.id
)
SELECT paths.id,
       path,
       callpath,
       name,
       SUM(duration) AS duration
FROM (SELECT id,
             path,
             name,
             callpath,
             duration
      FROM EXTRAP_RESOLVED_CALLPATHS
               LEFT JOIN osrt_activity ON id = osrt_activity.callchainId) AS paths
GROUP BY path, name 
    """))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()
