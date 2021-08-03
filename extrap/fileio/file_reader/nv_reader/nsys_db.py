# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import sqlite3
from pathlib import Path
from sqlite3 import Cursor
from typing import Tuple, List

from extrap.util.deprecation import deprecated
from extrap.util.exceptions import FileFormatError


class NsysReport:
    def __init__(self, filename):
        filename = Path(filename)
        self._check_file(filename)
        self.db = sqlite3.connect(filename)
        self._prepare_shared()

    @staticmethod
    def _check_file(filename):
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
        if self._check_table_exists('EXTRAP_RESOLVED_CALLPATHS'):
            # if necessary upgrade to newer view
            if self.db.execute("""SELECT name AS CNTREC
FROM pragma_table_info('EXTRAP_RESOLVED_CALLPATHS')
WHERE name = 'correlationId'""").fetchone() is None:
                self.db.execute("DROP VIEW EXTRAP_RESOLVED_CALLPATHS")
            else:
                return

        if self._check_table_exists('NVTX_EVENTS'):
            # if possible use extra_prof data
            domain_id = self.db.execute(
                "SELECT domain_id FROM NVTX_EVENTS  WHERE text='de.tu-darmstadt.parallel.extra_prof'").fetchone()
            if domain_id:
                domain_id = domain_id[0]
                self._prepare_callpaths_from_extra_prof(domain_id)
                return

        self._prepare_callpaths_from_nsys_tracing()

    def _prepare_callpaths_from_nsys_tracing(self):
        self.db.execute("""CREATE VIEW IF NOT EXISTS EXTRAP_RESOLVED_CALLPATHS AS
WITH resolved_callchains AS (
    SELECT CUDA_CALLCHAINS.id, symbol, stackDepth, VALUE AS NAME
    FROM CUDA_CALLCHAINS
             LEFT JOIN StringIds ON symbol = StringIds.id
    WHERE unresolved IS NULL
    ORDER BY CUDA_CALLCHAINS.id, stackDepth DESC
),
     callchains AS (SELECT id,
                           GROUP_CONCAT(symbol, ',') AS path,
                           MAX(stackDepth)           AS stackDepth,
                           GROUP_CONCAT(name, '->')  AS callpath
                    FROM resolved_callchains
                    GROUP BY id)
SELECT correlationId,
       path || ',' || nameId                          AS path,
       callchains.callpath || '->' || StringIds.value AS callpath,
       stackDepth + 1                                 AS stackDepth,
       CA.end - CA.start                              AS duration
FROM callchains
         INNER JOIN main.CUPTI_ACTIVITY_KIND_RUNTIME AS CA ON callchains.id - 1 = CA.callchainId
         LEFT JOIN StringIds ON nameId = StringIds.id
GROUP BY correlationId
         """)

    def _prepare_callpaths_from_extra_prof(self, domain_id):
        self.db.execute(f"""CREATE VIEW IF NOT EXISTS EXTRAP_RESOLVED_CALLPATHS AS
WITH nvtx AS (
    SELECT start, end, value, uint32Value AS depth, textId AS path, globalTid
    FROM NVTX_EVENTS
             LEFT JOIN StringIds ON textId = StringIds.id
    WHERE eventType = 59
      AND domainId = {domain_id}
      AND textId IS NOT NULL
),
     nvtxrec AS (
         SELECT *
         FROM nvtx
         WHERE depth = 0
         UNION ALL
         SELECT nvtx.start,
                nvtx.end,
                nvtxrec.value || '->' || nvtx.value,
                nvtx.depth,
                nvtxrec.path || ',' || nvtx.path,
                nvtxrec.globalTid
         FROM nvtxrec
                  INNER JOIN nvtx ON nvtxrec.start < nvtx.start AND nvtx.end < nvtxrec.end AND
                                     nvtxrec.depth = nvtx.depth - 1 AND nvtxrec.globalTid = nvtx.globalTid
         ORDER BY nvtx.start
     )
SELECT correlationId,
       path || ',' || nameId                    AS path,
       nvtxrec.value || '->' || StringIds.value AS callpath,
       MAX(depth) + 1                           AS stackDepth,
       CA.end - CA.start                        AS duration
FROM nvtxrec
         INNER JOIN main.CUPTI_ACTIVITY_KIND_RUNTIME AS CA
                    ON nvtxrec.start < CA.start AND CA.end < nvtxrec.end AND CA.globalTid = nvtxrec.globalTid
         LEFT JOIN StringIds ON nameId = id
GROUP BY correlationId
UNION ALL
SELECT NULL AS correlationId, path, value AS callpath, depth AS stackDepth, end - start AS duration
FROM nvtxrec""")

    def get_mem_copies(self) -> List[Tuple[int, str, str, str, float, int, str, float]]:
        if not self._check_table_exists('CUPTI_ACTIVITY_KIND_MEMCPY'):
            return []
        c = self.db.cursor()
        return list(c.execute("""WITH cupti_memory AS (
    SELECT correlationId,
           (END - START) AS duration,
           CASE copyKind
               WHEN 0 THEN 'UNKNOWN'
               WHEN 1 THEN 'HOST_TO_DEVICE'
               WHEN 2 THEN 'DEVICE_TO_HOST'
               WHEN 3 THEN 'HOST_TO_DEVICE_ARRAY'
               WHEN 4 THEN 'DEVICE_ARRAY_TO_HOST'
               WHEN 5 THEN 'DEVICE_ARRAY_TO_DEVICE_ARRAY'
               WHEN 6 THEN 'DEVICE_ARRAY_TO_DEVICE'
               WHEN 7 THEN 'DEVICE_TO_DEVICE_ARRAY'
               WHEN 8 THEN 'DEVICE_TO_DEVICE'
               WHEN 9 THEN 'HOST_TO_HOST'
               WHEN 10 THEN 'PEER_TO_PEER'
               END       AS copyKind,
           bytes,
           NULL          AS shortName
    FROM CUPTI_ACTIVITY_KIND_MEMCPY),
     cupti_memory_overlap AS (
         SELECT correlationId,
                (MIN(CAKS.end, CAKK.end) - MAX(CAKS.start, CAKK.start)) AS duration,
                CASE copyKind
                    WHEN 0 THEN 'UNKNOWN'
                    WHEN 1 THEN 'HOST_TO_DEVICE'
                    WHEN 2 THEN 'DEVICE_TO_HOST'
                    WHEN 3 THEN 'HOST_TO_DEVICE_ARRAY'
                    WHEN 4 THEN 'DEVICE_ARRAY_TO_HOST'
                    WHEN 5 THEN 'DEVICE_ARRAY_TO_DEVICE_ARRAY'
                    WHEN 6 THEN 'DEVICE_ARRAY_TO_DEVICE'
                    WHEN 7 THEN 'DEVICE_TO_DEVICE_ARRAY'
                    WHEN 8 THEN 'DEVICE_TO_DEVICE'
                    WHEN 9 THEN 'HOST_TO_HOST'
                    WHEN 10 THEN 'PEER_TO_PEER'
                    END                                                 AS copyKind,
                NULL                                                    AS bytes,
                shortName
         FROM CUPTI_ACTIVITY_KIND_MEMCPY AS CAKS
                  INNER JOIN (SELECT start, END, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL) AS CAKK
                             ON CAKK.start BETWEEN CAKS.start AND CAKS.end OR CAKK.end BETWEEN CAKS.start AND CAKS.end
     ),
     cupti_activity AS (
         SELECT CA.correlationId,
                NULL        AS demangledName,
                copyKind,
                bytes,
                CA.duration AS other_duration
         FROM (SELECT *
               FROM cupti_memory
               UNION ALL
               SELECT *
               FROM cupti_memory_overlap
              ) AS CA
     )
SELECT paths.correlationId,
       path,
       callpath,
       demangledName       AS name,
       SUM(duration)       AS duration,
       SUM(bytes)          AS bytes,
       copyKind,
       SUM(other_duration) AS other_duration
FROM (SELECT EXTRAP_RESOLVED_CALLPATHS.correlationId,
             path,
             demangledName,
             callpath,
             duration,
             copyKind,
             other_duration,
             bytes
      FROM EXTRAP_RESOLVED_CALLPATHS
               INNER JOIN cupti_activity
                          ON EXTRAP_RESOLVED_CALLPATHS.correlationId = CUPTI_ACTIVITY.correlationId) AS paths
GROUP BY path, demangledName, copyKind 
    """))

    def get_synchronization(self) -> List[Tuple[int, str, str, str, float, float, str, float]]:
        if not self._check_table_exists('CUPTI_ACTIVITY_KIND_SYNCHRONIZATION'):
            return []
        c = self.db.cursor()
        return list(c.execute("""WITH cupti_synchronization AS (
    SELECT correlationId,
           (END - START) AS duration,
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
                  LEFT JOIN (SELECT start, END, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL) AS CAKK
                            ON CAKK.start BETWEEN CAKS.start AND CAKS.end OR CAKK.end BETWEEN CAKS.start AND CAKS.end
     ),
     cupti_activity AS (
         SELECT correlationId,
                value         AS demangledName,
                syncType,
                durationGPU,
                CAKK.duration AS other_duration
         FROM (SELECT *
               FROM cupti_synchronization
               UNION ALL
               SELECT *
               FROM cupti_synchronization_kernels) AS CAKK

                  LEFT JOIN StringIds ON shortName = StringIds.id
     )
SELECT paths.correlationId,
       path,
       callpath,
       demangledName       AS kernelName,
       SUM(duration)       AS duration,
       SUM(durationGPU)    AS durationGPU,
       syncType,
       SUM(other_duration) AS other_duration
FROM (SELECT EXTRAP_RESOLVED_CALLPATHS.correlationId,
             path,
             demangledName,
             callpath,
             duration,
             durationGPU,
             syncType,
             other_duration
      FROM EXTRAP_RESOLVED_CALLPATHS
               INNER JOIN cupti_activity
                          ON EXTRAP_RESOLVED_CALLPATHS.correlationId = CUPTI_ACTIVITY.correlationId) AS paths
GROUP BY path, demangledName, syncType 
    """))

    def get_kernel_runtimes(self) -> List[Tuple[int, str, str, str, float, float, float]]:
        if not self._check_table_exists('CUPTI_ACTIVITY_KIND_KERNEL'):
            return []
        c = self.db.cursor()
        return list(c.execute("""WITH cupti_kernel AS (
    SELECT correlationId,
           NULL                                                     AS duration,
           (END - START)                                            AS durationGPU,
           shortName,
           ('(' || gridX || ',' || gridY || ',' || gridZ || ')')    AS grid,
           ('(' || blockX || ',' || blockY || ',' || blockZ || ')') AS block,
           sharedMemoryExecuted
    FROM CUPTI_ACTIVITY_KIND_KERNEL
),
     cupti_activity AS (
         SELECT correlationId,
                value         AS demangledName,
                grid,
                block,
                sharedMemoryExecuted,
                durationGPU,
                CAKK.duration AS other_duration
         FROM cupti_kernel AS CAKK
                  LEFT JOIN StringIds ON shortName = StringIds.id
     )
SELECT paths.correlationId,
       path,
       callpath,
       demangledName       AS kernelName,
       SUM(duration)       AS duration,
       SUM(durationGPU)    AS durationGPU,
       SUM(other_duration) AS other_duration
FROM (SELECT EXTRAP_RESOLVED_CALLPATHS.correlationId,
             path,
             demangledName,
             --(demangledName|| '<<<' || grid || ',' || block || ',' || sharedMemoryExecuted || '>>>') AS demangledName,
             callpath,
             duration,
             durationGPU,
             other_duration
      FROM EXTRAP_RESOLVED_CALLPATHS
               LEFT JOIN cupti_activity
                         ON EXTRAP_RESOLVED_CALLPATHS.correlationId = CUPTI_ACTIVITY.correlationId) AS paths
GROUP BY path, demangledName 
"""))

    def get_kernelid_paths(self) -> Cursor:

        c = self.db.cursor()
        return c.execute("""WITH cupti_kernel AS (
    SELECT correlationId,
           gridId,
           (END - START)                                            AS durationGPU,
           shortName,
           ('(' || gridX || ',' || gridY || ',' || gridZ || ')')    AS grid,
           ('(' || blockX || ',' || blockY || ',' || blockZ || ')') AS block,
           sharedMemoryExecuted
    FROM CUPTI_ACTIVITY_KIND_KERNEL
),
     cupti_activity AS (
         SELECT correlationId,
                gridId,
                value                                                               AS demangledName,
                grid,
                block,
                sharedMemoryExecuted,
                durationGPU
         FROM cupti_kernel AS CAKK
                  LEFT JOIN StringIds ON shortName = StringIds.id
         WHERE correlationId IS NOT NULL
     )
SELECT demangledName AS name,
       grid,
       block,
       sharedMemoryExecuted,
       callpath
FROM cupti_activity
         LEFT JOIN EXTRAP_RESOLVED_CALLPATHS ON EXTRAP_RESOLVED_CALLPATHS.correlationId = CUPTI_ACTIVITY.correlationId

    """)

    @deprecated
    def get_os_runtimes(self) -> List[Tuple[int, str, str, str, float, float, str, float]]:
        return []

    #         if not self._check_table_exists('OSRT_API'):
    #             return []
    #         c = self.db.cursor()
    #         return list(c.execute("""WITH osrt_activity AS (
    #     SELECT callchainId,
    #            END - START AS duration,
    #            VALUE       AS NAME
    #     FROM OSRT_API
    #              LEFT JOIN StringIds ON nameId = StringIds.id
    # )
    # SELECT paths.id,
    #        path,
    #        callpath,
    #        name,
    #        SUM(duration) AS duration
    # FROM (SELECT id,
    #              path,
    #              name,
    #              callpath,
    #              duration
    #       FROM EXTRAP_RESOLVED_CALLPATHS
    #                LEFT JOIN osrt_activity ON id = osrt_activity.callchainId) AS paths
    # GROUP BY path, name
    #     """))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()
