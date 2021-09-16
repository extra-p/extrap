# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.
from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path
from typing import Tuple, List, Generator

from extrap.util.caching import cached_property
from extrap.util.deprecation import deprecated
from extrap.util.exceptions import FileFormatError

UNIT_SEPARATOR = '\x1F'

_FILE_FORMAT_VERSION = 2


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

    def _get_extrap_format_version(self):
        if not self._check_table_exists('EXTRAP_FORMAT'):
            return 0
        return self.db.execute("SELECT version FROM EXTRAP_FORMAT").fetchone()[0]

    def _update_extrap_format_version(self, version=1):
        if self._check_table_exists('EXTRAP_FORMAT'):
            self.db.execute(f"""DROP VIEW EXTRAP_FORMAT""")
        self.db.execute(f"""CREATE VIEW EXTRAP_FORMAT AS
                    SELECT {_FILE_FORMAT_VERSION} AS version
                    FROM sqlite_master
                    LIMIT 1""")

    def _prepare_shared(self):
        with self.db:
            if self._check_table_exists('EXTRAP_RESOLVED_CALLPATHS'):
                # if necessary upgrade to newer view
                if self.db.execute("""SELECT name AS CNTREC
                    FROM pragma_table_info('EXTRAP_RESOLVED_CALLPATHS')
                    WHERE name = 'correlationId'""").fetchone() is None:
                    self.db.execute("DROP VIEW EXTRAP_RESOLVED_CALLPATHS")
                elif self._get_extrap_format_version() < 1:
                    self.db.execute("""-- Add uncorrelated node to callpaths
                    INSERT INTO EXTRAP_RESOLVED_CALLPATHS (correlationId, callpath, stackDepth)
                    VALUES (0, '[Uncorrelated]', 0)""")
                    self._convert_to_exclusive()
                    self._update_extrap_format_version()
                    return
                elif self._get_extrap_format_version() > _FILE_FORMAT_VERSION:
                    warnings.warn(f"Version {self._get_extrap_format_version()} of the Extra-P callpath data is "
                                  "not supported by this Extra-P version, this may cause problems.")
                    return
                elif not self._check_table_exists('EXTRAP_GPU_IDLE') and self._check_table_exists('NVTX_EVENTS'):
                    domain_id = self._ep_nvtx_domain_id()
                    if domain_id:
                        self._prepare_gpu_idle(domain_id)
                else:
                    return

            if self._check_table_exists('NVTX_EVENTS'):
                # if possible use extra_prof data
                domain_id = self._ep_nvtx_domain_id()
                if domain_id:
                    self._prepare_callpaths_from_extra_prof(domain_id)
                    self._convert_to_exclusive()
                    self._update_extrap_format_version()
                    self._prepare_gpu_idle(domain_id)
                    return
            else:
                self._prepare_callpaths_from_nsys_tracing()
                self._update_extrap_format_version()

    def _ep_nvtx_domain_id(self):
        domain_id = self.db.execute(
            "SELECT domainId FROM NVTX_EVENTS WHERE eventType=75 AND text='de.tu-darmstadt.parallel.extra_prof' LIMIT 1").fetchone()
        if domain_id:
            return int(domain_id[0])
        else:
            return None

    def _prepare_callpaths_from_nsys_tracing(self):
        self.db.execute("""CREATE VIEW IF NOT EXISTS EXTRAP_RESOLVED_CALLPATHS AS
WITH resolved_callchains AS (
    SELECT CUDA_CALLCHAINS.id, symbol, stackDepth, value AS name
    FROM CUDA_CALLCHAINS
             LEFT JOIN StringIds ON symbol = StringIds.id
    WHERE unresolved IS NULL
    ORDER BY CUDA_CALLCHAINS.id, stackDepth DESC
),
     callchains AS (SELECT id, MAX(stackDepth) AS stackDepth, GROUP_CONCAT(name, '->') AS callpath
                    FROM resolved_callchains
                    GROUP BY id)
SELECT correlationId,
       callchains.callpath || '->' || StringIds.value AS callpath,
       stackDepth + 1                                 AS stackDepth,
       CA.end - CA.start                              AS duration
FROM callchains
         INNER JOIN main.CUPTI_ACTIVITY_KIND_RUNTIME AS CA ON callchains.id - 1 = CA.callchainId
         LEFT JOIN StringIds ON nameId = StringIds.id
GROUP BY correlationId
         """)

    def _prepare_callpaths_from_extra_prof(self, domain_id):
        self.db.execute("""-- Create index for fast correlation
CREATE INDEX IF NOT EXISTS EXTRAP_NVTX_INDEX
    ON NVTX_EVENTS (globalTid, start, end);
                                """)

        self.db.execute(f"""CREATE TEMP VIEW EXTRAP_TEMP_CALLPATHS AS
SELECT start, end, text AS callpath, uint32Value AS depth, globalTid
FROM NVTX_EVENTS
WHERE eventType = 59
  AND domainId = {domain_id}
  AND text IS NOT NULL
""")

        self.db.execute(f"""-- Create result table including the correlations
CREATE TABLE EXTRAP_RESOLVED_CALLPATHS AS
SELECT correlationId,
       callpath || char(31) || '->' || value AS callpath,
       depth + 1                             AS stackDepth,
       CA.end - CA.start                     AS duration
FROM main.CUPTI_ACTIVITY_KIND_RUNTIME AS CA
         INNER JOIN EXTRAP_TEMP_CALLPATHS
                    ON CA.globalTid = EXTRAP_TEMP_CALLPATHS.globalTid AND
                       EXTRAP_TEMP_CALLPATHS.start = (SELECT MAX(TP.start)
                                                      FROM EXTRAP_TEMP_CALLPATHS AS TP
                                                      WHERE CA.globalTid = TP.globalTid
                                                        AND TP.start < CA.start
                                                        AND CA.end < TP.end
                       )
         LEFT JOIN StringIds ON nameId = id
UNION ALL
SELECT NULL AS correlationId, callpath, depth AS stackDepth, SUM(END - start) AS duration
FROM EXTRAP_TEMP_CALLPATHS
GROUP BY callpath""")

        self.db.execute("""-- Add uncorrelated node to callpaths
INSERT INTO EXTRAP_RESOLVED_CALLPATHS (correlationId, callpath, stackDepth)
VALUES (0, '[Uncorrelated]', 0)""")

    def _convert_to_exclusive(self):
        max_depth = self.db.execute("""-- Determines max. callpath length
SELECT MAX(stackDepth) AS max_depth
FROM EXTRAP_RESOLVED_CALLPATHS""").fetchone()
        max_depth = int(max_depth[0])

        for depth in range(max_depth + 1):
            self.db.execute("""UPDATE EXTRAP_RESOLVED_CALLPATHS
SET duration=duration - (
    SELECT SUM(temp.duration)
    FROM EXTRAP_RESOLVED_CALLPATHS AS temp
    WHERE temp.stackDepth = ?
      AND temp.callpath LIKE EXTRAP_RESOLVED_CALLPATHS.callpath || '%'
)
WHERE EXTRAP_RESOLVED_CALLPATHS.stackDepth = ?
  AND EXISTS(
        SELECT duration
        FROM EXTRAP_RESOLVED_CALLPATHS AS temp
        WHERE correlationId IS NULL
          AND temp.stackDepth = ?
          AND temp.callpath LIKE EXTRAP_RESOLVED_CALLPATHS.callpath || '%'
    )""", [depth + 1, depth, depth + 1])

    def _prepare_gpu_idle(self, domain_id):
        if self._check_table_exists('EXTRAP_GPU_IDLE'):
            return
        self.db.execute(f"""CREATE TEMP VIEW IF NOT EXISTS EXTRAP_TEMP_CALLPATHS AS
SELECT start, end, text AS callpath, uint32Value AS depth, globalTid
FROM NVTX_EVENTS
WHERE eventType = 59
  AND domainId = {domain_id}
  AND text IS NOT NULL
    """)

        self.db.executescript("""-- Create indexes for fast correlation, make sure to delete them after usage
CREATE INDEX IF NOT EXISTS EXTRAP_NVTX_INDEX_START
    ON NVTX_EVENTS (domainId, eventType, start);
CREATE INDEX IF NOT EXISTS EXTRAP_NVTX_INDEX_END
    ON NVTX_EVENTS (domainId, eventType, end);
    """)

        self.db.execute("""CREATE TEMP TABLE EXTRAP_TEMP_GPU_BUSY AS
WITH steps AS (
    SELECT ROW_NUMBER() OVER ( ORDER BY e_step ) RowNum, *
    FROM (SELECT NVTX_EVENTS.start AS e_step, correlationId
          FROM NVTX_EVENTS
                   INNER JOIN CUPTI_ACTIVITY_KIND_KERNEL AS CAKK
                              ON (NVTX_EVENTS.start BETWEEN CAKK.start AND CAKK.end)
          WHERE domainId = :domainId
            AND eventType = 59
          UNION ALL
          SELECT NVTX_EVENTS.end AS e_step, correlationId
          FROM NVTX_EVENTS
                   INNER JOIN CUPTI_ACTIVITY_KIND_KERNEL AS CAKK
                              ON (NVTX_EVENTS.end BETWEEN CAKK.start AND CAKK.end)
          WHERE domainId = :domainId
            AND eventType = 59
          UNION ALL
          SELECT DISTINCT CAKK.start AS e_step, correlationId
          FROM NVTX_EVENTS
                   INNER JOIN CUPTI_ACTIVITY_KIND_KERNEL AS CAKK
                              ON (NVTX_EVENTS.start BETWEEN CAKK.start AND CAKK.end) OR
                                 (NVTX_EVENTS.end BETWEEN CAKK.start AND CAKK.end)
          WHERE domainId = :domainId
            AND eventType = 59
          UNION ALL
          SELECT DISTINCT CAKK.end AS e_step, correlationId
          FROM NVTX_EVENTS
                   INNER JOIN CUPTI_ACTIVITY_KIND_KERNEL AS CAKK
                              ON (NVTX_EVENTS.start BETWEEN CAKK.start AND CAKK.end) OR
                                 (NVTX_EVENTS.end BETWEEN CAKK.start AND CAKK.end)
          WHERE domainId = :domainId
            AND eventType = 59
         )),
     split_up_kernels AS (
         SELECT steps.e_step AS start, steps2.e_step AS END, steps.correlationId AS correlationId
         FROM steps
                  INNER JOIN steps AS steps2
                             ON steps.RowNum + 1 = steps2.RowNum AND steps.correlationId = steps2.correlationId
     )
SELECT start, end, correlationId
FROM CUPTI_ACTIVITY_KIND_KERNEL
WHERE CUPTI_ACTIVITY_KIND_KERNEL.correlationId NOT IN
      (SELECT split_up_kernels.correlationId FROM split_up_kernels)
UNION ALL
SELECT *
FROM split_up_kernels
    """, {"domainId": 1})

        self.db.execute("""CREATE TABLE EXTRAP_GPU_IDLE AS
WITH gpu_busy_agg AS (
    SELECT callpath, SUM(EXTRAP_TEMP_GPU_BUSY.end - EXTRAP_TEMP_GPU_BUSY.start) AS duration
    FROM EXTRAP_TEMP_CALLPATHS,
         EXTRAP_TEMP_GPU_BUSY
    WHERE EXTRAP_TEMP_CALLPATHS.start = (SELECT MAX(EXTRAP_TEMP_CALLPATHS.start)
                                         FROM EXTRAP_TEMP_CALLPATHS
                                         WHERE EXTRAP_TEMP_CALLPATHS.start <= EXTRAP_TEMP_GPU_BUSY.start
                                           AND EXTRAP_TEMP_GPU_BUSY.end <= EXTRAP_TEMP_CALLPATHS.end)
    GROUP BY EXTRAP_TEMP_CALLPATHS.callpath)
SELECT EXTRAP_RESOLVED_CALLPATHS.callpath, EXTRAP_RESOLVED_CALLPATHS.duration - gpu_busy_agg.duration AS duration
FROM EXTRAP_RESOLVED_CALLPATHS
         INNER JOIN gpu_busy_agg ON gpu_busy_agg.callpath = EXTRAP_RESOLVED_CALLPATHS.callpath
WHERE correlationId IS NULL
  AND gpu_busy_agg.duration < EXTRAP_RESOLVED_CALLPATHS.duration
    """)

        self.db.executescript("""-- Drop indexes to reduce performance overhead in other methods
DROP INDEX IF EXISTS EXTRAP_NVTX_INDEX_START;
DROP INDEX IF EXISTS EXTRAP_NVTX_INDEX_END;
    """)

    @cached_property
    def _symbol_table(self):
        table = self.db.execute("""-- Extracts symbol table from StringIds
SELECT SUBSTR(value, 4, pos - 5) AS ptr, SUBSTR(value, pos) AS name
FROM (SELECT *, INSTR(SUBSTR(value, 4), char(31)) + 4 AS pos
      FROM StringIds
      WHERE id BETWEEN (SELECT id FROM StringIds WHERE value = "EXTRA_PROF_SYMBOLS")
          AND (SELECT id FROM StringIds WHERE value = "EXTRA_PROF_SYMBOLS_END")
        AND VALUE LIKE "EP" || char(31) || "%")""")
        return dict(table)

    def decode_callpath(self, callpath: str) -> str:
        if callpath[0] != UNIT_SEPARATOR:
            return callpath
        ptrs = callpath[1:].split(UNIT_SEPARATOR)
        rest = None
        if ptrs[-1].startswith('->'):
            rest = ptrs[-1]
            ptrs = ptrs[:-1]

        callpath = "->".join([self._symbol_table[p] for p in ptrs])
        if rest:
            callpath += rest
        return callpath

    def get_mem_copies(self) -> List[Tuple[int, str, str, float, int, str, float]]:
        if not self._check_table_exists('CUPTI_ACTIVITY_KIND_MEMCPY'):
            return []
        query_result = self.db.execute("""WITH cupti_memory AS (
    SELECT correlationId,
           (end - start) AS duration,
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
               WHEN 11 THEN 'UMA_HOST_TO_DEVICE'
               WHEN 12 THEN 'UMA_DEVICE_TO_HOST'
               ELSE 'UNKNOWN_COPY_KIND_' || copyKind
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
                    WHEN 11 THEN 'UMA_HOST_TO_DEVICE'
                    WHEN 12 THEN 'UMA_DEVICE_TO_HOST'
                    ELSE 'UNKNOWN_COPY_KIND_' || copyKind
                    END                                                 AS copyKind,
                NULL                                                    AS bytes,
                shortName
         FROM CUPTI_ACTIVITY_KIND_MEMCPY AS CAKS
                  INNER JOIN (SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL) AS CAKK
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
       callpath,
       demangledName       AS name,
       SUM(duration)       AS duration,
       SUM(bytes)          AS bytes,
       copyKind,
       SUM(other_duration) AS other_duration
FROM (SELECT EXTRAP_RESOLVED_CALLPATHS.correlationId,
             callpath,
             demangledName,
             duration,
             copyKind,
             other_duration,
             bytes
      FROM EXTRAP_RESOLVED_CALLPATHS
               INNER JOIN cupti_activity
                          ON EXTRAP_RESOLVED_CALLPATHS.correlationId = CUPTI_ACTIVITY.correlationId) AS paths
GROUP BY callpath, demangledName, copyKind 
    """)
        return [(correlation_id, self.decode_callpath(callpath), name, duration, bytes, copyKind, other_duration)
                for correlation_id, callpath, name, duration, bytes, copyKind, other_duration in query_result]

    def get_synchronization(self) -> List[Tuple[int, str, str, float, float, str, float]]:
        if not self._check_table_exists('CUPTI_ACTIVITY_KIND_SYNCHRONIZATION'):
            return []
        result = self.db.execute("""WITH cupti_synchronization AS (
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
       callpath,
       demangledName       AS kernelName,
       SUM(duration)       AS duration,
       SUM(durationGPU)    AS durationGPU,
       syncType,
       SUM(other_duration) AS other_duration
FROM (SELECT EXTRAP_RESOLVED_CALLPATHS.correlationId,
             callpath,
             demangledName,
             duration,
             durationGPU,
             syncType,
             other_duration
      FROM EXTRAP_RESOLVED_CALLPATHS
               INNER JOIN cupti_activity
                          ON EXTRAP_RESOLVED_CALLPATHS.correlationId = CUPTI_ACTIVITY.correlationId) AS paths
GROUP BY callpath, demangledName, syncType 
    """)
        return [(correlation_id, self.decode_callpath(callpath), name, duration, durationGPU, syncType, other_duration)
                for correlation_id, callpath, name, duration, durationGPU, syncType, other_duration in result]

    def get_kernel_runtimes(self) -> List[Tuple[int, str, str, float, float, float]]:
        if not self._check_table_exists('CUPTI_ACTIVITY_KIND_KERNEL'):
            return []
        result = self.db.execute("""WITH cupti_kernel AS (
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
       callpath,
       demangledName       AS kernelName,
       SUM(duration)       AS duration,
       SUM(durationGPU)    AS durationGPU,
       SUM(other_duration) AS other_duration
FROM (SELECT EXTRAP_RESOLVED_CALLPATHS.correlationId,
             callpath,
             demangledName,
             --(demangledName|| '<<<' || grid || ',' || block || ',' || sharedMemoryExecuted || '>>>') AS demangledName,
             duration,
             durationGPU,
             other_duration
      FROM EXTRAP_RESOLVED_CALLPATHS
               LEFT JOIN cupti_activity
                         ON EXTRAP_RESOLVED_CALLPATHS.correlationId = CUPTI_ACTIVITY.correlationId) AS paths
GROUP BY callpath, demangledName 
""")
        return [(correlation_id, self.decode_callpath(callpath), name, duration, durationGPU, other_duration)
                for correlation_id, callpath, name, duration, durationGPU, other_duration in result]

    def get_gpu_idle(self) -> List[Tuple[str, int]]:
        if not self._check_table_exists('EXTRAP_GPU_IDLE'):
            return []
        result = self.db.execute("""SELECT * FROM EXTRAP_GPU_IDLE""")
        return [(self.decode_callpath(callpath), duration) for callpath, duration in result]

    def get_kernelid_paths(self) -> Generator[Tuple[str, str, str, int, str], None, None]:
        result = self.db.execute("""WITH cupti_kernel AS (
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
         SELECT correlationId,
                gridId,
                value AS demangledName,
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
        return ((name, grid, block, sharedMem, self.decode_callpath(callpath))
                for (name, grid, block, sharedMem, callpath) in result)

    def get_peak_flops(self) -> float:
        return self.db.execute("""-- Calculates peak FLOPs
SELECT SUM(numThreadsPerWarp * coreClockRate * maxIPC * numMultiprocessors)
           AS peak_flops
FROM TARGET_INFO_CUDA_GPU
        """).fetchone()[0]

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