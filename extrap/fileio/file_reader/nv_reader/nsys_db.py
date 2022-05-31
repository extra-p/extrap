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
from extrap.util.exceptions import FileFormatError

UNIT_SEPARATOR = '\x1F'

_FILE_FORMAT_VERSION = 2


class NsysReport:
    def __init__(self, filename, number_threads_fallback=1):
        filename = Path(filename)
        self.number_threads_fallback = number_threads_fallback
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
        elif filename.suffix == ".nsight-cuprof-report" or filename.suffix == ".ncu-rep":
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
            self._create_gpu_info()
            self._create_concurrent_kernel_view()
            self._create_overlap_activities_view()

            if self._check_table_exists('EXTRAP_RESOLVED_CALLPATHS'):
                # if necessary upgrade to newer table
                if self.db.execute("""SELECT name AS CNTREC FROM pragma_table_info('EXTRAP_RESOLVED_CALLPATHS')
                                        WHERE name = 'correlationId'""").fetchone() is None:
                    self.db.execute("DROP VIEW EXTRAP_RESOLVED_CALLPATHS")
                elif self.db.execute("""SELECT name FROM sqlite_master WHERE type = 'view'
                                            AND name = 'EXTRAP_RESOLVED_CALLPATHS'""").fetchone() is not None:
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
                self._convert_to_exclusive()
                self._update_extrap_format_version()

    def _create_gpu_info(self):
        if self._check_table_exists('TARGET_INFO_CUDA_GPU'):
            self.db.execute("""CREATE VIEW IF NOT EXISTS EXTRAP_GPU_INFO AS
SELECT numMultiprocessors * numThreadsPerWarp * maxWarpsPerMultiprocessor AS maxThreads
FROM TARGET_INFO_CUDA_GPU
WHERE computeCapabilityMajor > 0
            """)
        elif self._check_table_exists('TARGET_INFO_GPU') \
                and self.db.execute("""SELECT name AS CNTREC FROM pragma_table_info('TARGET_INFO_GPU')
                                        WHERE name = 'smCount'""").fetchone() is not None:
            self.db.execute("""CREATE VIEW IF NOT EXISTS EXTRAP_GPU_INFO AS
SELECT smCount * threadsPerWarp * maxWarpsPerSm AS maxThreads
FROM TARGET_INFO_GPU
WHERE computeMajor > 0
            """)
        else:
            self.db.execute(f"""CREATE VIEW IF NOT EXISTS EXTRAP_GPU_INFO AS
SELECT {self.number_threads_fallback} AS maxThreads
FROM sqlite_master
LIMIT 1            """)

    def _create_concurrent_kernel_view(self):
        if self._check_table_exists('CUPTI_ACTIVITY_KIND_MEMSET'):
            memset_activities = """SELECT correlationId      AS correlationIdK,
           start              AS startK,
           end                AS endK,
           'MEMSET ' || value AS shortNameK,
           0                  AS roof_threads
    FROM CUPTI_ACTIVITY_KIND_MEMSET
            """
            memset_activities = "UNION ALL " + memset_activities
        else:
            memset_activities = ""
        self.db.execute(f"""-- Creates a view which contains time spans of kernel execution and the number of concurrently executing kernels
CREATE VIEW IF NOT EXISTS EXTRAP_CONCURRENT_KERNELS AS
WITH kernel_overlap_base AS (SELECT correlationId                                                     AS correlationIdK,
                                    start                                                             AS startK,
                                    end                                                               AS endK,
                                    shortName                                                         AS shortNameK,
                                    MIN(gridX * gridY * gridZ * blockX * blockY * blockZ, maxThreads) AS roof_threads
                             FROM CUPTI_ACTIVITY_KIND_KERNEL,
                                  EXTRAP_GPU_INFO
                             {memset_activities}),
     kernel_overlap_events AS ( -- A single event stream of kernel starts/ends
         SELECT *
         FROM (SELECT correlationIdK, startK AS eventK, shortNameK, 0 AS endCount, roof_threads
               FROM kernel_overlap_base
               UNION ALL
               SELECT correlationIdK, endK AS eventK, shortNameK, 1 AS endCount, roof_threads
               FROM kernel_overlap_base)
         ORDER BY eventK),
     intermediate AS ( -- The count of simultaneously executing kernels for each start/end
         SELECT base.correlationIdK,
                base.startK,
                eventK,
                base.endK,
                COUNT(eventK)                                 AS concurrency,
                GROUP_CONCAT(DISTINCT base.correlationIdK) || ',' ||
                GROUP_CONCAT(DISTINCT overlap.correlationIdK) AS correlationId,
                overlap.correlationIdK                        AS eventCorrelationId,
                SUM(base.roof_threads)                        AS roof_threads,
                overlap.roof_threads                          AS event_roof_threads,
                endCount
         FROM kernel_overlap_events AS overlap
                  INNER JOIN kernel_overlap_base AS base
                             ON eventK BETWEEN startK AND endK
         GROUP BY eventK, overlap.correlationIdK
         ORDER BY eventK),
     concurrency_per_event AS (SELECT ROW_NUMBER() OVER (ORDER BY eventK)       AS rowNum,
                                      GROUP_CONCAT(DISTINCT correlationIdK)     AS correlationIdK,
                                      startK,
                                      eventK,
                                      endK,
                                      concurrency,
                                      GROUP_CONCAT(DISTINCT correlationIdK)     AS correlationId,
                                      GROUP_CONCAT(DISTINCT eventCorrelationId) AS eventCorrelationId,
                                      roof_threads,
                                      SUM(event_roof_threads)                   AS event_roof_threads,
                                      SUM(endCount)                             AS endCount
                               FROM intermediate
                               GROUP BY eventK)
     -- The total time spend in at least one kernel, summing up the time between starts/ends
     -- not considering the simultaneously executing kernels
SELECT agg_start.eventK                           AS start,
       agg_end.eventK                             AS end,
       agg_start.concurrency - agg_start.endCount AS concurrency,
       agg_end.eventK - agg_start.eventK          AS duration,
       CASE
           WHEN agg_start.endCount > 0 THEN agg_start.roof_threads - agg_start.event_roof_threads
           ELSE agg_start.roof_threads END        AS roof_threads
FROM concurrency_per_event AS agg_end
         INNER JOIN concurrency_per_event AS agg_start
                    ON agg_start.rowNum + 1 == agg_end.rowNum
WHERE agg_start.concurrency - agg_start.endCount > 0
    """)

    def _create_overlap_activities_view(self):
        activity_tables = [
            ("CUPTI_ACTIVITY_KIND_KERNEL", """--
SELECT correlationId, start, end, value AS name, 0 AS type
FROM CUPTI_ACTIVITY_KIND_KERNEL
         LEFT JOIN StringIds ON StringIds.id = shortName"""),
            ("CUPTI_ACTIVITY_KIND_MEMCPY", """--
SELECT correlationId,
       start,
       end,
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
           END AS name,
       1       AS type
FROM CUPTI_ACTIVITY_KIND_MEMCPY"""),
            ("CUPTI_ACTIVITY_KIND_MEMSET", """--
SELECT correlationId,
       start,
       end,
       'MEMSET ' || value AS name,
       2                  AS type
FROM CUPTI_ACTIVITY_KIND_MEMSET"""),
        ]

        activity_command = " UNION ALL ".join([c for t, c in activity_tables if self._check_table_exists(t)])

        self.db.execute(f"""CREATE VIEW IF NOT EXISTS EXTRAP_GPU_ACTIVITIES AS {activity_command}""")

        self.db.execute("""CREATE VIEW IF NOT EXISTS EXTRAP_OVERLAP_ACTIVITIES AS
WITH overlap_events AS ( -- A single event stream of activity starts/ends
         SELECT correlationId, start AS eventK, 0 AS endCount
         FROM EXTRAP_GPU_ACTIVITIES
         UNION ALL
         SELECT correlationId, end AS eventK, 1 AS endCount
         FROM EXTRAP_GPU_ACTIVITIES
         ORDER BY eventK),
     intermediate AS (SELECT *, COUNT(eventK) AS concurrency
                      FROM overlap_events
                               LEFT JOIN EXTRAP_GPU_ACTIVITIES AS base
                                         ON eventK BETWEEN start AND end
                      GROUP BY eventK, overlap_events.correlationId),
     concurrency_per_event AS (SELECT ROW_NUMBER() OVER (ORDER BY eventK ) AS rowNum,
                                      eventK,
                                      concurrency,
                                      start,
                                      end,
                                      SUM(endCount)                        AS endCount
                               FROM intermediate
                               GROUP BY eventK)
SELECT agg_start.eventK                           AS start,
       agg_end.eventK                             AS end,
       agg_start.concurrency - agg_start.endCount AS concurrency,
       agg_end.eventK - agg_start.eventK          AS duration
FROM concurrency_per_event AS agg_end
         INNER JOIN concurrency_per_event AS agg_start
                    ON agg_start.rowNum + 1 == agg_end.rowNum
WHERE agg_start.concurrency - agg_start.endCount > 0
        """)

    def _ep_nvtx_domain_id(self):
        domain_id = self.db.execute(
            "SELECT domainId FROM NVTX_EVENTS WHERE eventType=75 AND text='de.tu-darmstadt.parallel.extra_prof' LIMIT 1").fetchone()
        if domain_id:
            return int(domain_id[0])
        else:
            return None

    def _prepare_callpaths_from_nsys_tracing(self):
        self.db.execute("""CREATE TABLE IF NOT EXISTS EXTRAP_RESOLVED_CALLPATHS AS
WITH resolved_callchains AS (
    SELECT CUDA_CALLCHAINS.id, symbol, stackDepth, value AS name
    FROM CUDA_CALLCHAINS
             LEFT JOIN StringIds ON symbol = StringIds.id
    WHERE unresolved IS NULL
    ORDER BY CUDA_CALLCHAINS.id, stackDepth DESC
),
     callchains AS (SELECT id, MAX(stackDepth) AS stackDepth, GROUP_CONCAT(REPLACE(name,'->','- >'), '->') AS callpath
                    FROM resolved_callchains
                    GROUP BY id)
SELECT correlationId,
       callchains.callpath || '->' || REPLACE(StringIds.value,'->','- >') AS callpath,
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
SELECT NULL AS correlationId, callpath, depth AS stackDepth, SUM(end - start) AS duration
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
         SELECT steps.e_step AS start, steps2.e_step AS end, steps.correlationId AS correlationId
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
SELECT SUBSTR(value, 4, pos - 5) AS ptr, REPLACE(SUBSTR(value, pos),'->','- >') AS name
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

    def get_mem_copies(self) -> List[Tuple[int, str, str, float, int, str, bool, float, float]]:
        if not self._check_table_exists('CUPTI_ACTIVITY_KIND_MEMCPY'):
            return []
        query_result = self.db.execute("""WITH cupti_memory AS ( -- memory copy operations
    SELECT correlationId,
           (end - start)                AS duration,
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
               END                      AS copyKind,
           srcKind == 0 OR dstKind == 0 AS pageable,
           bytes
    FROM CUPTI_ACTIVITY_KIND_MEMCPY),
     individual_kernel_overlaps AS ( -- All the kernels executing simultaneously to the copy operations
         SELECT correlationId,
                MAX(CA.start, CO.start) AS startO,
                MIN(CA.end, CO.end)     AS endO,
                shortName
         FROM CUPTI_ACTIVITY_KIND_MEMCPY AS CA
                  INNER JOIN (SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL) AS CO
                             ON (CO.start BETWEEN CA.start AND CA.end OR CO.end BETWEEN CA.start AND CA.end) OR
                                (CA.start BETWEEN CO.start AND CO.end AND CA.end BETWEEN CO.start AND CO.end)),
     kernel_overlap AS (SELECT correlationId,
                               SUM(MIN(CA.end, CO.end) - MAX(CA.start, CO.start)) overlap_duration
                        FROM CUPTI_ACTIVITY_KIND_MEMCPY AS CA
                                 INNER JOIN (SELECT start, end FROM EXTRAP_CONCURRENT_KERNELS) AS CO
                                            ON (CO.start BETWEEN CA.start AND CA.end OR
                                                CO.end BETWEEN CA.start AND CA.end) OR
                                               (CA.start BETWEEN CO.start AND CO.end AND
                                                CA.end BETWEEN CO.start AND CO.end)
                        GROUP BY correlationId),
     base_memory_overlap AS (SELECT CA.correlationId         AS correlationId1,
                                    CAO.correlationId        AS correlationId2,
                                    MAX(CA.start, CAO.start) AS startO,
                                    MIN(CA.end, CAO.end)     AS endO,
                                    CA.start,
                                    CAO.start,
                                    CA.end,
                                    CAO.end,
                                    CA.copyKind              AS copyKind1,
                                    CAO.copyKind             AS copyKind2
                             FROM CUPTI_ACTIVITY_KIND_MEMCPY AS CA
                                      INNER JOIN (SELECT start, end, copyKind, correlationId
                                                  FROM CUPTI_ACTIVITY_KIND_MEMCPY) AS CAO
                                                 ON (CAO.start BETWEEN CA.start AND CA.end OR
                                                     CAO.end BETWEEN CA.start AND CA.end) AND
                                                    CAO.correlationId != CA.correlationId),
     individual_memory_overlaps AS (SELECT correlationId1  AS correlationId,
                                          startO,
                                          endO,
                                          (endO - startO) AS duration,
                                          copyKind1       AS copyKind
                                   FROM base_memory_overlap
                                   UNION ALL
                                   SELECT correlationId2  AS correlationId,
                                          startO,
                                          endO,
                                          (endO - startO) AS duration,
                                          copyKind2       AS copyKind
                                   FROM base_memory_overlap),
     memory_overlap AS (SELECT correlationId,
                               CASE
                                   WHEN CO.end IS NULL AND CO.start IS NULL THEN
                                       SUM(duration / 2)
                                   ELSE
                                       SUM((duration - (MIN(CO.end, CA.endO) - MAX(CO.start, CA.startO))) /
                                           2) END AS overlap_duration
                        FROM individual_memory_overlaps AS CA
                                 LEFT JOIN (SELECT start, end
                                            FROM EXTRAP_CONCURRENT_KERNELS) AS CO
                                           ON (CO.start BETWEEN CA.startO AND CA.endO OR
                                               CO.end BETWEEN CA.startO AND CA.endO) OR
                                              (CA.startO BETWEEN CO.start AND CO.end AND
                                               CA.endO BETWEEN CO.start AND CO.end)),
     overlap AS (SELECT correlationId, SUM(overlap_duration) AS overlap_duration
                 FROM (SELECT * FROM kernel_overlap UNION ALL SELECT * FROM memory_overlap)
                 GROUP BY correlationId),
     cupti_activity AS (SELECT cupti_memory.correlationId,
                               NULL AS demangledName,
                               copyKind,
                               pageable,
                               bytes,
                               duration,
                               overlap_duration
                        FROM cupti_memory
                                 LEFT JOIN overlap
                                           ON cupti_memory.correlationId = overlap.correlationId),
     individual_overlaps AS (SELECT correlationId, startO, endO, value
                             FROM individual_kernel_overlaps
                                      LEFT JOIN StringIds ON shortName = StringIds.id
                             UNION ALL
                             SELECT correlationId,
                                    startO,
                                    endO,
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
                                        END AS value
                             FROM individual_memory_overlaps),
     cupti_activity_overlap AS (SELECT IO.correlationId,
                                       value         AS demangledName,
                                       copyKind,
                                       pageable,
                                       NULL          AS bytes,
                                       NULL          AS duration,
                                       endO - startO AS overlap_duration
                                FROM individual_overlaps AS IO
                                         LEFT JOIN cupti_activity
                                                   ON cupti_activity.correlationId = IO.correlationId)
SELECT paths.correlationId,
       callpath,
       demangledName                              AS name,
       SUM(duration)                              AS duration,
       SUM(bytes)                                 AS bytes,
       copyKind,
       pageable OR NOT (callpath GLOB '*Async_*') AS blocking,
       SUM(gpu_duration)                          AS gpu_duration,
       SUM(overlap_duration)                      AS overlap_duration
FROM (SELECT EXTRAP_RESOLVED_CALLPATHS.correlationId,
             callpath,
             demangledName,
             EXTRAP_RESOLVED_CALLPATHS.duration,
             copyKind,
             pageable,
             CA.duration AS gpu_duration,
             bytes,
             overlap_duration
      FROM EXTRAP_RESOLVED_CALLPATHS
               INNER JOIN (SELECT * FROM cupti_activity UNION ALL SELECT * FROM cupti_activity_overlap) AS CA
                          ON EXTRAP_RESOLVED_CALLPATHS.correlationId = CA.correlationId) AS paths
GROUP BY callpath, demangledName, copyKind, blocking 
    """)
        return [
            (correlation_id, self.decode_callpath(callpath), name, duration, bytes, copyKind, blocking, gpu_duration,
             overlap_duration)
            for correlation_id, callpath, name, duration, bytes, copyKind, blocking, gpu_duration, overlap_duration in
            query_result]

    def get_synchronization(self) -> List[Tuple[int, str, str, float, str, float, float]]:
        if not self._check_table_exists('CUPTI_ACTIVITY_KIND_SYNCHRONIZATION'):
            return []
        result = self.db.execute("""WITH cupti_synchronization AS (SELECT correlationId,
                                      start,
                                      end,
                                      (end - start) AS duration,
                                      CASE syncType
                                          WHEN 0 THEN 'UNKNOWN'
                                          WHEN 1 THEN 'EVENT_SYNCHRONIZE'
                                          WHEN 2 THEN 'STREAM_WAIT_EVENT'
                                          WHEN 3 THEN 'STREAM_SYNCHRONIZE'
                                          WHEN 4 THEN 'CONTEXT_SYNCHRONIZE'
                                          END       AS syncType
                               FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION),
     cupti_sync_overlap AS (SELECT CAKS.correlationId,
                                   (MIN(CAKS.end, GA.end) - MAX(CAKS.start, GA.start)) AS overlap_duration,
                                   syncType,
                                   name
                            FROM cupti_synchronization AS CAKS
                                     LEFT JOIN EXTRAP_GPU_ACTIVITIES AS GA
                                               ON GA.correlationId != CAKS.correlationId AND
                                                  GA.end BETWEEN CAKS.start AND CAKS.end
         -- only end needs to be within the operation, to be considered overlapping. 
         -- All actions running after the end of the operation are not synced with it
     ),
     overlap AS (SELECT correlationId, SUM((MIN(CA.end, CO.end) - MAX(CA.start, CO.start))) AS overlap_duration
                 FROM cupti_synchronization AS CA
                          INNER JOIN EXTRAP_OVERLAP_ACTIVITIES AS CO ON (CO.start BETWEEN CA.start AND CA.end OR
                                                                         CO.end BETWEEN CA.start AND CA.end)
                 GROUP BY correlationId),
     cupti_activity AS (SELECT cupti_synchronization.correlationId,
                               duration,
                               syncType,
                               NULL                  AS name,
                               SUM(overlap_duration) AS overlap_duration
                        FROM cupti_synchronization
                                 LEFT JOIN overlap ON overlap.correlationId = cupti_synchronization.correlationId
                        GROUP BY cupti_synchronization.correlationId
                        UNION ALL
                        SELECT correlationId, NULL AS duration, syncType, name, overlap_duration
                        FROM cupti_sync_overlap)
SELECT paths.correlationId,
       callpath,
       name                  AS overlap_name,
       SUM(duration)         AS duration,
       syncType,
       SUM(other_duration)   AS other_duration,
       SUM(overlap_duration) AS overlap_duration
FROM (SELECT cupti_activity.correlationId,
             callpath,
             name,
             EXTRAP_RESOLVED_CALLPATHS.duration,
             syncType,
             cupti_activity.duration AS other_duration,
             overlap_duration
      FROM EXTRAP_RESOLVED_CALLPATHS
               INNER JOIN cupti_activity
                          ON EXTRAP_RESOLVED_CALLPATHS.correlationId = cupti_activity.correlationId) AS paths
GROUP BY callpath, name, syncType 
    """)
        return [
            (correlation_id, self.decode_callpath(callpath), name, duration, syncType, other_duration, overlap_duration)
            for correlation_id, callpath, name, duration, syncType, other_duration, overlap_duration in result]

    def get_mem_sets(self) -> List[Tuple[int, str, str, float, int, bool, float, float]]:
        if not self._check_table_exists('CUPTI_ACTIVITY_KIND_MEMSET'):
            return []
        query_result = self.db.execute("""WITH cupti_memory AS (SELECT correlationId,
                             start,
                             end,
                             (end - start) AS duration,
                             memKind == 0  AS pageable,
                             bytes
                      FROM CUPTI_ACTIVITY_KIND_MEMSET),
     cupti_memory_overlap AS (SELECT CA.correlationId,
                                     (MIN(CA.end, GO.end) - MAX(CA.start, GO.start)) AS overlap_duration,
                                     NULL                                            AS pageable,
                                     name
                              FROM CUPTI_ACTIVITY_KIND_MEMSET AS CA
                                       INNER JOIN EXTRAP_GPU_ACTIVITIES AS GO
                                                  ON GO.type!=1 AND CA.correlationId != GO.correlationId AND
                                                     ((GO.start BETWEEN CA.start AND CA.end OR
                                                       GO.end BETWEEN CA.start AND CA.end) OR
                                                      (CA.start BETWEEN GO.start AND GO.end AND
                                                       CA.end BETWEEN GO.start AND GO.end))),
     overlap AS (SELECT correlationId, SUM((MIN(CA.end, GO.end) - MAX(CA.start, GO.start))) AS overlap_duration
                 FROM cupti_memory AS CA
                          INNER JOIN EXTRAP_CONCURRENT_KERNELS AS GO ON (GO.start BETWEEN CA.start AND CA.end OR
                                                                         GO.end BETWEEN CA.start AND CA.end) OR
                                                                        (CA.start BETWEEN GO.start AND GO.end AND
                                                                         CA.end BETWEEN GO.start AND GO.end)
                 WHERE (GO.start != CA.start AND GO.end != CA.end)
                    OR go.concurrency > 1
                 GROUP BY correlationId),
     cupti_activity AS (SELECT cupti_memory.correlationId,
                               duration,
                               pageable,
                               bytes,
                               NULL                  AS name,
                               SUM(overlap_duration) AS overlap_duration
                        FROM cupti_memory
                                 LEFT JOIN overlap ON overlap.correlationId = cupti_memory.correlationId
                        GROUP BY cupti_memory.correlationId
                        UNION ALL
                        SELECT correlationId, NULL AS duration, pageable, NULL AS bytes, name, overlap_duration
                        FROM cupti_memory_overlap)
SELECT paths.correlationId,
       callpath,
       name,
       SUM(duration)                              AS duration,
       SUM(bytes)                                 AS bytes,
       pageable OR NOT (callpath GLOB '*Async_*') AS blocking,
       SUM(other_duration)                        AS other_duration,
       SUM(overlap_duration)                      AS overlap_duration
FROM (SELECT EXTRAP_RESOLVED_CALLPATHS.correlationId,
             callpath,
             name,
             EXTRAP_RESOLVED_CALLPATHS.duration,
             pageable,
             cupti_activity.duration AS other_duration,
             bytes,
             overlap_duration
      FROM EXTRAP_RESOLVED_CALLPATHS
               INNER JOIN cupti_activity
                          ON EXTRAP_RESOLVED_CALLPATHS.correlationId = CUPTI_ACTIVITY.correlationId) AS paths
GROUP BY callpath, name, blocking 
        """)
        return [
            (correlation_id, self.decode_callpath(callpath), name, duration, bytes, blocking, other_duration,
             overlap_duration)
            for correlation_id, callpath, name, duration, bytes, blocking, other_duration, overlap_duration in
            query_result]

    def get_kernel_runtimes(self) -> List[Tuple[int, str, str, float, float, float, float]]:
        if not self._check_table_exists('CUPTI_ACTIVITY_KIND_KERNEL'):
            return []
        result = self.db.execute("""WITH cupti_kernel AS (SELECT correlationId,
                             NULL                                                              AS duration,
                             (end - start)                                                     AS durationGPU,
                             start,
                             end,
                             shortName,
                             ('(' || gridX || ',' || gridY || ',' || gridZ || ')')             AS grid,
                             ('(' || blockX || ',' || blockY || ',' || blockZ || ')')          AS block,
                             MIN(gridX * gridY * gridZ * blockX * blockY * blockZ, maxThreads) AS roof_threads,
                             sharedMemoryExecuted
                      FROM CUPTI_ACTIVITY_KIND_KERNEL,
                           EXTRAP_GPU_INFO),
     cupti_activity AS (SELECT correlationId,
                               value         AS demangledName,
                               grid,
                               block,
                               sharedMemoryExecuted,
                               durationGPU,
                               CAKK.duration AS other_duration
                        FROM cupti_kernel AS CAKK
                                 LEFT JOIN StringIds ON shortName = StringIds.id),
     overlap AS (SELECT correlationId,
                        SUM(CO.duration * (CO.roof_threads - CA.roof_threads) / CO.roof_threads) AS overlap_duration
                 FROM cupti_kernel AS CA
                          INNER JOIN EXTRAP_CONCURRENT_KERNELS AS CO
                                     ON CO.start BETWEEN CA.start AND CA.end AND CO.end BETWEEN CA.start AND CA.end
                 GROUP BY correlationId)
SELECT paths.correlationId,
       callpath,
       demangledName         AS kernelName,
       SUM(duration)         AS duration,
       SUM(durationGPU)      AS durationGPU,
       SUM(other_duration)   AS other_duration,
       SUM(overlap_duration) AS overlap_duaration
FROM (SELECT EXTRAP_RESOLVED_CALLPATHS.correlationId,
             callpath,
             demangledName,
             --(demangledName|| '<<<' || grid || ',' || block || ',' || sharedMemoryExecuted || '>>>') AS demangledName,
             duration,
             durationGPU,
             other_duration,
             overlap_duration
      FROM cupti_activity
               LEFT JOIN EXTRAP_RESOLVED_CALLPATHS
                         ON EXTRAP_RESOLVED_CALLPATHS.correlationId = CUPTI_ACTIVITY.correlationId
               LEFT JOIN overlap ON cupti_activity.correlationId = overlap.correlationId
      WHERE NOT (demangledName GLOB 'cudaFree*' OR demangledName GLOB 'cudaMalloc*' OR
                 demangledName GLOB 'cudaHostAlloc*')) AS paths
GROUP BY callpath, demangledName 
""")
        return [(correlation_id, self.decode_callpath(callpath), name, duration, durationGPU, other_duration,
                 overlap_duration)
                for correlation_id, callpath, name, duration, durationGPU, other_duration, overlap_duration in result]

    def get_mem_alloc_free(self) -> List[Tuple[int, str, str, float, bool, bool, float]]:
        if not self._check_table_exists('CUPTI_ACTIVITY_KIND_KERNEL'):
            return []
        result = self.db.execute("""WITH mem_alloc_activities AS (SELECT correlationId,
                                     start,
                                     end,
                                     (end - start)            AS duration,
                                     value NOT LIKE '%async%' AS blocking,
                                     value LIKE '%host%'      AS host
                              FROM main.CUPTI_ACTIVITY_KIND_RUNTIME AS CA
                                       LEFT JOIN StringIds ON nameId = id
                              WHERE value GLOB 'cudaFree*'
                                 OR value GLOB 'cudaMalloc*'
                                 OR value GLOB 'cudaHostAlloc*'),
     cupti_sync_overlap AS (SELECT CA.correlationId,
                                   blocking,
                                   host,
                                   (MIN(CA.end, GA.end) - MAX(CA.start, GA.start)) AS overlap_duration,
                                   name
                            FROM mem_alloc_activities AS CA
                                     LEFT JOIN EXTRAP_GPU_ACTIVITIES AS GA
                                               ON GA.correlationId != CA.correlationId AND
                                                  (GA.end BETWEEN CA.start AND CA.end)
                                 -- only end needs to be within the operation, to be considered overlapping. 
                                 -- All actions running after the end of the operation are not synced with it
                            WHERE NOT host),
     overlap AS (SELECT correlationId, SUM((MIN(CA.end, GO.end) - MAX(CA.start, GO.start))) AS overlap_duration
                 FROM mem_alloc_activities AS CA
                          INNER JOIN EXTRAP_OVERLAP_ACTIVITIES AS GO ON (GO.start BETWEEN CA.start AND CA.end OR
                                                                         GO.end BETWEEN CA.start AND CA.end)
                 WHERE NOT host
                 GROUP BY correlationId),
     cupti_activity AS (SELECT mem_alloc_activities.correlationId,
                               duration,
                               blocking,
                               host,
                               NULL                  AS name,
                               SUM(overlap_duration) AS overlap_duration
                        FROM mem_alloc_activities
                                 LEFT JOIN overlap ON overlap.correlationId = mem_alloc_activities.correlationId
                        GROUP BY mem_alloc_activities.correlationId
                        UNION ALL
                        SELECT correlationId, NULL AS duration, blocking, host, name, overlap_duration
                        FROM cupti_sync_overlap)
SELECT paths.correlationId,
       callpath,
       name,
       SUM(duration)         AS duration,
       blocking,
       host,
       SUM(overlap_duration) AS overlap_duration
FROM (SELECT EXTRAP_RESOLVED_CALLPATHS.correlationId,
             callpath,
             name,
             EXTRAP_RESOLVED_CALLPATHS.duration,
             blocking,
             host,
             cupti_activity.duration AS other_duration,
             overlap_duration
      FROM EXTRAP_RESOLVED_CALLPATHS
               INNER JOIN cupti_activity
                          ON EXTRAP_RESOLVED_CALLPATHS.correlationId = cupti_activity.correlationId) AS paths
GROUP BY callpath, name""")
        return [(correlation_id, self.decode_callpath(callpath), name, duration, blocking, host, overlap_duration)
                for correlation_id, callpath, name, duration, blocking, host, overlap_duration in result]

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

    def get_cpu_times(self) -> List[Tuple[str, float]]:
        result = self.db.execute("""
        SELECT callpath,duration FROM EXTRAP_RESOLVED_CALLPATHS WHERE correlationId IS NULL
        """)
        return [(self.decode_callpath(callpath), duration)
                for callpath, duration in result]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()
