# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021-2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import unittest
from itertools import groupby
from pathlib import Path

from extrap.entities.metric import Metric
from extrap.fileio.file_reader.nv_reader import NsightFileReader, NsysReport
from extrap.modelers.model_generator import ModelGenerator


class TestLoadNsightFiles(unittest.TestCase):

    def test_load_sampled_callchains_only_time(self):
        experiment = NsightFileReader().read_experiment('data/nsight/sampled_callchains', only_time=True)
        self.assertNotIn(Metric('device__attribute_max_threads_per_block'), experiment.metrics)
        self.assertSetEqual(set(experiment.metrics), {Metric('time')})

    def test_load_sampled_callchains_without_device_attributes(self):
        experiment = NsightFileReader().read_experiment('data/nsight/sampled_callchains')
        self.assertNotIn(Metric('device__attribute_max_threads_per_block'), experiment.metrics)
        self.assertSetEqual(set(experiment.metrics),
                            {Metric(name) for name in
                             ['time', 'gpu__time_duration.sum', 'launch__block_dim_x', 'launch__block_dim_y',
                              'launch__block_dim_z', 'launch__block_size', 'launch__context_id', 'launch__device_id',
                              'launch__function_pcs', 'launch__grid_dim_x',
                              'launch__grid_dim_y', 'launch__grid_dim_z', 'launch__grid_size',
                              'launch__occupancy_limit_blocks', 'launch__occupancy_limit_registers',
                              'launch__occupancy_limit_shared_mem', 'launch__occupancy_limit_warps',
                              'launch__occupancy_per_block_size', 'launch__occupancy_per_register_count',
                              'launch__occupancy_per_shared_mem_size',
                              'launch__registers_per_thread',
                              'launch__registers_per_thread_allocated',
                              'launch__shared_mem_config_size', 'launch__shared_mem_per_block',
                              'launch__shared_mem_per_block_allocated',
                              'launch__shared_mem_per_block_driver',
                              'launch__shared_mem_per_block_dynamic',
                              'launch__shared_mem_per_block_static', 'launch__stream_id',
                              'launch__thread_count', 'launch__waves_per_multiprocessor',
                              'profiler__perfworks_session_reuse', 'profiler__replayer_bytes_mem_accessible.avg',
                              'profiler__replayer_bytes_mem_accessible.max',
                              'profiler__replayer_bytes_mem_accessible.min',
                              'profiler__replayer_bytes_mem_accessible.sum',
                              'profiler__replayer_bytes_mem_backed_up.avg',
                              'profiler__replayer_bytes_mem_backed_up.max',
                              'profiler__replayer_bytes_mem_backed_up.min',
                              'profiler__replayer_bytes_mem_backed_up.sum', 'profiler__replayer_passes',
                              'profiler__replayer_passes_type_warmup',
                              'sm__inst_executed.avg', 'sm__inst_executed.max', 'sm__inst_executed.min',
                              'sm__inst_executed.sum', 'sm__maximum_warps_avg_per_active_cycle',
                              'sm__maximum_warps_per_active_cycle_pct', 'sm__sass_inst_executed_op_ld.avg',
                              'sm__sass_inst_executed_op_ld.max', 'sm__sass_inst_executed_op_ld.min',
                              'sm__sass_inst_executed_op_ld.sum', 'sm__sass_inst_executed_op_st.avg',
                              'sm__sass_inst_executed_op_st.max', 'sm__sass_inst_executed_op_st.min',
                              'sm__sass_inst_executed_op_st.sum', 'smsp__maximum_warps_avg_per_active_cycle']
                             })

    def test_load_sampled_callchains_with_device_attributes(self):
        reader = NsightFileReader()
        reader.ignore_device_attributes = False
        experiment = reader.read_experiment('data/nsight/sampled_callchains')
        self.assertSetEqual(set(experiment.metrics),
                            {Metric(name) for name in
                             ['time', 'gpu__time_duration.sum', 'device__attribute_architecture',
                              'device__attribute_async_engine_count', 'device__attribute_can_flush_remote_writes',
                              'device__attribute_can_map_host_memory', 'device__attribute_can_tex2d_gather',
                              'device__attribute_can_use_64_bit_stream_mem_ops',
                              'device__attribute_can_use_host_pointer_for_registered_mem',
                              'device__attribute_can_use_stream_mem_ops',
                              'device__attribute_can_use_stream_wait_value_nor', 'device__attribute_chip',
                              'device__attribute_clock_rate', 'device__attribute_compute_capability_major',
                              'device__attribute_compute_capability_minor', 'device__attribute_compute_mode',
                              'device__attribute_compute_preemption_supported',
                              'device__attribute_concurrent_kernels', 'device__attribute_concurrent_managed_access',
                              'device__attribute_cooperative_launch',
                              'device__attribute_cooperative_multi_device_launch', 'device__attribute_device_index',
                              'device__attribute_direct_managed_mem_access_from_host',
                              'device__attribute_ecc_enabled',
                              'device__attribute_fb_bus_width', 'device__attribute_fbp_count',
                              'device__attribute_generic_compression_supported',
                              'device__attribute_global_l1_cache_supported',
                              'device__attribute_global_memory_bus_width',
                              'device__attribute_gpu_direct_rdma_with_cuda_vmm_supported',
                              'device__attribute_gpu_overlap', 'device__attribute_gpu_pci_device_id',
                              'device__attribute_gpu_pci_ext_device_id',
                              'device__attribute_gpu_pci_ext_downstream_link_rate',
                              'device__attribute_gpu_pci_ext_downstream_link_width',
                              'device__attribute_gpu_pci_ext_gen', 'device__attribute_gpu_pci_ext_gpu_gen',
                              'device__attribute_gpu_pci_ext_gpu_link_rate',
                              'device__attribute_gpu_pci_ext_gpu_link_width',
                              'device__attribute_gpu_pci_revision_id', 'device__attribute_gpu_pci_sub_system_id',
                              'device__attribute_handle_type_posix_file_descriptor_supported',
                              'device__attribute_handle_type_win32_handle_supported',
                              'device__attribute_handle_type_win32_kmt_handle_supported',
                              'device__attribute_host_native_atomic_supported',
                              'device__attribute_host_register_supported', 'device__attribute_implementation',
                              'device__attribute_integrated', 'device__attribute_kernel_exec_timeout',
                              'device__attribute_l2_cache_size', 'device__attribute_l2s_count',
                              'device__attribute_limits_max_cta_per_sm',
                              'device__attribute_local_l1_cache_supported', 'device__attribute_managed_memory',
                              'device__attribute_max_access_policy_window_size', 'device__attribute_max_block_dim_x',
                              'device__attribute_max_block_dim_y', 'device__attribute_max_block_dim_z',
                              'device__attribute_max_blocks_per_multiprocessor',
                              'device__attribute_max_gpu_frequency_khz', 'device__attribute_max_grid_dim_x',
                              'device__attribute_max_grid_dim_y', 'device__attribute_max_grid_dim_z',
                              'device__attribute_max_ipc_per_multiprocessor',
                              'device__attribute_max_ipc_per_scheduler', 'device__attribute_max_mem_frequency_khz',
                              'device__attribute_max_persisting_l2_cache_size', 'device__attribute_max_pitch',
                              'device__attribute_max_registers_per_block',
                              'device__attribute_max_registers_per_multiprocessor',
                              'device__attribute_max_registers_per_thread',
                              'device__attribute_max_shared_memory_per_block',
                              'device__attribute_max_shared_memory_per_block_optin',
                              'device__attribute_max_shared_memory_per_multiprocessor',
                              'device__attribute_max_threads_per_block',
                              'device__attribute_max_threads_per_multiprocessor',
                              'device__attribute_max_warps_per_multiprocessor',
                              'device__attribute_max_warps_per_scheduler',
                              'device__attribute_maximum_surface1d_layered_layers',
                              'device__attribute_maximum_surface1d_layered_width',
                              'device__attribute_maximum_surface1d_width',
                              'device__attribute_maximum_surface2d_height',
                              'device__attribute_maximum_surface2d_layered_height',
                              'device__attribute_maximum_surface2d_layered_layers',
                              'device__attribute_maximum_surface2d_layered_width',
                              'device__attribute_maximum_surface2d_width',
                              'device__attribute_maximum_surface3d_depth',
                              'device__attribute_maximum_surface3d_height',
                              'device__attribute_maximum_surface3d_width',
                              'device__attribute_maximum_surfacecubemap_layered_layers',
                              'device__attribute_maximum_surfacecubemap_layered_width',
                              'device__attribute_maximum_surfacecubemap_width',
                              'device__attribute_maximum_texture1d_layered_layers',
                              'device__attribute_maximum_texture1d_layered_width',
                              'device__attribute_maximum_texture1d_linear_width',
                              'device__attribute_maximum_texture1d_mipmapped_width',
                              'device__attribute_maximum_texture1d_width',
                              'device__attribute_maximum_texture2d_gather_height',
                              'device__attribute_maximum_texture2d_gather_width',
                              'device__attribute_maximum_texture2d_height',
                              'device__attribute_maximum_texture2d_layered_height',
                              'device__attribute_maximum_texture2d_layered_layers',
                              'device__attribute_maximum_texture2d_layered_width',
                              'device__attribute_maximum_texture2d_linear_height',
                              'device__attribute_maximum_texture2d_linear_pitch',
                              'device__attribute_maximum_texture2d_linear_width',
                              'device__attribute_maximum_texture2d_mipmapped_height',
                              'device__attribute_maximum_texture2d_mipmapped_width',
                              'device__attribute_maximum_texture2d_width',
                              'device__attribute_maximum_texture3d_depth',
                              'device__attribute_maximum_texture3d_depth_alternate',
                              'device__attribute_maximum_texture3d_height',
                              'device__attribute_maximum_texture3d_height_alternate',
                              'device__attribute_maximum_texture3d_width',
                              'device__attribute_maximum_texture3d_width_alternate',
                              'device__attribute_maximum_texturecubemap_layered_layers',
                              'device__attribute_maximum_texturecubemap_layered_width',
                              'device__attribute_maximum_texturecubemap_width',
                              'device__attribute_memory_clock_rate', 'device__attribute_memory_pools_supported',
                              'device__attribute_multi_gpu_board', 'device__attribute_multi_gpu_board_group_id',
                              'device__attribute_multiprocessor_count', 'device__attribute_num_l2s_per_fbp',
                              'device__attribute_num_schedulers_per_multiprocessor',
                              'device__attribute_num_tex_per_multiprocessor',
                              'device__attribute_pageable_memory_access',
                              'device__attribute_pageable_memory_access_uses_host_page_tables',
                              'device__attribute_pci_bus_id', 'device__attribute_pci_device_id',
                              'device__attribute_pci_domain_id', 'device__attribute_ram_location',
                              'device__attribute_ram_type', 'device__attribute_reserved_shared_memory_per_block',
                              'device__attribute_sass_level',
                              'device__attribute_single_to_double_precision_perf_ratio',
                              'device__attribute_sparse_cuda_array_supported',
                              'device__attribute_stream_priorities_supported', 'device__attribute_surface_alignment',
                              'device__attribute_tcc_driver', 'device__attribute_texture_alignment',
                              'device__attribute_texture_pitch_alignment', 'device__attribute_total_constant_memory',
                              'device__attribute_total_memory', 'device__attribute_unified_addressing',
                              'device__attribute_virtual_address_management_supported',
                              'device__attribute_warp_size', 'launch__block_dim_x', 'launch__block_dim_y',
                              'launch__block_dim_z', 'launch__block_size', 'launch__context_id', 'launch__device_id',
                              'launch__function_pcs', 'launch__grid_dim_x',
                              'launch__grid_dim_y', 'launch__grid_dim_z', 'launch__grid_size',
                              'launch__occupancy_limit_blocks', 'launch__occupancy_limit_registers',
                              'launch__occupancy_limit_shared_mem', 'launch__occupancy_limit_warps',
                              'launch__occupancy_per_block_size', 'launch__occupancy_per_register_count',
                              'launch__occupancy_per_shared_mem_size',
                              'launch__registers_per_thread',
                              'launch__registers_per_thread_allocated',
                              'launch__shared_mem_config_size', 'launch__shared_mem_per_block',
                              'launch__shared_mem_per_block_allocated',
                              'launch__shared_mem_per_block_driver',
                              'launch__shared_mem_per_block_dynamic',
                              'launch__shared_mem_per_block_static', 'launch__stream_id',
                              'launch__thread_count', 'launch__waves_per_multiprocessor', 'nvlink__bandwidth',
                              'nvlink__count_logical', 'nvlink__count_physical', 'nvlink__dev0type',
                              'nvlink__dev1type',
                              'profiler__perfworks_session_reuse', 'profiler__replayer_bytes_mem_accessible.avg',
                              'profiler__replayer_bytes_mem_accessible.max',
                              'profiler__replayer_bytes_mem_accessible.min',
                              'profiler__replayer_bytes_mem_accessible.sum',
                              'profiler__replayer_bytes_mem_backed_up.avg',
                              'profiler__replayer_bytes_mem_backed_up.max',
                              'profiler__replayer_bytes_mem_backed_up.min',
                              'profiler__replayer_bytes_mem_backed_up.sum', 'profiler__replayer_passes',
                              'profiler__replayer_passes_type_warmup',
                              'sm__inst_executed.avg', 'sm__inst_executed.max', 'sm__inst_executed.min',
                              'sm__inst_executed.sum', 'sm__maximum_warps_avg_per_active_cycle',
                              'sm__maximum_warps_per_active_cycle_pct', 'sm__sass_inst_executed_op_ld.avg',
                              'sm__sass_inst_executed_op_ld.max', 'sm__sass_inst_executed_op_ld.min',
                              'sm__sass_inst_executed_op_ld.sum', 'sm__sass_inst_executed_op_st.avg',
                              'sm__sass_inst_executed_op_st.max', 'sm__sass_inst_executed_op_st.min',
                              'sm__sass_inst_executed_op_st.sum', 'smsp__maximum_warps_avg_per_active_cycle']
                             })

    def test_load_and_model_sampled_callchains(self):
        experiment = NsightFileReader().read_experiment('data/nsight/sampled_callchains')
        ModelGenerator(experiment).model_all()

    def test_load_only_metrics_without_device_attributes(self):
        path = Path('data/nsight/sampled_callchains')
        ncu_files = list(path.glob('*/[!.]*.ncu-rep'))
        experiment = NsightFileReader().read_ncu_files(path, ncu_files)
        self.assertNotIn(Metric('time'), experiment.metrics)
        self.assertNotIn(Metric('device__attribute_max_threads_per_block'), experiment.metrics)
        self.assertSetEqual(set(experiment.metrics),
                            {Metric(name) for name in
                             ['gpu__time_duration.sum', 'launch__block_dim_x', 'launch__block_dim_y',
                              'launch__block_dim_z', 'launch__block_size', 'launch__context_id', 'launch__device_id',
                              'launch__function_pcs', 'launch__grid_dim_x',
                              'launch__grid_dim_y', 'launch__grid_dim_z', 'launch__grid_size',
                              'launch__occupancy_limit_blocks', 'launch__occupancy_limit_registers',
                              'launch__occupancy_limit_shared_mem', 'launch__occupancy_limit_warps',
                              'launch__occupancy_per_block_size', 'launch__occupancy_per_register_count',
                              'launch__occupancy_per_shared_mem_size',
                              'launch__registers_per_thread',
                              'launch__registers_per_thread_allocated',
                              'launch__shared_mem_config_size', 'launch__shared_mem_per_block',
                              'launch__shared_mem_per_block_allocated',
                              'launch__shared_mem_per_block_driver',
                              'launch__shared_mem_per_block_dynamic',
                              'launch__shared_mem_per_block_static', 'launch__stream_id',
                              'launch__thread_count', 'launch__waves_per_multiprocessor',
                              'profiler__perfworks_session_reuse', 'profiler__replayer_bytes_mem_accessible.avg',
                              'profiler__replayer_bytes_mem_accessible.max',
                              'profiler__replayer_bytes_mem_accessible.min',
                              'profiler__replayer_bytes_mem_accessible.sum',
                              'profiler__replayer_bytes_mem_backed_up.avg',
                              'profiler__replayer_bytes_mem_backed_up.max',
                              'profiler__replayer_bytes_mem_backed_up.min',
                              'profiler__replayer_bytes_mem_backed_up.sum', 'profiler__replayer_passes',
                              'profiler__replayer_passes_type_warmup',
                              'sm__inst_executed.avg', 'sm__inst_executed.max', 'sm__inst_executed.min',
                              'sm__inst_executed.sum', 'sm__maximum_warps_avg_per_active_cycle',
                              'sm__maximum_warps_per_active_cycle_pct', 'sm__sass_inst_executed_op_ld.avg',
                              'sm__sass_inst_executed_op_ld.max', 'sm__sass_inst_executed_op_ld.min',
                              'sm__sass_inst_executed_op_ld.sum', 'sm__sass_inst_executed_op_st.avg',
                              'sm__sass_inst_executed_op_st.max', 'sm__sass_inst_executed_op_st.min',
                              'sm__sass_inst_executed_op_st.sum', 'smsp__maximum_warps_avg_per_active_cycle']
                             })

    def test_load_concurrent_example(self):
        experiment = NsightFileReader().read_experiment('data/nsight/concurrent_example')

        print(experiment.callpaths)


class TestNsysDbReader(unittest.TestCase):
    def test_concurrent_kernel_execution(self):
        report = NsysReport('data/nsight/nsys_db/con_kernel.sqlite')
        mem_copies = report.get_mem_copies()
        self.assertEqual(11, len([m[-2] for m in mem_copies if m[-2] is not None]))

        self.assertEqual(4, len([correlation_id
                                 for correlation_id, callpath, name, duration, bytes, copyKind, blocking, gpu_duration,
                                     overlap_duration
                                 in mem_copies
                                 if '6' in callpath and bytes is None and gpu_duration is None
                                 and overlap_duration is not None]))
        for correlation_id, callpath, name, duration, bytes, copyKind, blocking, gpu_duration, overlap_duration in mem_copies:
            if overlap_duration is not None:
                self.assertGreaterEqual(overlap_duration, 0)
            if name is None and overlap_duration is not None:
                self.assertGreaterEqual(gpu_duration, overlap_duration)
        print(mem_copies)
        kernels = report.get_kernel_runtimes()
        self.assertEqual(7,
                         len([(correlation_id, callpath, name, duration, durationGPU, other_duration, overlap_duration)
                              for correlation_id, callpath, name, duration, durationGPU, other_duration,
                                  overlap_duration
                              in kernels
                              if durationGPU is not None and name is not None]))
        for correlation_id, callpath, name, duration, durationGPU, other_duration, overlap_duration in kernels:
            self.assertGreaterEqual(overlap_duration, 0)
            self.assertGreaterEqual(durationGPU, overlap_duration)
        print(kernels)
        sync = report.get_synchronization()
        self.assertEqual(6, len([(correlation_id, callpath, name, duration, syncType, other_duration, overlap_duration)
                                 for
                                 correlation_id, callpath, name, duration, syncType, other_duration, overlap_duration
                                 in sync
                                 if name is None]))
        for correlation_id, callpath, name, duration, syncType, other_duration, overlap_duration in sync:
            if overlap_duration is not None:
                self.assertGreaterEqual(overlap_duration, 0)
            if name is None and overlap_duration is not None:
                self.assertGreaterEqual(other_duration, overlap_duration)
        for correlation_id, grp in groupby(sorted(sync, key=lambda x: x[0]), lambda x: x[0]):
            # checks if overlap_duration greater or equal than max(all(overlap_durations of the same correlationId))
            grp = list(grp)
            agg = [(correlation_id, name, other_duration, overlap_duration)
                   for correlation_id, _, name, _, _, other_duration, overlap_duration in grp if name is None]
            self.assertEqual(1, len(agg))
            max_overlap_duration = max(grp, key=lambda x: x[-1])
            if max_overlap_duration[-1] is None:
                self.assertEqual(agg[0][-1], max_overlap_duration[-1])
            else:
                self.assertLessEqual(max_overlap_duration[-1], agg[0][-1])
        print(sync)

        mem_alloc = report.get_mem_alloc_free()
        self.assertEqual(4, len([(correlation_id, callpath, name, duration, blocking, host, overlap_duration)
                                 for
                                 correlation_id, callpath, name, duration, blocking, host, overlap_duration
                                 in mem_alloc
                                 if name is None]))

        print(mem_alloc)

        mem_set = report.get_mem_sets()
        self.assertEqual(1, len([(correlation_id, callpath, name, duration, bytes, blocking, other_duration,
                                  overlap_duration)
                                 for
                                 correlation_id, callpath, name, duration, bytes, blocking, other_duration,
                                 overlap_duration
                                 in mem_set
                                 if name is None]))
        self.assertEqual(mem_set[0][-2], mem_set[0][-1])
        print(mem_set)


if __name__ == '__main__':
    unittest.main()
