# kaitai-struct-compiler --python-package "." --target python ncu-report.ksy
meta:
  id: nsight_cuprof_report
  application: Nsight Compute
  endian: le
  ks-opaque-types: true
seq:
  - id: magic
    contents: [ "NVR", 0 ]
  - id: sizeof_header
    type: u4
  - id: header
    size: sizeof_header
    type: file_header
  - id: blocks
    type: block
    repeat: eos
types:
  block:
    seq:
      - id: sizeof_header
        type: u4
      - id: header
        size: sizeof_header
        type: block_header
      - id: payloads
        type: payload_entries(header.as<i_block_header>.num_sources, header.as<i_block_header>.num_results)
        size: header.as<i_block_header>.payload_size

  i_block_header:
    instances:
      num_sources:
        value: 0
      num_results:
        value: 0
      payload_size:
        value: 0


  payload_entries:
    params:
      - id: num_sources
        type: u4
      - id: num_results
        type: u4
    seq:
      - id: sources
        type: payload_source
        repeat: expr
        repeat-expr: num_sources
      - id: results
        type: payload_result
        repeat: expr
        repeat-expr: num_results
  payload_source:
    seq:
      - id: sizeof_payload
        type: u4
      - id: entry
        size: sizeof_payload
        type: profile_source
  payload_result:
    seq:
      - id: sizeof_payload
        type: u4
      - id: entry
        size: sizeof_payload
        type: profile_result
