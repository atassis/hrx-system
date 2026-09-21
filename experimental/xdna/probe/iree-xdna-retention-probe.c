// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Research probe: can the firmware's one-entry PDI cache retain an entry's
// array state in a PDI-less libamdf context? Invocation 0 is split into its
// establishing prefix and the control body it shares with invocation 1. The
// prefix is re-encoded as a partial-PDI CDO placed in context-private backing,
// as XRT places full-ELF PDIs, and each synthesized command is
// [LOADPDI(id, address) + optional prefix + control body].

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "amdf/amdf.h"
#include "amdf/xdna.h"
#include "experimental/xdna/amdf_status.h"
#include "experimental/xdna/executable.h"
#include "iree/base/api.h"
#include "iree/base/byte_sequence.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/npu2.h"
#include "iree/io/file_contents.h"

IREE_FLAG(string, image, "", "Complete canonical .xdna executable file.");
IREE_FLAG(string, entry, "", "Exported entry; empty selects ordinal zero.");
IREE_FLAG(int32_t, columns, 0, "Logical context column count of the image.");
IREE_FLAG(int32_t, device, 0, "XDNA endpoint ordinal.");
IREE_FLAG(string, target_id, "amd.xdna.strix.17f0_10",
          "Image target used by --host_only; devices report their own.");
IREE_FLAG(string, kernel, "none",
          "Output oracle: mul_i32 (out = lhs * rhs), copy (out = in), none.");
IREE_FLAG(string, modes, "establish,continuation",
          "Comma-separated modes run in order on one context: establish, "
          "continuation, nop-establish, nop-alternate, pdi-establish, pdi, "
          "pdi-alternate, pdi-alias, pdi-rekey.");
IREE_FLAG(int32_t, calls, 50, "Measured calls per mode.");
IREE_FLAG(int32_t, pdi_id_base, 0,
          "First of four PDI ids (establish A, establish B, NOP A, NOP B); "
          "zero derives a base from the process id.");
IREE_FLAG(bool, pdi_size_field, false,
          "Write the PDI byte length into LOADPDI; aiecc full-ELF streams "
          "leave it zero and XRT patches only the address.");
IREE_FLAG(int32_t, timeout_ms, 1500, "Per-call wait before reporting HANG.");
IREE_FLAG(int32_t, drain_ms, 30000,
          "Wait for a hung call to retire (driver TDR) before teardown.");
IREE_FLAG(int32_t, gap_ms, 0, "Untimed idle sleep before every call.");
IREE_FLAG(string, interpose_command, "",
          "Shell command run before call --interpose_call of every mode.");
IREE_FLAG(int32_t, interpose_call, -1, "Call index preceded by interposition.");
IREE_FLAG(bool, trace_calls, false, "Print one line per call.");
IREE_FLAG(string, dump_dir, "",
          "Directory receiving invocations, CDO, PDIs and commands.");
IREE_FLAG(bool, host_only, false,
          "Build and dump every stream with synthetic addresses; no device.");

enum {
  PROBE_TXN_HEADER_SIZE = 16,
  PROBE_TXN_OP_WRITE = 0,
  PROBE_TXN_OP_BLOCKWRITE = 1,
  PROBE_TXN_OP_MASKWRITE = 3,
  PROBE_TXN_OP_MASKPOLL = 4,
  PROBE_TXN_OP_NOOP = 5,
  PROBE_TXN_OP_MASKPOLL_BUSY = 7,
  PROBE_TXN_OP_LOADPDI = 8,
  PROBE_TXN_OP_CUSTOM_FIRST = 0x80,
  PROBE_TXN_LOADPDI_SIZE = 16,
};

// The vocabulary bootgen leaves in the PDIs XRT loads: 32-bit WRITE and
// MASK_WRITE, DMA_WRITE with a split 64-bit address, and NOP padding.
enum {
  PROBE_CDO_HEADER_SIZE = 20,
  PROBE_CDO_MASK_WRITE = 0x102,
  PROBE_CDO_WRITE = 0x103,
  PROBE_CDO_DMA_WRITE = 0x105,
  PROBE_CDO_NOP = 0x111,
  PROBE_CDO_LONG_LENGTH = 255,
};

// Unsigned partial-PDI framing of libamdf/src/xdna/bootstrap.c.
enum {
  PROBE_PDI_TABLE_OFFSET = 16,
  PROBE_PDI_TABLE_LENGTH = 128,
  PROBE_PDI_IMAGE_OFFSET = 144,
  PROBE_PDI_IMAGE_LENGTH = 64,
  PROBE_PDI_PARTITION_OFFSET = 208,
  PROBE_PDI_PARTITION_LENGTH = 128,
  PROBE_PDI_CDO_OFFSET = 336,
};

typedef enum probe_region_e {
  PROBE_REGION_PDI_ESTABLISH_A = 0,
  PROBE_REGION_PDI_ESTABLISH_B,
  PROBE_REGION_PDI_NOP_A,
  PROBE_REGION_PDI_NOP_B,
  PROBE_REGION_COMMAND_FIRST,
} probe_region_t;

typedef enum probe_command_e {
  PROBE_COMMAND_PDI_A = 0,
  PROBE_COMMAND_PDI_B,
  PROBE_COMMAND_PDI_ALIAS,
  PROBE_COMMAND_PDI_REKEY,
  PROBE_COMMAND_NOP_A,
  PROBE_COMMAND_NOP_B,
  PROBE_COMMAND_PDI_ESTABLISH,
  PROBE_SYNTHESIZED_COMMAND_COUNT,
  PROBE_COMMAND_ESTABLISH = PROBE_SYNTHESIZED_COMMAND_COUNT,
  PROBE_COMMAND_CONTINUATION,
  PROBE_COMMAND_COUNT,
} probe_command_t;

static const char* const probe_command_names[PROBE_COMMAND_COUNT] = {
    "cmd_pdi_a",         "cmd_pdi_b", "cmd_pdi_alias",
    "cmd_pdi_rekey",     "cmd_nop_a", "cmd_nop_b",
    "cmd_pdi_establish", "inv0",      "inv1",
};

typedef struct probe_mode_t {
  const char* name;
  // Submits invocation 0 once, unmeasured, before the calls.
  bool prime;
  uint8_t sequence_length;
  probe_command_t sequence[2];
} probe_mode_t;

static const probe_mode_t probe_modes[] = {
    {"establish", false, 1, {PROBE_COMMAND_ESTABLISH}},
    {"continuation", true, 1, {PROBE_COMMAND_CONTINUATION}},
    {"nop-establish", false, 1, {PROBE_COMMAND_NOP_A}},
    {"nop-alternate", false, 2, {PROBE_COMMAND_NOP_A, PROBE_COMMAND_NOP_B}},
    {"pdi-establish", false, 1, {PROBE_COMMAND_PDI_ESTABLISH}},
    {"pdi", false, 1, {PROBE_COMMAND_PDI_A}},
    {"pdi-alternate", false, 2, {PROBE_COMMAND_PDI_A, PROBE_COMMAND_PDI_B}},
    // Same id at another address, then another id at the same address: which
    // operand keys the firmware's cache.
    {"pdi-alias", false, 2, {PROBE_COMMAND_PDI_A, PROBE_COMMAND_PDI_ALIAS}},
    {"pdi-rekey", false, 2, {PROBE_COMMAND_PDI_A, PROBE_COMMAND_PDI_REKEY}},
};

//===----------------------------------------------------------------------===//
// Native transaction and CDO encoding
//===----------------------------------------------------------------------===//

typedef struct probe_bytes_t {
  iree_allocator_t allocator;
  uint8_t* data;
  iree_host_size_t length;
  iree_host_size_t capacity;
} probe_bytes_t;

static iree_status_t probe_bytes_grow(probe_bytes_t* bytes,
                                      iree_host_size_t count,
                                      uint8_t** out_tail) {
  if (bytes->length + count > bytes->capacity) {
    iree_host_size_t capacity = iree_max(4096, bytes->capacity);
    while (capacity < bytes->length + count) {
      capacity *= 2;
    }
    IREE_RETURN_IF_ERROR(iree_allocator_realloc(bytes->allocator, capacity,
                                                (void**)&bytes->data));
    bytes->capacity = capacity;
  }
  *out_tail = bytes->data + bytes->length;
  memset(*out_tail, 0, count);
  bytes->length += count;
  return iree_ok_status();
}

static iree_status_t probe_bytes_append_u32(probe_bytes_t* bytes,
                                            uint32_t value) {
  uint8_t* tail = NULL;
  IREE_RETURN_IF_ERROR(probe_bytes_grow(bytes, 4, &tail));
  iree_unaligned_store_le_u32(tail, value);
  return iree_ok_status();
}

static void probe_bytes_reset(probe_bytes_t* bytes) {
  iree_allocator_free(bytes->allocator, bytes->data);
  bytes->data = NULL;
  bytes->length = bytes->capacity = 0;
}

static iree_const_byte_span_t probe_bytes_span(const probe_bytes_t* bytes) {
  return iree_make_const_byte_span(bytes->data, bytes->length);
}

static iree_status_t probe_txn_op_size(iree_const_byte_span_t stream,
                                       iree_host_size_t offset,
                                       uint32_t* out_size) {
  if (offset + 4 > stream.data_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "transaction op at %" PRIhsz " is truncated",
                            offset);
  }
  const uint8_t* op = stream.data + offset;
  uint32_t size = 0;
  switch (op[0]) {
    case PROBE_TXN_OP_WRITE:
      size = 24;
      break;
    case PROBE_TXN_OP_MASKWRITE:
    case PROBE_TXN_OP_MASKPOLL:
    case PROBE_TXN_OP_MASKPOLL_BUSY:
      size = 28;
      break;
    case PROBE_TXN_OP_NOOP:
      size = 4;
      break;
    case PROBE_TXN_OP_LOADPDI:
      size = PROBE_TXN_LOADPDI_SIZE;
      break;
    case PROBE_TXN_OP_BLOCKWRITE:
      if (offset + 16 > stream.data_length) {
        break;
      }
      size = iree_unaligned_load_le_u32(op + 12);
      break;
    default:
      if (op[0] < PROBE_TXN_OP_CUSTOM_FIRST) {
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unknown transaction opcode 0x%x at %" PRIhsz,
                                op[0], offset);
      }
      if (offset + 8 > stream.data_length) {
        break;
      }
      size = iree_unaligned_load_le_u32(op + 4);
      break;
  }
  if (size < 4 || size % 4 != 0 || size > stream.data_length - offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "transaction op 0x%x at %" PRIhsz
                            " has invalid size %u",
                            op[0], offset, size);
  }
  *out_size = size;
  return iree_ok_status();
}

static iree_status_t probe_txn_walk(iree_const_byte_span_t stream,
                                    iree_host_size_t offset, uint32_t op_count,
                                    iree_host_size_t* out_end) {
  for (uint32_t i = 0; i < op_count; ++i) {
    uint32_t size = 0;
    IREE_RETURN_IF_ERROR(probe_txn_op_size(stream, offset, &size));
    offset += size;
  }
  *out_end = offset;
  return iree_ok_status();
}

static iree_status_t probe_txn_header(iree_const_byte_span_t stream,
                                      const char* name,
                                      uint32_t* out_op_count) {
  if (stream.data_length < PROBE_TXN_HEADER_SIZE || stream.data[0] != 0 ||
      stream.data[1] != 1 ||
      iree_unaligned_load_le_u32(stream.data + 12) != stream.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s is not one complete version 0.1 transaction",
                            name);
  }
  *out_op_count = iree_unaligned_load_le_u32(stream.data + 8);
  return iree_ok_status();
}

typedef struct probe_split_t {
  // Version, device generation and geometry bytes shared by both invocations.
  uint8_t header[PROBE_TXN_HEADER_SIZE];
  iree_const_byte_span_t establish;
  uint32_t establish_op_count;
  iree_const_byte_span_t control;
  uint32_t control_op_count;
} probe_split_t;

// Invocation 0 must be exactly [header, establishing ops, invocation 1 body].
static iree_status_t probe_split_invocations(iree_const_byte_span_t inv0,
                                             iree_const_byte_span_t inv1,
                                             probe_split_t* out_split) {
  uint32_t op_count0 = 0, op_count1 = 0;
  IREE_RETURN_IF_ERROR(probe_txn_header(inv0, "invocation 0", &op_count0));
  IREE_RETURN_IF_ERROR(probe_txn_header(inv1, "invocation 1", &op_count1));
  if (memcmp(inv0.data + 2, inv1.data + 2, 6) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocations describe different arrays");
  }
  const iree_const_byte_span_t control =
      iree_make_const_byte_span(inv1.data + PROBE_TXN_HEADER_SIZE,
                                inv1.data_length - PROBE_TXN_HEADER_SIZE);
  if (op_count1 >= op_count0 ||
      inv0.data_length < PROBE_TXN_HEADER_SIZE + control.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation 1 is not a strict suffix");
  }
  const iree_host_size_t control_offset =
      inv0.data_length - control.data_length;
  if (memcmp(inv0.data + control_offset, control.data, control.data_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation 0 does not end with invocation 1");
  }
  iree_host_size_t end = 0;
  IREE_RETURN_IF_ERROR(
      probe_txn_walk(inv0, PROBE_TXN_HEADER_SIZE, op_count0 - op_count1, &end));
  if (end != control_offset) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "establishing ops end at %" PRIhsz
                            ", control begins at %" PRIhsz,
                            end, control_offset);
  }
  IREE_RETURN_IF_ERROR(
      probe_txn_walk(inv1, PROBE_TXN_HEADER_SIZE, op_count1, &end));
  if (end != inv1.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation 1 ops do not fill its range");
  }
  memcpy(out_split->header, inv1.data, PROBE_TXN_HEADER_SIZE);
  out_split->establish =
      iree_make_const_byte_span(inv0.data + PROBE_TXN_HEADER_SIZE,
                                control_offset - PROBE_TXN_HEADER_SIZE);
  out_split->establish_op_count = op_count0 - op_count1;
  out_split->control = control;
  out_split->control_op_count = op_count1;
  return iree_ok_status();
}

// Pads with one NOP so a DMA_WRITE payload starts 16-byte aligned relative to
// the CDO, as cdo_driver.c does before bootgen.
static iree_status_t probe_cdo_align_payload(probe_bytes_t* cdo,
                                             uint32_t prefix_words) {
  const uint32_t misalignment =
      (uint32_t)((cdo->length + prefix_words * 4) % 16);
  if (misalignment == 0) {
    return iree_ok_status();
  }
  const uint32_t pad_words = (16 - misalignment) / 4;
  IREE_RETURN_IF_ERROR(
      probe_bytes_append_u32(cdo, ((pad_words - 1) << 16) | PROBE_CDO_NOP));
  for (uint32_t i = 1; i < pad_words; ++i) {
    IREE_RETURN_IF_ERROR(probe_bytes_append_u32(cdo, 0));
  }
  return iree_ok_status();
}

static iree_status_t probe_cdo_append_op(probe_bytes_t* cdo,
                                         iree_const_byte_span_t ops,
                                         iree_host_size_t offset,
                                         uint32_t size) {
  const uint8_t* op = ops.data + offset;
  switch (op[0]) {
    case PROBE_TXN_OP_WRITE:
    case PROBE_TXN_OP_MASKWRITE: {
      const uint64_t address = iree_unaligned_load_le_u64(op + 8);
      if (address > UINT32_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "register 0x%" PRIx64 " exceeds 32 bits",
                                address);
      }
      const uint32_t value = iree_unaligned_load_le_u32(op + 16);
      if (op[0] == PROBE_TXN_OP_WRITE) {
        IREE_RETURN_IF_ERROR(
            probe_bytes_append_u32(cdo, (2u << 16) | PROBE_CDO_WRITE));
        IREE_RETURN_IF_ERROR(probe_bytes_append_u32(cdo, (uint32_t)address));
        return probe_bytes_append_u32(cdo, value);
      }
      IREE_RETURN_IF_ERROR(
          probe_bytes_append_u32(cdo, (3u << 16) | PROBE_CDO_MASK_WRITE));
      IREE_RETURN_IF_ERROR(probe_bytes_append_u32(cdo, (uint32_t)address));
      IREE_RETURN_IF_ERROR(
          probe_bytes_append_u32(cdo, iree_unaligned_load_le_u32(op + 20)));
      return probe_bytes_append_u32(cdo, value);
    }
    case PROBE_TXN_OP_BLOCKWRITE: {
      const uint32_t address = iree_unaligned_load_le_u32(op + 8);
      const uint32_t word_count = (size - 16) / 4;
      const uint32_t payload_words = word_count + 2;
      const bool long_form = payload_words >= PROBE_CDO_LONG_LENGTH;
      IREE_RETURN_IF_ERROR(probe_cdo_align_payload(cdo, long_form ? 4 : 3));
      if (long_form) {
        IREE_RETURN_IF_ERROR(probe_bytes_append_u32(
            cdo, (PROBE_CDO_LONG_LENGTH << 16) | PROBE_CDO_DMA_WRITE));
        IREE_RETURN_IF_ERROR(probe_bytes_append_u32(cdo, payload_words));
      } else {
        IREE_RETURN_IF_ERROR(probe_bytes_append_u32(
            cdo, (payload_words << 16) | PROBE_CDO_DMA_WRITE));
      }
      IREE_RETURN_IF_ERROR(probe_bytes_append_u32(cdo, 0));
      IREE_RETURN_IF_ERROR(probe_bytes_append_u32(cdo, address));
      uint8_t* payload = NULL;
      IREE_RETURN_IF_ERROR(probe_bytes_grow(cdo, word_count * 4, &payload));
      memcpy(payload, op + 16, word_count * 4);
      return iree_ok_status();
    }
    case PROBE_TXN_OP_NOOP:
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "establishing opcode 0x%x at %" PRIhsz
                              " has no CDO form",
                              op[0], offset);
  }
}

// Builds one CDO object: |nop_count| NOPs, then |ops| re-encoded.
static iree_status_t probe_cdo_build(iree_const_byte_span_t ops,
                                     uint32_t op_count, uint32_t nop_count,
                                     probe_bytes_t* cdo) {
  uint8_t* header = NULL;
  IREE_RETURN_IF_ERROR(probe_bytes_grow(cdo, PROBE_CDO_HEADER_SIZE, &header));
  for (uint32_t i = 0; i < nop_count; ++i) {
    IREE_RETURN_IF_ERROR(probe_bytes_append_u32(cdo, PROBE_CDO_NOP));
  }
  iree_host_size_t offset = 0;
  for (uint32_t i = 0; i < op_count; ++i) {
    uint32_t size = 0;
    IREE_RETURN_IF_ERROR(probe_txn_op_size(ops, offset, &size));
    IREE_RETURN_IF_ERROR(probe_cdo_append_op(cdo, ops, offset, size));
    offset += size;
  }
  header = cdo->data;
  const uint32_t words[4] = {
      4,
      UINT32_C(0x004F4443),
      UINT32_C(0x200),
      (uint32_t)((cdo->length - PROBE_CDO_HEADER_SIZE) / 4),
  };
  uint32_t sum = 0;
  for (int i = 0; i < 4; ++i) {
    iree_unaligned_store_le_u32(header + 4 * i, words[i]);
    sum += words[i];
  }
  iree_unaligned_store_le_u32(header + 16, ~sum);
  return iree_ok_status();
}

static void probe_pdi_checksum(uint8_t* bytes, uint32_t byte_length) {
  uint32_t sum = 0;
  for (uint32_t offset = 0; offset < byte_length - 4; offset += 4) {
    sum += iree_unaligned_load_le_u32(bytes + offset);
  }
  iree_unaligned_store_le_u32(bytes + byte_length - 4, ~sum);
}

static iree_status_t probe_pdi_build(iree_const_byte_span_t cdo,
                                     probe_bytes_t* pdi) {
  const uint32_t storage_length =
      (uint32_t)iree_host_align(cdo.data_length, 16);
  uint8_t* bytes = NULL;
  IREE_RETURN_IF_ERROR(
      probe_bytes_grow(pdi, PROBE_PDI_CDO_OFFSET + storage_length, &bytes));
  iree_unaligned_store_le_u32(bytes + 0, 0xDD);
  iree_unaligned_store_le_u32(bytes + 4, UINT32_C(0x11223344));
  iree_unaligned_store_le_u32(bytes + 8, UINT32_C(0x55667788));
  iree_unaligned_store_le_u32(bytes + 12, UINT32_C(0x99AABBCC));

  uint8_t* table = bytes + PROBE_PDI_TABLE_OFFSET;
  iree_unaligned_store_le_u32(table + 0, UINT32_C(0x00040000));
  iree_unaligned_store_le_u32(table + 4, 1);
  iree_unaligned_store_le_u32(table + 8, PROBE_PDI_IMAGE_OFFSET / 4);
  iree_unaligned_store_le_u32(table + 12, 1);
  iree_unaligned_store_le_u32(table + 16, PROBE_PDI_PARTITION_OFFSET / 4);
  iree_unaligned_store_le_u32(table + 24, UINT32_C(0x14CA8093));
  iree_unaligned_store_le_u32(table + 40, UINT32_C(0x50504449));
  iree_unaligned_store_le_u32(table + 44,
                              (PROBE_PDI_TABLE_LENGTH / 4) |
                                  ((PROBE_PDI_IMAGE_LENGTH / 4) << 8) |
                                  ((PROBE_PDI_PARTITION_LENGTH / 4) << 16));
  iree_unaligned_store_le_u32(
      table + 48, (PROBE_PDI_IMAGE_LENGTH + PROBE_PDI_PARTITION_LENGTH) / 4);
  iree_unaligned_store_le_u32(table + 68, 1);
  probe_pdi_checksum(table, PROBE_PDI_TABLE_LENGTH);

  uint8_t* image = bytes + PROBE_PDI_IMAGE_OFFSET;
  iree_unaligned_store_le_u32(image + 0, PROBE_PDI_PARTITION_OFFSET / 4);
  iree_unaligned_store_le_u32(image + 4, 1);
  memcpy(image + 16, "aie_image", 9);
  iree_unaligned_store_le_u32(image + 32, UINT32_C(0x1C000000));
  // Bootgen writes all-ones here; libamdf's bootstrap writes zero.
  iree_unaligned_store_le_u32(image + 56, UINT32_MAX);
  probe_pdi_checksum(image, PROBE_PDI_IMAGE_LENGTH);

  uint8_t* partition = bytes + PROBE_PDI_PARTITION_OFFSET;
  iree_unaligned_store_le_u32(partition + 0, storage_length / 4);
  iree_unaligned_store_le_u32(partition + 4, (uint32_t)cdo.data_length / 4);
  iree_unaligned_store_le_u32(partition + 8, storage_length / 4);
  iree_unaligned_store_le_u32(partition + 24, UINT32_MAX);
  iree_unaligned_store_le_u32(partition + 28, UINT32_MAX);
  iree_unaligned_store_le_u32(partition + 32, PROBE_PDI_CDO_OFFSET / 4);
  iree_unaligned_store_le_u32(partition + 36, UINT32_C(0x02000006));
  iree_unaligned_store_le_u32(partition + 40, 1);
  probe_pdi_checksum(partition, PROBE_PDI_PARTITION_LENGTH);

  memcpy(bytes + PROBE_PDI_CDO_OFFSET, cdo.data, cdo.data_length);
  return iree_ok_status();
}

// Operand layout of aie-rt's XAie_LoadPdiHdr and mlir-aie's
// txn_append_loadpdi: opcode, pad, u16 id, u32 size, u64 address.
static iree_status_t probe_command_build(const probe_split_t* split,
                                         bool establish, uint16_t pdi_id,
                                         uint32_t pdi_size_field,
                                         uint64_t pdi_address,
                                         probe_bytes_t* command) {
  const iree_host_size_t establish_length =
      establish ? split->establish.data_length : 0;
  const iree_host_size_t length = PROBE_TXN_HEADER_SIZE +
                                  PROBE_TXN_LOADPDI_SIZE + establish_length +
                                  split->control.data_length;
  uint8_t* bytes = NULL;
  IREE_RETURN_IF_ERROR(probe_bytes_grow(command, length, &bytes));
  memcpy(bytes, split->header, 8);
  iree_unaligned_store_le_u32(bytes + 8,
                              1 + (establish ? split->establish_op_count : 0) +
                                  split->control_op_count);
  iree_unaligned_store_le_u32(bytes + 12, (uint32_t)length);
  uint8_t* load = bytes + PROBE_TXN_HEADER_SIZE;
  load[0] = PROBE_TXN_OP_LOADPDI;
  iree_unaligned_store_le_u16(load + 2, pdi_id);
  iree_unaligned_store_le_u32(load + 4, pdi_size_field);
  iree_unaligned_store_le_u64(load + 8, pdi_address);
  uint8_t* body = load + PROBE_TXN_LOADPDI_SIZE;
  memcpy(body, split->establish.data, establish_length);
  memcpy(body + establish_length, split->control.data,
         split->control.data_length);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Native resources
//===----------------------------------------------------------------------===//

typedef struct probe_allocation_t {
  amdf_memory_t* memory;
  amdf_host_mapping_t* mapping;
  // Heap backing of a --host_only allocation.
  void* host_storage;
  iree_byte_span_t view;
  uint64_t address;
} probe_allocation_t;

typedef struct probe_binding_t {
  probe_allocation_t allocation;
  iree_hal_buffer_t* buffer;
  bool is_output;
} probe_binding_t;

typedef struct probe_t {
  iree_allocator_t host_allocator;
  const amdf_api_t* api;
  const amdf_xdna_api_t* xdna_api;
  amdf_instance_t* instance;
  amdf_memory_scope_t* system_scope;
  amdf_endpoint_t* endpoint;
  amdf_device_t* device;
  amdf_xdna_context_t* context;
  amdf_memory_scope_t* private_scope;
  uint32_t queue_family_ordinal;
  amdf_kernel_queue_t* queue;
  uint32_t instruction_alignment;
  // Next synthetic address handed out by --host_only allocations.
  uint64_t synthetic_address;
  iree_hal_amd_xdna_image_t* image;
  uint32_t entry_ordinal;
  iree_xdna_elf_entry_record_t entry;
  uint32_t storage_count;
  probe_allocation_t* storage_allocations;
  iree_hal_amd_xdna_executable_storage_t* storage;
  uint32_t binding_count;
  probe_binding_t* bindings;
  iree_hal_amd_xdna_executable_binding_t* resolved_bindings;
  // PDIs and synthesized commands, one instruction-aligned region each.
  probe_allocation_t scratch;
  iree_host_size_t region_length;
  uint16_t pdi_id_base;
  probe_split_t split;
  amdf_xdna_kernel_command_t commands[PROBE_COMMAND_COUNT];
} probe_t;

// The executable adapter only checks that a native handle is present.
static int probe_host_only_memory_sentinel;

static iree_status_t probe_allocation_create(probe_t* probe, bool command,
                                             uint64_t byte_length,
                                             uint64_t alignment,
                                             probe_allocation_t* out) {
  if (FLAG_host_only) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
        probe->host_allocator, (iree_host_size_t)byte_length, 4096, 0,
        &out->host_storage));
    memset(out->host_storage, 0, (iree_host_size_t)byte_length);
    out->memory = (amdf_memory_t*)&probe_host_only_memory_sentinel;
    out->view = iree_make_byte_span(out->host_storage, byte_length);
    out->address = iree_align_uint64(probe->synthetic_address, alignment);
    probe->synthetic_address =
        iree_align_uint64(out->address + byte_length, 0x10000);
    return iree_ok_status();
  }
  amdf_memory_scope_t* scope =
      command ? probe->private_scope : probe->system_scope;
  const amdf_memory_address_kind_t address_kind =
      command ? AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE
              : AMDF_MEMORY_ADDRESS_XDNA_DMA;
  const amdf_memory_device_access_t access = {
      .device = probe->device,
      .requirements =
          {
              .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                        (command ? AMDF_MEMORY_ACCESS_EXECUTE : 0),
              .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
              .address_kinds = UINT64_C(1) << address_kind,
          },
  };
  amdf_memory_profile_t profile = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
      .structure_size = sizeof(profile),
  };
  for (uint32_t ordinal = 0;; ++ordinal) {
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities),
    };
    const amdf_status_t status = probe->api->memory_scope_query_device_profile(
        scope, ordinal, 1, &access, &profile, &capabilities);
    if (amdf_status_code(status) == AMDF_STATUS_CODE_OUT_OF_RANGE) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "no host-mappable XDNA allocation profile");
    }
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        status, "memory_scope_query_device_profile"));
    const amdf_memory_profile_roles_t roles =
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    if ((profile.roles & roles) == roles &&
        (profile.supported_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE) != 0) {
      break;
    }
  }
  const uint64_t granularity = profile.allocation.byte_length_granularity;
  uint64_t requested_alignment =
      iree_max(alignment, profile.allocation.minimum_alignment);
  if (profile.allocation.maximum_alignment != 0) {
    requested_alignment =
        iree_min(requested_alignment, profile.allocation.maximum_alignment);
  }
  const amdf_memory_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create_info),
      .memory_profile_ordinal = profile.ordinal,
      .access_count = 1,
      .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .byte_length = iree_align_uint64(byte_length, granularity),
      .minimum_alignment = requested_alignment,
      .accesses = &access,
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      probe->api->memory_create(scope, &create_info, &out->memory),
      "memory_create"));
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      probe->api->memory_query_address(out->memory, 0, address_kind,
                                       &out->address),
      "memory_query_address"));
  // amdxdna aligns heap buffers to 32 KiB (npu4_family.h dev_mem_buf_shift),
  // beyond what the private profile advertises.
  if (out->address & (alignment - 1)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "address 0x%" PRIx64 " is not %" PRIu64
                            "-byte aligned",
                            out->address, alignment);
  }
  const amdf_memory_map_info_t map_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
      .structure_size = sizeof(map_info),
      .byte_length = byte_length,
      .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      probe->api->memory_map(out->memory, &map_info, &out->mapping),
      "memory_map"));
  amdf_host_mapping_info_t mapping_info = {
      .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
      .structure_size = sizeof(mapping_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      probe->api->host_mapping_query_info(out->mapping, &mapping_info),
      "host_mapping_query_info"));
  out->view = iree_make_byte_span(mapping_info.pointer, byte_length);
  return iree_ok_status();
}

static iree_status_t probe_allocation_sync(probe_t* probe,
                                           probe_allocation_t* allocation,
                                           amdf_host_cache_operation_t op) {
  if (FLAG_host_only) {
    return iree_ok_status();
  }
  return IREE_HAL_AMD_STATUS_FROM_AMDF(
      probe->api->host_mapping_cache_control(allocation->mapping, op, 0,
                                             allocation->view.data_length),
      "host_mapping_cache_control");
}

static iree_status_t probe_allocation_release(probe_t* probe,
                                              probe_allocation_t* allocation) {
  if (allocation->host_storage) {
    iree_allocator_free_aligned(probe->host_allocator,
                                allocation->host_storage);
    allocation->host_storage = NULL;
    allocation->memory = NULL;
    return iree_ok_status();
  }
  if (allocation->mapping) {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        probe->api->host_mapping_destroy(allocation->mapping),
        "host_mapping_destroy"));
    allocation->mapping = NULL;
  }
  if (allocation->memory) {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        probe->api->memory_destroy(allocation->memory), "memory_destroy"));
    allocation->memory = NULL;
  }
  return iree_ok_status();
}

static iree_status_t probe_open_device(
    probe_t* probe, iree_hal_amd_xdna_aie2p_target_t* target) {
  if (FLAG_host_only) {
    probe->instruction_alignment = 32 * 1024;
    probe->synthetic_address = UINT64_C(0x10000000);
    return iree_hal_amd_xdna_aie2p_npu2_target_initialize(
        iree_make_cstring_view(FLAG_target_id), (uint16_t)FLAG_columns, target);
  }
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST,
                     &probe->api),
      "query_api"));
  const void* extension = NULL;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      probe->api->query_extension(
          AMDF_EXTENSION_XDNA, AMDF_XDNA_EXTENSION_VERSION_1,
          AMDF_XDNA_EXTENSION_VERSION_LATEST, &extension),
      "query_extension(XDNA)"));
  probe->xdna_api = extension;
  const amdf_instance_create_info_t instance_info = {
      .type = AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .structure_size = sizeof(instance_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      probe->api->instance_create(&instance_info, &probe->instance),
      "instance_create"));

  amdf_memory_scope_t* scopes[8];
  uint32_t scope_count = 0;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      probe->api->instance_enumerate_memory_scopes(
          probe->instance, IREE_ARRAYSIZE(scopes), scopes, &scope_count),
      "instance_enumerate_memory_scopes"));
  for (uint32_t i = 0; i < scope_count && !probe->system_scope; ++i) {
    amdf_memory_scope_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
        .structure_size = sizeof(info),
    };
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        probe->api->memory_scope_query_info(scopes[i], &info),
        "memory_scope_query_info"));
    if (info.kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM) {
      probe->system_scope = scopes[i];
    }
  }
  if (!probe->system_scope) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE, "no system memory scope");
  }

  amdf_endpoint_summary_t summaries[16];
  uint32_t count = 0;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      probe->api->endpoint_enumerate(probe->instance, IREE_ARRAYSIZE(summaries),
                                     summaries, &count),
      "endpoint_enumerate"));
  uint32_t xdna_ordinal = 0;
  for (uint32_t i = 0; i < count && !probe->endpoint; ++i) {
    if (summaries[i].engine_kind != AMDF_ENGINE_KIND_XDNA ||
        xdna_ordinal++ != (uint32_t)FLAG_device) {
      continue;
    }
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        probe->api->endpoint_open(probe->instance, &summaries[i].id,
                                  &probe->endpoint),
        "endpoint_open"));
  }
  if (!probe->endpoint) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "XDNA endpoint %d is unavailable", FLAG_device);
  }

  amdf_xdna_endpoint_info_t endpoint_info = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO,
      .structure_size = sizeof(endpoint_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      probe->xdna_api->endpoint_query_info(probe->endpoint, &endpoint_info),
      "xdna.endpoint_query_info"));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
      iree_make_cstring_view(endpoint_info.target_id), (uint16_t)FLAG_columns,
      target));
  const amdf_xdna_device_create_info_t device_info_create = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_CREATE_INFO,
      .structure_size = sizeof(device_info_create),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      probe->xdna_api->device_create(probe->endpoint, &device_info_create,
                                     &probe->device),
      "xdna.device_create"));
  amdf_xdna_device_info_t device_info = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO,
      .structure_size = sizeof(device_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      probe->xdna_api->device_query_info(probe->device, &device_info),
      "xdna.device_query_info"));
  probe->instruction_alignment = device_info.instruction.address_alignment;
  target->instruction_alignment = device_info.instruction.address_alignment;

  amdf_endpoint_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
      .structure_size = sizeof(info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      probe->api->endpoint_query_info(probe->endpoint, &info),
      "endpoint_query_info"));
  probe->queue_family_ordinal = UINT32_MAX;
  for (uint32_t i = 0; i < info.queue_family_count; ++i) {
    amdf_queue_family_info_t family = {
        .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
        .structure_size = sizeof(family),
    };
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        probe->api->endpoint_query_queue_family_info(probe->endpoint, i,
                                                     &family),
        "endpoint_query_queue_family_info"));
    if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_XDNA &&
        (family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_KERNEL)) {
      probe->queue_family_ordinal = i;
      break;
    }
  }
  if (probe->queue_family_ordinal == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "endpoint has no XDNA kernel queue family");
  }

  const amdf_xdna_context_create_info_t context_info = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_CREATE_INFO,
      .structure_size = sizeof(context_info),
      .logical_column_count = (uint32_t)FLAG_columns,
      .physical_column_origin = AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY,
      .acceptable_scheduling_modes = AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED,
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      probe->xdna_api->context_create(probe->device, &context_info,
                                      &probe->context),
      "xdna.context_create"));
  uint32_t private_count = 0;
  return IREE_HAL_AMD_STATUS_FROM_AMDF(
      probe->xdna_api->context_enumerate_memory_scopes(
          probe->context, 1, &probe->private_scope, &private_count),
      "xdna.context_enumerate_memory_scopes");
}

static iree_status_t probe_prepare_bindings(probe_t* probe) {
  const iree_hal_amd_xdna_image_tables_t* tables =
      iree_hal_amd_xdna_image_tables(probe->image);
  probe->binding_count = probe->entry.binding_count;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      probe->host_allocator,
      probe->binding_count * (sizeof(probe_binding_t) +
                              sizeof(iree_hal_amd_xdna_executable_binding_t)),
      (void**)&probe->bindings));
  probe->resolved_bindings =
      (iree_hal_amd_xdna_executable_binding_t*)(probe->bindings +
                                                probe->binding_count);
  for (uint32_t i = 0; i < probe->binding_count; ++i) {
    const iree_xdna_elf_binding_record_t contract =
        iree_hal_amd_xdna_image_tables_binding(tables,
                                               probe->entry.first_binding + i);
    probe_binding_t* binding = &probe->bindings[i];
    binding->is_output =
        (contract.access & IREE_XDNA_ELF_BINDING_ACCESS_WRITE) != 0;
    const uint64_t length = iree_align_uint64(contract.minimum_byte_length, 64);
    IREE_RETURN_IF_ERROR(probe_allocation_create(
        probe, /*command=*/false, length,
        iree_max(contract.minimum_alignment, 64), &binding->allocation));
    iree_hal_memory_access_t access =
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE;
    if (!iree_host_size_has_alignment((uintptr_t)binding->allocation.view.data,
                                      IREE_HAL_HEAP_BUFFER_ALIGNMENT)) {
      access |= IREE_HAL_MEMORY_ACCESS_UNALIGNED;
    }
    IREE_RETURN_IF_ERROR(iree_hal_heap_buffer_wrap(
        iree_hal_buffer_placement_undefined(),
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
            IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
            IREE_HAL_MEMORY_TYPE_HOST_CACHED,
        access, IREE_HAL_BUFFER_USAGE_STORAGE, length, binding->allocation.view,
        iree_hal_buffer_release_callback_null(), probe->host_allocator,
        &binding->buffer));
    probe->resolved_bindings[i] = (iree_hal_amd_xdna_executable_binding_t){
        .buffer_ref = iree_hal_make_buffer_ref(binding->buffer, 0, length),
        .memory = binding->allocation.memory,
        .device_address = binding->allocation.address,
    };
  }
  return iree_ok_status();
}

static iree_status_t probe_prepare_storage(probe_t* probe) {
  const iree_hal_amd_xdna_image_tables_t* tables =
      iree_hal_amd_xdna_image_tables(probe->image);
  probe->storage_count = probe->entry.allocation_use_count;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      probe->host_allocator,
      probe->storage_count * (sizeof(probe_allocation_t) +
                              sizeof(iree_hal_amd_xdna_executable_storage_t)),
      (void**)&probe->storage_allocations));
  probe->storage =
      (iree_hal_amd_xdna_executable_storage_t*)(probe->storage_allocations +
                                                probe->storage_count);
  for (uint32_t i = 0; i < probe->storage_count; ++i) {
    const iree_xdna_elf_allocation_record_t requirement =
        iree_hal_amd_xdna_image_tables_allocation(
            tables, iree_hal_amd_xdna_image_tables_allocation_use(
                        tables, probe->entry.first_allocation_use + i));
    probe_allocation_t* allocation = &probe->storage_allocations[i];
    IREE_RETURN_IF_ERROR(probe_allocation_create(
        probe, requirement.domain == IREE_XDNA_ELF_ALLOCATION_DOMAIN_COMMAND,
        requirement.byte_length, requirement.alignment, allocation));
    probe->storage[i] = (iree_hal_amd_xdna_executable_storage_t){
        .mapping = allocation->view,
        .memory = allocation->memory,
        .device_address = allocation->address,
    };
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_executable_load(probe->image, probe->entry_ordinal,
                                        probe->storage_count, probe->storage));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_executable_bind(
      probe->image, probe->entry_ordinal, probe->storage_count, probe->storage,
      probe->binding_count, probe->resolved_bindings));
  for (uint32_t i = 0; i < probe->storage_count; ++i) {
    IREE_RETURN_IF_ERROR(
        probe_allocation_sync(probe, &probe->storage_allocations[i],
                              AMDF_HOST_CACHE_OPERATION_FLUSH));
  }
  return iree_ok_status();
}

static iree_const_byte_span_t probe_invocation_bytes(
    const probe_t* probe, uint32_t ordinal,
    amdf_xdna_kernel_command_t* out_command) {
  const iree_xdna_elf_invocation_record_t invocation =
      iree_hal_amd_xdna_image_tables_invocation(
          iree_hal_amd_xdna_image_tables(probe->image),
          probe->entry.first_invocation + ordinal);
  const iree_hal_amd_xdna_executable_storage_t* backing =
      &probe->storage[invocation.allocation_use];
  *out_command = (amdf_xdna_kernel_command_t){
      .memory = backing->memory,
      .access_ordinal = backing->access_ordinal,
      .byte_offset = backing->memory_byte_offset + invocation.byte_offset,
      .byte_length = invocation.byte_length,
  };
  return iree_make_const_byte_span(
      backing->mapping.data + invocation.byte_offset, invocation.byte_length);
}

static iree_status_t probe_write_dump(const char* name,
                                      iree_const_byte_span_t bytes,
                                      iree_allocator_t allocator) {
  if (FLAG_dump_dir[0] == 0) {
    return iree_ok_status();
  }
  char path[1024];
  snprintf(path, sizeof(path), "%s/%s", FLAG_dump_dir, name);
  return iree_io_file_contents_write(iree_make_cstring_view(path), bytes,
                                     allocator);
}

// Materializes the PDIs and synthesized commands in one scratch allocation.
static iree_status_t probe_prepare_commands(probe_t* probe) {
  const iree_xdna_elf_invocation_record_t inv0_record =
      iree_hal_amd_xdna_image_tables_invocation(
          iree_hal_amd_xdna_image_tables(probe->image),
          probe->entry.first_invocation);
  if (probe->entry.invocation_count < 2 || inv0_record.next_invocation == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry has no continuation invocation");
  }
  const iree_const_byte_span_t inv0 = probe_invocation_bytes(
      probe, 0, &probe->commands[PROBE_COMMAND_ESTABLISH]);
  const iree_const_byte_span_t inv1 =
      probe_invocation_bytes(probe, inv0_record.next_invocation,
                             &probe->commands[PROBE_COMMAND_CONTINUATION]);
  IREE_RETURN_IF_ERROR(probe_split_invocations(inv0, inv1, &probe->split));

  probe_bytes_t cdo = {.allocator = probe->host_allocator};
  probe_bytes_t nop_cdo = {.allocator = probe->host_allocator};
  probe_bytes_t pdi = {.allocator = probe->host_allocator};
  probe_bytes_t nop_pdi = {.allocator = probe->host_allocator};
  probe_bytes_t commands[PROBE_SYNTHESIZED_COMMAND_COUNT];
  for (int i = 0; i < PROBE_SYNTHESIZED_COMMAND_COUNT; ++i) {
    commands[i] = (probe_bytes_t){.allocator = probe->host_allocator};
  }
  iree_status_t status = probe_cdo_build(
      probe->split.establish, probe->split.establish_op_count, 0, &cdo);
  // Three NOPs reproduce the empty PDIs aiecc builds for its reset devices.
  if (iree_status_is_ok(status)) {
    status = probe_cdo_build(iree_const_byte_span_empty(), 0, 3, &nop_cdo);
  }
  if (iree_status_is_ok(status)) {
    status = probe_pdi_build(probe_bytes_span(&cdo), &pdi);
  }
  if (iree_status_is_ok(status)) {
    status = probe_pdi_build(probe_bytes_span(&nop_cdo), &nop_pdi);
  }

  iree_host_size_t largest = iree_max(pdi.length, nop_pdi.length);
  largest = iree_max(largest, PROBE_TXN_HEADER_SIZE + PROBE_TXN_LOADPDI_SIZE +
                                  inv0.data_length);
  probe->region_length = iree_host_align(largest, probe->instruction_alignment);
  const iree_host_size_t region_count =
      PROBE_REGION_COMMAND_FIRST + PROBE_SYNTHESIZED_COMMAND_COUNT;
  if (iree_status_is_ok(status)) {
    status = probe_allocation_create(
        probe, /*command=*/true, probe->region_length * region_count,
        probe->instruction_alignment, &probe->scratch);
  }
  uint64_t region_address[PROBE_REGION_COMMAND_FIRST];
  for (int i = 0; i < PROBE_REGION_COMMAND_FIRST; ++i) {
    region_address[i] = probe->scratch.address + i * probe->region_length;
  }
  const uint16_t id = probe->pdi_id_base;
  const uint32_t size_field = FLAG_pdi_size_field ? (uint32_t)pdi.length : 0;
  const uint32_t nop_size_field =
      FLAG_pdi_size_field ? (uint32_t)nop_pdi.length : 0;
  const struct {
    bool establish;
    uint16_t id;
    uint32_t size_field;
    uint64_t address;
  } recipes[PROBE_SYNTHESIZED_COMMAND_COUNT] = {
      [PROBE_COMMAND_PDI_A] = {false, id, size_field,
                               region_address[PROBE_REGION_PDI_ESTABLISH_A]},
      [PROBE_COMMAND_PDI_B] = {false, (uint16_t)(id + 1), size_field,
                               region_address[PROBE_REGION_PDI_ESTABLISH_B]},
      [PROBE_COMMAND_PDI_ALIAS] =
          {false, id, size_field, region_address[PROBE_REGION_PDI_ESTABLISH_B]},
      [PROBE_COMMAND_PDI_REKEY] =
          {false, (uint16_t)(id + 1), size_field,
           region_address[PROBE_REGION_PDI_ESTABLISH_A]},
      [PROBE_COMMAND_NOP_A] = {true, (uint16_t)(id + 2), nop_size_field,
                               region_address[PROBE_REGION_PDI_NOP_A]},
      [PROBE_COMMAND_NOP_B] = {true, (uint16_t)(id + 3), nop_size_field,
                               region_address[PROBE_REGION_PDI_NOP_B]},
      [PROBE_COMMAND_PDI_ESTABLISH] =
          {true, id, size_field, region_address[PROBE_REGION_PDI_ESTABLISH_A]},
  };
  for (int i = 0;
       iree_status_is_ok(status) && i < PROBE_SYNTHESIZED_COMMAND_COUNT; ++i) {
    status = probe_command_build(&probe->split, recipes[i].establish,
                                 recipes[i].id, recipes[i].size_field,
                                 recipes[i].address, &commands[i]);
  }

  if (iree_status_is_ok(status)) {
    uint8_t* base = probe->scratch.view.data;
    memcpy(base + PROBE_REGION_PDI_ESTABLISH_A * probe->region_length, pdi.data,
           pdi.length);
    memcpy(base + PROBE_REGION_PDI_ESTABLISH_B * probe->region_length, pdi.data,
           pdi.length);
    memcpy(base + PROBE_REGION_PDI_NOP_A * probe->region_length, nop_pdi.data,
           nop_pdi.length);
    memcpy(base + PROBE_REGION_PDI_NOP_B * probe->region_length, nop_pdi.data,
           nop_pdi.length);
    for (int i = 0; i < PROBE_SYNTHESIZED_COMMAND_COUNT; ++i) {
      const iree_host_size_t offset =
          (PROBE_REGION_COMMAND_FIRST + i) * probe->region_length;
      memcpy(base + offset, commands[i].data, commands[i].length);
      probe->commands[i] = (amdf_xdna_kernel_command_t){
          .memory = probe->scratch.memory,
          .byte_offset = offset,
          .byte_length = commands[i].length,
      };
    }
    status = probe_allocation_sync(probe, &probe->scratch,
                                   AMDF_HOST_CACHE_OPERATION_FLUSH);
  }

  if (iree_status_is_ok(status)) {
    fprintf(stdout,
            "SPLIT inv0=%" PRIhsz "B/%u ops establish=%" PRIhsz
            "B/%u ops control=%" PRIhsz "B/%u ops\n",
            inv0.data_length,
            probe->split.establish_op_count + probe->split.control_op_count,
            probe->split.establish.data_length, probe->split.establish_op_count,
            probe->split.control.data_length, probe->split.control_op_count);
    fprintf(stdout,
            "PDI establish=%" PRIhsz "B (cdo %" PRIhsz "B) nop=%" PRIhsz
            "B ids=%u..%u size_field=%s region=%" PRIhsz "B scratch=0x%" PRIx64
            "\n",
            pdi.length, cdo.length, nop_pdi.length, id, id + 3,
            FLAG_pdi_size_field ? "actual" : "zero", probe->region_length,
            probe->scratch.address);
    status = probe_write_dump("inv0.bin", inv0, probe->host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = probe_write_dump("inv1.bin", inv1, probe->host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = probe_write_dump("establish.cdo", probe_bytes_span(&cdo),
                              probe->host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = probe_write_dump("establish.pdi", probe_bytes_span(&pdi),
                              probe->host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = probe_write_dump("nop.pdi", probe_bytes_span(&nop_pdi),
                              probe->host_allocator);
  }
  for (int i = 0;
       iree_status_is_ok(status) && i < PROBE_SYNTHESIZED_COMMAND_COUNT; ++i) {
    char name[64];
    snprintf(name, sizeof(name), "%s.bin", probe_command_names[i]);
    status = probe_write_dump(name, probe_bytes_span(&commands[i]),
                              probe->host_allocator);
  }
  probe_bytes_reset(&cdo);
  probe_bytes_reset(&nop_cdo);
  probe_bytes_reset(&pdi);
  probe_bytes_reset(&nop_pdi);
  for (int i = 0; i < PROBE_SYNTHESIZED_COMMAND_COUNT; ++i) {
    probe_bytes_reset(&commands[i]);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Calls and modes
//===----------------------------------------------------------------------===//

static int64_t probe_now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000ll + ts.tv_nsec;
}

static void probe_sleep_ms(int32_t milliseconds) {
  struct timespec ts = {milliseconds / 1000, (milliseconds % 1000) * 1000000l};
  while (nanosleep(&ts, &ts) != 0) {
  }
}

static uint32_t probe_pattern(uint32_t seed, uint32_t binding, uint32_t word) {
  uint32_t x = seed * UINT32_C(0x9E3779B1) ^ binding * UINT32_C(0x85EBCA77) ^
               word * UINT32_C(0xC2B2AE3D);
  x ^= x >> 15;
  x *= UINT32_C(0x2C1B3C6D);
  x ^= x >> 12;
  return x;
}

static iree_status_t probe_prepare_call(probe_t* probe, uint32_t seed) {
  for (uint32_t b = 0; b < probe->binding_count; ++b) {
    probe_allocation_t* allocation = &probe->bindings[b].allocation;
    uint32_t* words = (uint32_t*)allocation->view.data;
    const iree_host_size_t word_count = allocation->view.data_length / 4;
    for (iree_host_size_t j = 0; j < word_count; ++j) {
      words[j] = probe->bindings[b].is_output
                     ? UINT32_C(0xDEADBEEF)
                     : probe_pattern(seed, b, (uint32_t)j);
    }
    IREE_RETURN_IF_ERROR(probe_allocation_sync(
        probe, allocation, AMDF_HOST_CACHE_OPERATION_FLUSH));
  }
  return iree_ok_status();
}

// Returns the number of wrong output words under the --kernel oracle.
static iree_status_t probe_verify_call(probe_t* probe, uint32_t* out_wrong) {
  *out_wrong = 0;
  for (uint32_t b = 0; b < probe->binding_count; ++b) {
    if (probe->bindings[b].is_output) {
      IREE_RETURN_IF_ERROR(
          probe_allocation_sync(probe, &probe->bindings[b].allocation,
                                AMDF_HOST_CACHE_OPERATION_INVALIDATE));
    }
  }
  if (strcmp(FLAG_kernel, "mul_i32") == 0) {
    const uint32_t* lhs =
        (const uint32_t*)probe->bindings[0].allocation.view.data;
    const uint32_t* rhs =
        (const uint32_t*)probe->bindings[1].allocation.view.data;
    const uint32_t* out =
        (const uint32_t*)probe->bindings[2].allocation.view.data;
    for (uint32_t j = 0; j < 16; ++j) {
      if (out[j] != lhs[j] * rhs[j]) {
        ++*out_wrong;
      }
    }
  } else if (strcmp(FLAG_kernel, "copy") == 0) {
    const uint32_t* in =
        (const uint32_t*)probe->bindings[0].allocation.view.data;
    const uint32_t* out =
        (const uint32_t*)probe->bindings[1].allocation.view.data;
    const iree_host_size_t word_count =
        probe->bindings[0].allocation.view.data_length / 4;
    for (iree_host_size_t j = 0; j < word_count; ++j) {
      if (out[j] != in[j]) {
        ++*out_wrong;
      }
    }
  }
  return iree_ok_status();
}

typedef enum probe_outcome_e {
  PROBE_OUTCOME_OK = 0,
  PROBE_OUTCOME_WRONG,
  PROBE_OUTCOME_HANG,
} probe_outcome_t;

static iree_status_t probe_call(probe_t* probe, probe_command_t command,
                                uint32_t seed, int64_t* out_ns,
                                uint32_t* out_wrong,
                                probe_outcome_t* out_outcome) {
  IREE_RETURN_IF_ERROR(probe_prepare_call(probe, seed));
  const amdf_xdna_kernel_queue_submission_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO,
      .structure_size = sizeof(info),
      .command_count = 1,
      .commands = &probe->commands[command],
  };
  uint64_t submission = 0;
  const int64_t start = probe_now_ns();
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      probe->xdna_api->kernel_queue_submit(probe->queue, &info, &submission),
      "kernel_queue_submit"));
  amdf_status_t wait_status = probe->api->kernel_queue_wait(
      probe->queue, submission, (uint64_t)FLAG_timeout_ms * 1000000ull, 0);
  *out_ns = probe_now_ns() - start;
  if (amdf_status_code(wait_status) == AMDF_STATUS_CODE_DEADLINE_EXCEEDED) {
    fprintf(stdout, "HANG command=%s after %d ms; draining up to %d ms\n",
            probe_command_names[command], FLAG_timeout_ms, FLAG_drain_ms);
    fflush(stdout);
    const int64_t drain_start = probe_now_ns();
    wait_status = probe->api->kernel_queue_wait(
        probe->queue, submission, (uint64_t)FLAG_drain_ms * 1000000ull, 0);
    if (amdf_status_code(wait_status) == AMDF_STATUS_CODE_DEADLINE_EXCEEDED) {
      fprintf(stdout, "STUCK command=%s; exiting without teardown\n",
              probe_command_names[command]);
      fflush(stdout);
      _exit(3);
    }
    iree_status_t drained = IREE_HAL_AMD_STATUS_FROM_AMDF(wait_status, "drain");
    fprintf(stdout, "DRAINED after %.1f ms: %s\n",
            (probe_now_ns() - drain_start) / 1e6,
            iree_status_code_string(iree_status_code(drained)));
    iree_status_ignore(drained);
    *out_outcome = PROBE_OUTCOME_HANG;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      IREE_HAL_AMD_STATUS_FROM_AMDF(wait_status, "kernel_queue_wait"));
  IREE_RETURN_IF_ERROR(probe_verify_call(probe, out_wrong));
  *out_outcome = *out_wrong ? PROBE_OUTCOME_WRONG : PROBE_OUTCOME_OK;
  return iree_ok_status();
}

static int probe_compare_i64(const void* a, const void* b) {
  const int64_t x = *(const int64_t*)a, y = *(const int64_t*)b;
  return (x > y) - (x < y);
}

static iree_status_t probe_run_mode(probe_t* probe, const probe_mode_t* mode,
                                    uint32_t* seed, bool* out_stop) {
  int64_t ns = 0;
  uint32_t wrong = 0;
  probe_outcome_t outcome = PROBE_OUTCOME_OK;
  if (mode->prime) {
    IREE_RETURN_IF_ERROR(probe_call(probe, PROBE_COMMAND_ESTABLISH, (*seed)++,
                                    &ns, &wrong, &outcome));
    fprintf(stdout, "PRIME mode=%s inv0 %.1f us wrong_words=%u%s\n", mode->name,
            ns / 1e3, wrong, outcome == PROBE_OUTCOME_HANG ? " HANG" : "");
    if (outcome != PROBE_OUTCOME_OK) {
      *out_stop = outcome == PROBE_OUTCOME_HANG;
      return iree_ok_status();
    }
  }
  const int32_t call_count = FLAG_calls;
  int64_t* times = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      probe->host_allocator, call_count * sizeof(int64_t), (void**)&times));
  int32_t completed = 0, wrong_calls = 0;
  iree_status_t status = iree_ok_status();
  for (int32_t i = 0; iree_status_is_ok(status) && i < call_count; ++i) {
    if (FLAG_gap_ms > 0) {
      probe_sleep_ms(FLAG_gap_ms);
    }
    const bool interposed = i == FLAG_interpose_call;
    if (interposed && FLAG_interpose_command[0] != 0) {
      fflush(stdout);
      const int rc = system(FLAG_interpose_command);
      fprintf(stdout, "INTERPOSE mode=%s before_call=%d exit=%d\n", mode->name,
              i, rc);
    }
    const probe_command_t command = mode->sequence[i % mode->sequence_length];
    status = probe_call(probe, command, (*seed)++, &ns, &wrong, &outcome);
    if (!iree_status_is_ok(status)) {
      fprintf(stdout, "ERROR mode=%s call=%d command=%s\n", mode->name, i,
              probe_command_names[command]);
      break;
    }
    if (outcome == PROBE_OUTCOME_HANG) {
      fprintf(stdout, "HANG mode=%s call=%d command=%s\n", mode->name, i,
              probe_command_names[command]);
      *out_stop = true;
      break;
    }
    times[completed++] = ns;
    if (outcome == PROBE_OUTCOME_WRONG) {
      ++wrong_calls;
    }
    if (FLAG_trace_calls || i == 0 || interposed ||
        outcome != PROBE_OUTCOME_OK) {
      fprintf(stdout,
              "CALL mode=%s call=%d command=%s %.1f us wrong_words=%u\n",
              mode->name, i, probe_command_names[command], ns / 1e3, wrong);
    }
  }
  if (completed > 0) {
    const double first_us = times[0] / 1e3;
    int64_t* rest = times + (completed > 1 ? 1 : 0);
    const int n = completed > 1 ? completed - 1 : 1;
    qsort(rest, n, sizeof(*rest), probe_compare_i64);
    fprintf(stdout,
            "RESULT mode=%s calls=%d wrong_calls=%d first_us=%.1f "
            "rest: p10=%.1f median=%.1f p90=%.1f max=%.1f us\n",
            mode->name, completed, wrong_calls, first_us, rest[n / 10] / 1e3,
            rest[n / 2] / 1e3, rest[(n * 9) / 10] / 1e3, rest[n - 1] / 1e3);
  }
  fflush(stdout);
  iree_allocator_free(probe->host_allocator, times);
  return status;
}

// Resolves each --modes entry, or validates the list when |probe| is NULL.
static iree_status_t probe_run_modes(probe_t* probe) {
  uint32_t seed = 1;
  bool stop = false;
  iree_string_view_t remaining = iree_make_cstring_view(FLAG_modes);
  while (!stop && !iree_string_view_is_empty(remaining)) {
    iree_string_view_t name;
    iree_string_view_split(remaining, ',', &name, &remaining);
    const probe_mode_t* mode = NULL;
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(probe_modes); ++i) {
      if (iree_string_view_equal(name,
                                 iree_make_cstring_view(probe_modes[i].name))) {
        mode = &probe_modes[i];
      }
    }
    if (!mode) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown mode '%.*s'", (int)name.size, name.data);
    }
    if (probe) {
      IREE_RETURN_IF_ERROR(probe_run_mode(probe, mode, &seed, &stop));
    }
  }
  return stop ? iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED,
                                 "a probe call did not complete")
              : iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

static iree_status_t probe_execute(probe_t* probe,
                                   iree_byte_sequence_t* image_bytes) {
  iree_hal_amd_xdna_aie2p_target_t target;
  IREE_RETURN_IF_ERROR(probe_open_device(probe, &target));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_create(
      image_bytes, &target, probe->host_allocator, &probe->image));
  if (FLAG_entry[0] != 0) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_find_entry(
        probe->image, iree_make_cstring_view(FLAG_entry),
        &probe->entry_ordinal));
  }
  probe->entry = iree_hal_amd_xdna_image_tables_entry(
      iree_hal_amd_xdna_image_tables(probe->image), probe->entry_ordinal);
  if ((strcmp(FLAG_kernel, "mul_i32") == 0 &&
       probe->entry.binding_count != 3) ||
      (strcmp(FLAG_kernel, "copy") == 0 && probe->entry.binding_count != 2)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--kernel=%s does not match %u bindings",
                            FLAG_kernel, probe->entry.binding_count);
  }
  IREE_RETURN_IF_ERROR(probe_prepare_bindings(probe));
  IREE_RETURN_IF_ERROR(probe_prepare_storage(probe));
  IREE_RETURN_IF_ERROR(probe_prepare_commands(probe));
  if (FLAG_host_only) {
    return iree_ok_status();
  }
  const amdf_xdna_kernel_queue_create_info_t queue_info = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO,
      .structure_size = sizeof(queue_info),
      .queue_family_ordinal = probe->queue_family_ordinal,
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      probe->xdna_api->kernel_queue_create(probe->context, &queue_info,
                                           &probe->queue),
      "xdna.kernel_queue_create"));
  return probe_run_modes(probe);
}

static iree_status_t probe_deinitialize(probe_t* probe) {
  if (probe->queue) {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        probe->api->kernel_queue_destroy(probe->queue),
        "kernel_queue_destroy"));
    probe->queue = NULL;
  }
  IREE_RETURN_IF_ERROR(probe_allocation_release(probe, &probe->scratch));
  for (uint32_t i = 0; probe->storage_allocations && i < probe->storage_count;
       ++i) {
    IREE_RETURN_IF_ERROR(
        probe_allocation_release(probe, &probe->storage_allocations[i]));
  }
  iree_allocator_free(probe->host_allocator, probe->storage_allocations);
  probe->storage_allocations = NULL;
  for (uint32_t i = 0; probe->bindings && i < probe->binding_count; ++i) {
    iree_hal_buffer_release(probe->bindings[i].buffer);
    IREE_RETURN_IF_ERROR(
        probe_allocation_release(probe, &probe->bindings[i].allocation));
  }
  iree_allocator_free(probe->host_allocator, probe->bindings);
  probe->bindings = NULL;
  iree_hal_amd_xdna_image_destroy(probe->image);
  probe->image = NULL;
  if (probe->context) {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        probe->xdna_api->context_destroy(probe->context), "context_destroy"));
    probe->context = NULL;
  }
  if (probe->device) {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        probe->api->device_destroy(probe->device), "device_destroy"));
    probe->device = NULL;
  }
  if (probe->endpoint) {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        probe->api->endpoint_close(probe->endpoint), "endpoint_close"));
    probe->endpoint = NULL;
  }
  if (probe->instance) {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        probe->api->instance_destroy(probe->instance), "instance_destroy"));
    probe->instance = NULL;
  }
  return iree_ok_status();
}

static iree_status_t probe_main(void) {
  if (FLAG_image[0] == 0 || FLAG_columns < 1 || FLAG_columns > 8 ||
      FLAG_calls < 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--image, --columns in [1, 8] and --calls >= 1 "
                            "are required");
  }
  IREE_RETURN_IF_ERROR(probe_run_modes(NULL));
  probe_t probe = {.host_allocator = iree_allocator_system()};
  probe.pdi_id_base = FLAG_pdi_id_base > 0
                          ? (uint16_t)FLAG_pdi_id_base
                          : (uint16_t)(0x1000 + (getpid() % 0x3000) * 4);
  iree_io_file_contents_t* contents = NULL;
  IREE_RETURN_IF_ERROR(iree_io_file_contents_read(
      iree_make_cstring_view(FLAG_image), probe.host_allocator, &contents));
  iree_byte_span_t span = iree_byte_span_empty();
  iree_byte_sequence_t* image_bytes = NULL;
  iree_status_t status = iree_allocator_clone(
      probe.host_allocator, contents->const_buffer, (void**)&span.data);
  if (iree_status_is_ok(status)) {
    span.data_length = contents->buffer.data_length;
    status = iree_byte_sequence_create_from_span_move(
        &span, probe.host_allocator, &image_bytes);
  }
  iree_allocator_free(probe.host_allocator, span.data);
  iree_io_file_contents_free(contents);
  if (iree_status_is_ok(status)) {
    status = probe_execute(&probe, image_bytes);
  }
  status = iree_status_join(status, probe_deinitialize(&probe));
  iree_byte_sequence_release(image_bytes);
  return status;
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "iree-xdna-retention-probe",
      "Measures LOADPDI-based state retention for one XDNA entry.\n"
      "Example: --image=mul_i32_npu4.xdna --columns=1 --kernel=mul_i32\n"
      "  --modes=establish,continuation,nop-establish,pdi-establish,pdi\n"
      "Prints SPLIT/PDI, then per mode CALL/HANG/ERROR and one RESULT line.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  iree_status_t status =
      argc == 1 ? probe_main()
                : iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                   "unexpected positional argument");
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
