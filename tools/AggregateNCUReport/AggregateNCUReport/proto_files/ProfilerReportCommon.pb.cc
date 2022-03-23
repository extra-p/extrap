// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: ProfilerReportCommon.proto

#include "ProfilerReportCommon.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
class ExecutableSettingsDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<ExecutableSettings> _instance;
} _ExecutableSettings_default_instance_;
static void InitDefaultsscc_info_ExecutableSettings_ProfilerReportCommon_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::_ExecutableSettings_default_instance_;
    new (ptr) ::ExecutableSettings();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_ExecutableSettings_ProfilerReportCommon_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, 0, InitDefaultsscc_info_ExecutableSettings_ProfilerReportCommon_2eproto}, {}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_ProfilerReportCommon_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_ProfilerReportCommon_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_ProfilerReportCommon_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_ProfilerReportCommon_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::ExecutableSettings, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::ExecutableSettings, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::ExecutableSettings, executablepath_),
  PROTOBUF_FIELD_OFFSET(::ExecutableSettings, workdirectory_),
  PROTOBUF_FIELD_OFFSET(::ExecutableSettings, cmdlineagruments_),
  PROTOBUF_FIELD_OFFSET(::ExecutableSettings, environment_),
  0,
  1,
  2,
  3,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 9, sizeof(::ExecutableSettings)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::_ExecutableSettings_default_instance_),
};

const char descriptor_table_protodef_ProfilerReportCommon_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\032ProfilerReportCommon.proto\"r\n\022Executab"
  "leSettings\022\026\n\016ExecutablePath\030\001 \002(\t\022\025\n\rWo"
  "rkDirectory\030\002 \001(\t\022\030\n\020CmdlineAgruments\030\003 "
  "\001(\t\022\023\n\013Environment\030\004 \001(\t"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_ProfilerReportCommon_2eproto_deps[1] = {
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_ProfilerReportCommon_2eproto_sccs[1] = {
  &scc_info_ExecutableSettings_ProfilerReportCommon_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_ProfilerReportCommon_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_ProfilerReportCommon_2eproto = {
  false, false, descriptor_table_protodef_ProfilerReportCommon_2eproto, "ProfilerReportCommon.proto", 144,
  &descriptor_table_ProfilerReportCommon_2eproto_once, descriptor_table_ProfilerReportCommon_2eproto_sccs, descriptor_table_ProfilerReportCommon_2eproto_deps, 1, 0,
  schemas, file_default_instances, TableStruct_ProfilerReportCommon_2eproto::offsets,
  file_level_metadata_ProfilerReportCommon_2eproto, 1, file_level_enum_descriptors_ProfilerReportCommon_2eproto, file_level_service_descriptors_ProfilerReportCommon_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_ProfilerReportCommon_2eproto = (static_cast<void>(::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_ProfilerReportCommon_2eproto)), true);

// ===================================================================

class ExecutableSettings::_Internal {
 public:
  using HasBits = decltype(std::declval<ExecutableSettings>()._has_bits_);
  static void set_has_executablepath(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_workdirectory(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
  static void set_has_cmdlineagruments(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
  static void set_has_environment(HasBits* has_bits) {
    (*has_bits)[0] |= 8u;
  }
  static bool MissingRequiredFields(const HasBits& has_bits) {
    return ((has_bits[0] & 0x00000001) ^ 0x00000001) != 0;
  }
};

ExecutableSettings::ExecutableSettings(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:ExecutableSettings)
}
ExecutableSettings::ExecutableSettings(const ExecutableSettings& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  executablepath_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (from._internal_has_executablepath()) {
    executablepath_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_executablepath(), 
      GetArena());
  }
  workdirectory_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (from._internal_has_workdirectory()) {
    workdirectory_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_workdirectory(), 
      GetArena());
  }
  cmdlineagruments_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (from._internal_has_cmdlineagruments()) {
    cmdlineagruments_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_cmdlineagruments(), 
      GetArena());
  }
  environment_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (from._internal_has_environment()) {
    environment_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_environment(), 
      GetArena());
  }
  // @@protoc_insertion_point(copy_constructor:ExecutableSettings)
}

void ExecutableSettings::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_ExecutableSettings_ProfilerReportCommon_2eproto.base);
  executablepath_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  workdirectory_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  cmdlineagruments_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  environment_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

ExecutableSettings::~ExecutableSettings() {
  // @@protoc_insertion_point(destructor:ExecutableSettings)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void ExecutableSettings::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
  executablepath_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  workdirectory_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  cmdlineagruments_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  environment_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void ExecutableSettings::ArenaDtor(void* object) {
  ExecutableSettings* _this = reinterpret_cast< ExecutableSettings* >(object);
  (void)_this;
}
void ExecutableSettings::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void ExecutableSettings::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const ExecutableSettings& ExecutableSettings::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_ExecutableSettings_ProfilerReportCommon_2eproto.base);
  return *internal_default_instance();
}


void ExecutableSettings::Clear() {
// @@protoc_insertion_point(message_clear_start:ExecutableSettings)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x0000000fu) {
    if (cached_has_bits & 0x00000001u) {
      executablepath_.ClearNonDefaultToEmpty();
    }
    if (cached_has_bits & 0x00000002u) {
      workdirectory_.ClearNonDefaultToEmpty();
    }
    if (cached_has_bits & 0x00000004u) {
      cmdlineagruments_.ClearNonDefaultToEmpty();
    }
    if (cached_has_bits & 0x00000008u) {
      environment_.ClearNonDefaultToEmpty();
    }
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* ExecutableSettings::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // required string ExecutablePath = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          auto str = _internal_mutable_executablepath();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          #ifndef NDEBUG
          ::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "ExecutableSettings.ExecutablePath");
          #endif  // !NDEBUG
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional string WorkDirectory = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          auto str = _internal_mutable_workdirectory();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          #ifndef NDEBUG
          ::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "ExecutableSettings.WorkDirectory");
          #endif  // !NDEBUG
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional string CmdlineAgruments = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 26)) {
          auto str = _internal_mutable_cmdlineagruments();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          #ifndef NDEBUG
          ::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "ExecutableSettings.CmdlineAgruments");
          #endif  // !NDEBUG
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional string Environment = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 34)) {
          auto str = _internal_mutable_environment();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          #ifndef NDEBUG
          ::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "ExecutableSettings.Environment");
          #endif  // !NDEBUG
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag,
            _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
            ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  _has_bits_.Or(has_bits);
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* ExecutableSettings::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:ExecutableSettings)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // required string ExecutablePath = 1;
  if (cached_has_bits & 0x00000001u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
      this->_internal_executablepath().data(), static_cast<int>(this->_internal_executablepath().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SERIALIZE,
      "ExecutableSettings.ExecutablePath");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_executablepath(), target);
  }

  // optional string WorkDirectory = 2;
  if (cached_has_bits & 0x00000002u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
      this->_internal_workdirectory().data(), static_cast<int>(this->_internal_workdirectory().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SERIALIZE,
      "ExecutableSettings.WorkDirectory");
    target = stream->WriteStringMaybeAliased(
        2, this->_internal_workdirectory(), target);
  }

  // optional string CmdlineAgruments = 3;
  if (cached_has_bits & 0x00000004u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
      this->_internal_cmdlineagruments().data(), static_cast<int>(this->_internal_cmdlineagruments().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SERIALIZE,
      "ExecutableSettings.CmdlineAgruments");
    target = stream->WriteStringMaybeAliased(
        3, this->_internal_cmdlineagruments(), target);
  }

  // optional string Environment = 4;
  if (cached_has_bits & 0x00000008u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
      this->_internal_environment().data(), static_cast<int>(this->_internal_environment().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SERIALIZE,
      "ExecutableSettings.Environment");
    target = stream->WriteStringMaybeAliased(
        4, this->_internal_environment(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:ExecutableSettings)
  return target;
}

size_t ExecutableSettings::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:ExecutableSettings)
  size_t total_size = 0;

  // required string ExecutablePath = 1;
  if (_internal_has_executablepath()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_executablepath());
  }
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x0000000eu) {
    // optional string WorkDirectory = 2;
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
          this->_internal_workdirectory());
    }

    // optional string CmdlineAgruments = 3;
    if (cached_has_bits & 0x00000004u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
          this->_internal_cmdlineagruments());
    }

    // optional string Environment = 4;
    if (cached_has_bits & 0x00000008u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
          this->_internal_environment());
    }

  }
  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void ExecutableSettings::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:ExecutableSettings)
  GOOGLE_DCHECK_NE(&from, this);
  const ExecutableSettings* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<ExecutableSettings>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:ExecutableSettings)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:ExecutableSettings)
    MergeFrom(*source);
  }
}

void ExecutableSettings::MergeFrom(const ExecutableSettings& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:ExecutableSettings)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x0000000fu) {
    if (cached_has_bits & 0x00000001u) {
      _internal_set_executablepath(from._internal_executablepath());
    }
    if (cached_has_bits & 0x00000002u) {
      _internal_set_workdirectory(from._internal_workdirectory());
    }
    if (cached_has_bits & 0x00000004u) {
      _internal_set_cmdlineagruments(from._internal_cmdlineagruments());
    }
    if (cached_has_bits & 0x00000008u) {
      _internal_set_environment(from._internal_environment());
    }
  }
}

void ExecutableSettings::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:ExecutableSettings)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void ExecutableSettings::CopyFrom(const ExecutableSettings& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:ExecutableSettings)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool ExecutableSettings::IsInitialized() const {
  if (_Internal::MissingRequiredFields(_has_bits_)) return false;
  return true;
}

void ExecutableSettings::InternalSwap(ExecutableSettings* other) {
  using std::swap;
  _internal_metadata_.Swap<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  executablepath_.Swap(&other->executablepath_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
  workdirectory_.Swap(&other->workdirectory_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
  cmdlineagruments_.Swap(&other->cmdlineagruments_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
  environment_.Swap(&other->environment_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
}

::PROTOBUF_NAMESPACE_ID::Metadata ExecutableSettings::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::ExecutableSettings* Arena::CreateMaybeMessage< ::ExecutableSettings >(Arena* arena) {
  return Arena::CreateMessageInternal< ::ExecutableSettings >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
