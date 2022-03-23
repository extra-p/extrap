// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: ProfilerStringTable.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_ProfilerStringTable_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_ProfilerStringTable_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3014000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3014000 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_ProfilerStringTable_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_ProfilerStringTable_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxiliaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[1]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_ProfilerStringTable_2eproto;
class ProfilerStringTable;
class ProfilerStringTableDefaultTypeInternal;
extern ProfilerStringTableDefaultTypeInternal _ProfilerStringTable_default_instance_;
PROTOBUF_NAMESPACE_OPEN
template<> ::ProfilerStringTable* Arena::CreateMaybeMessage<::ProfilerStringTable>(Arena*);
PROTOBUF_NAMESPACE_CLOSE

// ===================================================================

class ProfilerStringTable PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:ProfilerStringTable) */ {
 public:
  inline ProfilerStringTable() : ProfilerStringTable(nullptr) {}
  virtual ~ProfilerStringTable();

  ProfilerStringTable(const ProfilerStringTable& from);
  ProfilerStringTable(ProfilerStringTable&& from) noexcept
    : ProfilerStringTable() {
    *this = ::std::move(from);
  }

  inline ProfilerStringTable& operator=(const ProfilerStringTable& from) {
    CopyFrom(from);
    return *this;
  }
  inline ProfilerStringTable& operator=(ProfilerStringTable&& from) noexcept {
    if (GetArena() == from.GetArena()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance);
  }
  inline ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const ProfilerStringTable& default_instance();

  static inline const ProfilerStringTable* internal_default_instance() {
    return reinterpret_cast<const ProfilerStringTable*>(
               &_ProfilerStringTable_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(ProfilerStringTable& a, ProfilerStringTable& b) {
    a.Swap(&b);
  }
  inline void Swap(ProfilerStringTable* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(ProfilerStringTable* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline ProfilerStringTable* New() const final {
    return CreateMaybeMessage<ProfilerStringTable>(nullptr);
  }

  ProfilerStringTable* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<ProfilerStringTable>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const ProfilerStringTable& from);
  void MergeFrom(const ProfilerStringTable& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(ProfilerStringTable* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "ProfilerStringTable";
  }
  protected:
  explicit ProfilerStringTable(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_ProfilerStringTable_2eproto);
    return ::descriptor_table_ProfilerStringTable_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kStringsFieldNumber = 1,
  };
  // repeated string Strings = 1;
  int strings_size() const;
  private:
  int _internal_strings_size() const;
  public:
  void clear_strings();
  const std::string& strings(int index) const;
  std::string* mutable_strings(int index);
  void set_strings(int index, const std::string& value);
  void set_strings(int index, std::string&& value);
  void set_strings(int index, const char* value);
  void set_strings(int index, const char* value, size_t size);
  std::string* add_strings();
  void add_strings(const std::string& value);
  void add_strings(std::string&& value);
  void add_strings(const char* value);
  void add_strings(const char* value, size_t size);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>& strings() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>* mutable_strings();
  private:
  const std::string& _internal_strings(int index) const;
  std::string* _internal_add_strings();
  public:

  // @@protoc_insertion_point(class_scope:ProfilerStringTable)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string> strings_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_ProfilerStringTable_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// ProfilerStringTable

// repeated string Strings = 1;
inline int ProfilerStringTable::_internal_strings_size() const {
  return strings_.size();
}
inline int ProfilerStringTable::strings_size() const {
  return _internal_strings_size();
}
inline void ProfilerStringTable::clear_strings() {
  strings_.Clear();
}
inline std::string* ProfilerStringTable::add_strings() {
  // @@protoc_insertion_point(field_add_mutable:ProfilerStringTable.Strings)
  return _internal_add_strings();
}
inline const std::string& ProfilerStringTable::_internal_strings(int index) const {
  return strings_.Get(index);
}
inline const std::string& ProfilerStringTable::strings(int index) const {
  // @@protoc_insertion_point(field_get:ProfilerStringTable.Strings)
  return _internal_strings(index);
}
inline std::string* ProfilerStringTable::mutable_strings(int index) {
  // @@protoc_insertion_point(field_mutable:ProfilerStringTable.Strings)
  return strings_.Mutable(index);
}
inline void ProfilerStringTable::set_strings(int index, const std::string& value) {
  // @@protoc_insertion_point(field_set:ProfilerStringTable.Strings)
  strings_.Mutable(index)->assign(value);
}
inline void ProfilerStringTable::set_strings(int index, std::string&& value) {
  // @@protoc_insertion_point(field_set:ProfilerStringTable.Strings)
  strings_.Mutable(index)->assign(std::move(value));
}
inline void ProfilerStringTable::set_strings(int index, const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  strings_.Mutable(index)->assign(value);
  // @@protoc_insertion_point(field_set_char:ProfilerStringTable.Strings)
}
inline void ProfilerStringTable::set_strings(int index, const char* value, size_t size) {
  strings_.Mutable(index)->assign(
    reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_set_pointer:ProfilerStringTable.Strings)
}
inline std::string* ProfilerStringTable::_internal_add_strings() {
  return strings_.Add();
}
inline void ProfilerStringTable::add_strings(const std::string& value) {
  strings_.Add()->assign(value);
  // @@protoc_insertion_point(field_add:ProfilerStringTable.Strings)
}
inline void ProfilerStringTable::add_strings(std::string&& value) {
  strings_.Add(std::move(value));
  // @@protoc_insertion_point(field_add:ProfilerStringTable.Strings)
}
inline void ProfilerStringTable::add_strings(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  strings_.Add()->assign(value);
  // @@protoc_insertion_point(field_add_char:ProfilerStringTable.Strings)
}
inline void ProfilerStringTable::add_strings(const char* value, size_t size) {
  strings_.Add()->assign(reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_add_pointer:ProfilerStringTable.Strings)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>&
ProfilerStringTable::strings() const {
  // @@protoc_insertion_point(field_list:ProfilerStringTable.Strings)
  return strings_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>*
ProfilerStringTable::mutable_strings() {
  // @@protoc_insertion_point(field_mutable_list:ProfilerStringTable.Strings)
  return &strings_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)


// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_ProfilerStringTable_2eproto
