#ifndef IOHELPER_H
#define IOHELPER_H

#include "Types.h"
#include <string>
#include <fstream>
#include <stdint.h>

namespace EXTRAP
{
/**
 * This class is supposed to help with handling File I/O independant of the endianness of the file and the system.
 * It will write a checkpoint to the file at the first write and read it from file on the first read. It is crucial that all read and write accesses
 * to the file are handled by the helper.
 * Write is done in the systems endianness, reads will be converted if necessary.
 */
class IoHelper
{
public:

    /**
     * Creates an IoHelper for reading, this includes checking the endianness of the file.
     * The stream has to be opened and closed by the client.
     *
     * This implementation calls the check_system_and_file_endianness method for determining the endianness
     **/
    IoHelper( std::ifstream* stream );

    /**
     * Creates an IoHelper for writing, this includes the checkpoint for endianness.
     * The stream has to be opened and closed by the client.
     *
     * This implementation calls the write_endianness method for writing the endianness checkpoint if not
     **/
    IoHelper( std::ofstream* stream,
              bool           fakeEndianness = false );
    /**
     * writes a 64 bit ID to the stream. Endianness stays as is.
     * Return true on success, false if this IoHelper instance is not initialized for writing.
     **/
    bool
    writeId(
        int64_t value ) const;

    /**
     * writes a 64 bit Integer to the stream. Endianness stays as is.
     * Return true on success, false if this IoHelper instance is not initialized for writing.
     **/
    bool
    writeInt(
        const int64_t value ) const;

    /**
     * writes an EXTRAP::Value(double) to the stream. Endianness stays as is.
     * Return true on success, false if this IoHelper instance is not initialized for writing.
     **/
    bool
    writeValue(
        EXTRAP::Value value ) const;

    /**
     * writes a String to the stream in two parts. First the length as a 32 bit unsigned integer, then the string as sequence of chars.
     * Return true on success, false if this IoHelper instance is not initialized for writing.
     **/
    bool
    writeString(           const std::string value ) const;

    /**
     * Reads an ID from the stream as 64 bit Integer.
     * If the File was written using a different Endianness as the Systems endianness a conversion is applied.
     **/
    int64_t
    readId() const;

    /**
     * Reads an 64 bit Integer from the file.
     * If the File was written using a different Endianness as the Systems endianness a conversion is applied.
     **/
    int
    readInt() const;

    /**
     * Reads an EXTRAP::Value(double) from the stream.
     * If the File was written using a different Endianness as the Systems endianness a conversion is applied.
     **/
    EXTRAP::Value
    readValue() const;

    /**
     * Reads a String from the stream in two steps. First reads the number of chars to get as an unsigned 32 bit integer, then the chars.
     * If the File was written using a different Endianness as the Systems endianness a conversion is applied.
     **/
    std::string
    readString() const;

private:
//Indicates whether a conversion of input data is needed or not
    EndiannessConversion m_conversion;

    std::ifstream* m_input_stream;

    std::ofstream* m_output_stream;

    /**
     * checks whether or not the endianness of the stream matches the system endianness and sets the internal state accordingly
     **/
    void
    check_system_and_file_endianness();

    /**
     * writes an indicator for the endianness to the stream.
     **/
    void
    write_endianness() const;

    //Do not use this! For testing only!
    void
    write_endianness_wrong() const;

    /**
     * Reads a 32 bit unsigned integer from the stream. Helper Method for reading Strings.
     * If this is the first read from the stream, endianness will be checked here.
     **/
    uint32_t
    read_uint32_t() const;

    /**
     * Writes a 32 bit unsigned integer from the stream. Helper Method for writing Strings.
     * If this is the first write to the stream an endianness indicator is written to the stream.
     **/
    bool
    write_uint32_t(
        uint32_t value ) const;
};
};

#endif