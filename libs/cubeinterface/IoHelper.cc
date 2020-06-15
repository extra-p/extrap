 #include "IoHelper.h"
 #include "Utilities.h"
 #include <limits.h>
 #include <cstring>
namespace EXTRAP
{
IoHelper::IoHelper( std::ifstream* stream )
{
    this->m_output_stream = NULL;
    this->m_input_stream  = stream;
    this->check_system_and_file_endianness();
}

IoHelper::IoHelper( std::ofstream* stream, bool fakeEndianness )
{
    this->m_input_stream  = NULL;
    this->m_output_stream = stream;
    if ( !fakeEndianness )
    {
        this->write_endianness();
    }
    else
    {
        this->write_endianness_wrong();
    }
}

bool
IoHelper::writeId( int64_t value ) const
{
    if ( this->m_output_stream == NULL )
    {
        ErrorStream << "IoHelper is not initialized for writing. Terminating..." << std::endl;
        return false;
    }
    this->m_output_stream->write( reinterpret_cast<char*>( &value ), sizeof( int64_t ) );
    return true;
}

bool
IoHelper::writeInt( const int64_t value ) const
{
    if ( this->m_output_stream == NULL )
    {
        ErrorStream << "IoHelper is not initialized for writing. Terminating..." << std::endl;
        return false;
    }
    int64_t val = value;
    this->m_output_stream->write( reinterpret_cast<char*>( &val ), sizeof( int64_t ) );
    return true;
}

bool
IoHelper::write_uint32_t( const uint32_t value ) const
{
    if ( this->m_output_stream == NULL )
    {
        ErrorStream << "IoHelper is not initialized for writing. Terminating..." << std::endl;
        return false;
    }
    uint32_t val = value;
    this->m_output_stream->write( reinterpret_cast<char*>( &val ), sizeof( uint32_t ) );
    return true;
}

bool
IoHelper::writeValue( EXTRAP::Value value ) const
{
    if ( this->m_output_stream == NULL )
    {
        ErrorStream << "IoHelper is not initialized for writing. Terminating..." << std::endl;
        return false;
    }
    this->m_output_stream->write( reinterpret_cast<char*>( &value ), 8 );
    return true;
}

bool
IoHelper::writeString( const std::string value ) const
{
    if ( this->m_output_stream == NULL )
    {
        ErrorStream << "IoHelper is not initialized for writing. Terminating..." << std::endl;
        return false;
    }
    this->write_uint32_t( value.size() );
    this->m_output_stream->write( value.c_str(), value.size() );
    return true;
}

int64_t
IoHelper::readId() const
{
    int64_t value;
    this->m_input_stream->read( reinterpret_cast<char*>( &value ), sizeof( int64_t ) );
    if ( this->m_conversion == CONVERSION_REVERSE )
    {
        char reverse[ 8 ];
        char arr[ 8 ];
        memcpy( arr, &value, 8 );
        for ( int i = 0; i < 8; i++ )
        {
            reverse[ i ] = arr[ 8 - 1 - i ];
        }
        memcpy( &value, reverse, 8 );
    }
    return value;
}

int
IoHelper::readInt() const
{
    int64_t value;
    this->m_input_stream->read( reinterpret_cast<char*>( &value ), sizeof( int64_t ) );
    if ( this->m_conversion == CONVERSION_REVERSE )
    {
        char reverse[ 8 ];
        char arr[ 8 ];
        memcpy( arr, &value, 8 );
        for ( int i = 0; i < 8; i++ )
        {
            reverse[ i ] = arr[ 8 - 1 - i ];
        }
        memcpy( &value, reverse, 8 );
        DebugStream << "Value read: " << std::hex << "0x" << value << std::endl;
    }
    if ( sizeof( int ) == 8 )
    {
        return ( int )value;
    }
    else if ( value > INT_MAX )
    {
        return INT_MAX;
    }
    else if ( value < INT_MIN )
    {
        return INT_MIN;
    }
    else
    {
        return ( int )value;
    }
}

uint32_t
IoHelper::read_uint32_t() const
{
    uint32_t value;
    this->m_input_stream->read( reinterpret_cast<char*>( &value ), sizeof( uint32_t ) );
    if ( this->m_conversion == CONVERSION_REVERSE )
    {
        char reverse[ 4 ];
        char arr[ 4 ];
        memcpy( arr, &value, 4 );
        for ( int i = 0; i < 4; i++ )
        {
            reverse[ i ] = arr[ 4 - 1 - i ];
        }
        memcpy( &value, reverse, 4 );
        DebugStream << "Value read: " << std::hex << "0x" << value << std::endl;
    }
    return value;
}

EXTRAP::Value
IoHelper::readValue() const
{
    EXTRAP::Value value;
    this->m_input_stream->read( reinterpret_cast<char*>( &value ), 8 );
    if ( this->m_conversion == CONVERSION_REVERSE )
    {
        char arr[ 8 ];
        std::memcpy( arr, &value, 8 );
        char reverse[ 8 ];
        for ( int i = 0; i < 8; i++ )
        {
            reverse[ i ] = arr[ 8 - 1 - i ];
        }
        std::memcpy( &value, reverse, 8 );
    }
    return value;
}

std::string
IoHelper::readString() const
{
    uint32_t    size = this->read_uint32_t();
    std::string value;
    value.resize( size );
    this->m_input_stream->read( &value[ 0 ], size );
    return value;
}
void
IoHelper::write_endianness() const
{
    short s = 1;
    this->m_output_stream->write( reinterpret_cast<char*>( &s ), sizeof( short ) );
}
void
IoHelper::write_endianness_wrong() const
{
    short s = 16;
    this->m_output_stream->write( reinterpret_cast<char*>( &s ), sizeof( short ) );
}

void
IoHelper::check_system_and_file_endianness()
{
    short s;
    this->m_input_stream->read( reinterpret_cast<char*>( &s ), sizeof( short ) );
    if ( s == 1 )
    {
        this->m_conversion = CONVERSION_NONE;
    }
    else
    {
        this->m_conversion = CONVERSION_REVERSE;
    }
}
};