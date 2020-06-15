#include "MessageStream.h"
#include <iostream>

namespace EXTRAP
{
MessageStream MessageStream::Errors( "[Extra-P]: Error: " );
MessageStream MessageStream::Warnings( "[Extra-P]: Warning: " );
MessageStream MessageStream::Notes( "[Extra-P]: Note: " );

MessageStream::  MessageStream( const std::string& prefix )
{
    m_output_stream = &std::cerr;
    m_width         = 80;
    //m_prefix = "[Extra-P]: ";
    m_prefix = prefix;
}

MessageStream&
MessageStream::operator<<( std::ostream& ( *pf )( std::ostream & ) )
{
    ( *m_output_stream ) << m_prefix;
    std::string message = m_current.str();
    while ( message.length() > m_width - 5 )
    {
        std::string::size_type pos = message.rfind( " ", m_width - 5 );
        if ( pos == 0 || pos == std::string::npos )
        {
            pos = m_width - 5;
        }
        ( *m_output_stream ) << message.substr( 0, pos + 1 ) << "\n    ";
        message = message.substr( pos + 1 );
    }
    ( *m_output_stream ) << message << std::endl;
    return *this;
}

void
MessageStream::setOutputStream( std::ostream* output )
{
    m_output_stream = output;
}
};