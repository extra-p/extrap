#ifndef MESSAGE_STREAM_H
#define MESSAGE_STREAM_H

#include <sstream>
#include <ostream>

namespace EXTRAP
{
/**
 * This class implements a stream that formats output for error massages,
 * warnings, or other output. It directs the output to any other output stream.
 * Thus, it acts as a layer between the output stream and the application.
 * to apply some formatting. 1. It inserts line breaks to ensure a maximum width
 * of the output. 2. It puts a prefix in front of new messages and 3. it
 * indents multi-line messages.
 */
class MessageStream
{
public:
    MessageStream( const std::string& prefix );

    template<typename T>
    MessageStream&
    operator<<( T value )
    {
        m_current << value;
        return *this;
    }

    virtual MessageStream&
    operator<<( std::ostream& ( *pf )( std::ostream & ) );

    void
    setOutputStream( std::ostream* output );

private:
    std::stringstream m_current;
    std::ostream*     m_output_stream;
    std::string       m_prefix;
    int               m_width;

public:
    static MessageStream Errors;
    static MessageStream Warnings;
    static MessageStream Notes;
};
};

#endif