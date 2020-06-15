#ifndef UTILITIES_H
#define UTILITIES_H

#include "MessageStream.h"
#include <iostream>

void
matrixInverse( double* matrix,
               int     n );

#define SAFE_RETURN( result ) \
    if ( !result ) \
    { \
        return false; \
    } \

#define SAFE_RETURN_NULL( result ) \
    if ( result == NULL ) \
    { \
        return NULL; \
    } \

#define ErrorStream MessageStream::Errors

#define WarningStream MessageStream::Warnings

#define NoteStream MessageStream::Notes

//#define DebugStream std::cout << "[EXTRA-P] Debug: "
#define DebugStream if ( 0 ) std::cout

#endif