#pragma once

#ifndef SO_PRIVATE
#define SO_PRIVATE(name) extra_prof_so_private_##name
#define SO_PRIVATE_NAME(name) "extra_prof_so_private_" #name
#endif
