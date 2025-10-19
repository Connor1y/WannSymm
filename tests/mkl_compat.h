/* MKL compatibility header for tests
 * This header provides compatibility when MKL is not available
 * by mapping MKL functions to standard LAPACKE and CBLAS
 */

#ifndef MKL_COMPAT_H
#define MKL_COMPAT_H

#ifdef __MKL_H__
/* Real MKL is already included */
#else
/* Use standard LAPACKE and CBLAS */
#include <lapacke.h>
#include <cblas.h>
#endif

#endif /* MKL_COMPAT_H */
