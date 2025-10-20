/* MKL compatibility header for tests
 * This header provides compatibility when MKL is not available
 * by mapping MKL functions to standard LAPACKE and CBLAS
 */

#ifndef __MKL_H__
#define __MKL_H__

#include <lapacke.h>
#include <cblas.h>

#endif /* __MKL_H__ */
