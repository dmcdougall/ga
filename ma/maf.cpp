c
c     $Id: maf.cpp,v 1.3 1994-12-29 06:44:23 og845 Exp $
c

c
c     FORTRAN routines for a portable dynamic memory allocator.
c

#define MAF_INTERNAL

cc
cc    private routines
cc

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      integer function ma_set_sizes ()

      implicit none

#include "f2c.h"
#include "mafdecls.h"

      ma_set_sizes = 0

#ifdef _CRAY
      if (f2c_inform_base_fcd(MT_BYTE, byte_mb(1), byte_mb(2)) .eq.
     $    MA_FALSE) return
#else /* _CRAY */
      if (f2c_inform_base(MT_BYTE, byte_mb(1), byte_mb(2)) .eq.
     $    MA_FALSE) return
#endif /* _CRAY */
      if (f2c_inform_base(MT_INT, int_mb(1), int_mb(2)) .eq.
     $    MA_FALSE) return
      if (f2c_inform_base(MT_LOG, log_mb(1), log_mb(2)) .eq.
     $    MA_FALSE) return
      if (f2c_inform_base(MT_REAL, real_mb(1), real_mb(2)) .eq.
     $    MA_FALSE) return
      if (f2c_inform_base(MT_DBL, dbl_mb(1), dbl_mb(2)) .eq.
     $    MA_FALSE) return
      if (f2c_inform_base(MT_SCPL, scpl_mb(1), scpl_mb(2)) .eq.
     $    MA_FALSE) return
      if (f2c_inform_base(MT_DCPL, dcpl_mb(1), dcpl_mb(2)) .eq.
     $    MA_FALSE) return

      ma_set_sizes = 1

      return
      end

cc
cc    public routines
cc

c     In general, each routine simply calls its corresponding f2c_ C
c     wrapper routine, which performs any necessary argument munging
c     and then calls the corresponding C routine.

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      logical function MA_alloc_get (datatype, nelem, name, memhandle,
     $    index)

      implicit none

      integer datatype
      integer nelem
      character*(*) name
      integer memhandle
      integer index

#include "f2c.h"

      if (f2c_alloc_get(datatype, nelem, name, memhandle, index) .eq.
     $    MA_TRUE) then
          MA_alloc_get = .true.
      else
          MA_alloc_get = .false.
      endif

      return
      end

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      logical function MA_allocate_heap (datatype, nelem, name,
     $    memhandle)

      implicit none

      integer datatype
      integer nelem
      character*(*) name
      integer memhandle

#include "f2c.h"

      if (f2c_allocate_heap(datatype, nelem, name, memhandle) .eq.
     $    MA_TRUE) then
          MA_allocate_heap = .true.
      else
          MA_allocate_heap = .false.
      endif

      return
      end

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      logical function MA_chop_stack (memhandle)

      implicit none

      integer memhandle

#include "f2c.h"

      if (f2c_chop_stack(memhandle) .eq. MA_TRUE) then
          MA_chop_stack = .true.
      else
          MA_chop_stack = .false.
      endif

      return
      end

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      logical function MA_free_heap (memhandle)

      implicit none

      integer memhandle

#include "f2c.h"

      if (f2c_free_heap(memhandle) .eq. MA_TRUE) then
          MA_free_heap = .true.
      else
          MA_free_heap = .false.
      endif

      return
      end

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      logical function MA_get_index (memhandle, index)

      implicit none

      integer memhandle
      integer index

#include "f2c.h"

      if (f2c_get_index(memhandle, index) .eq. MA_TRUE) then
          MA_get_index = .true.
      else
          MA_get_index = .false.
      endif

      return
      end

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      logical function MA_get_next_memhandle (ithandle, memhandle)

      implicit none

      integer ithandle
      integer memhandle

#include "f2c.h"

      if (f2c_get_next_memhandle(ithandle, memhandle) .eq. MA_TRUE)
     $    then
          MA_get_next_memhandle = .true.
      else
          MA_get_next_memhandle = .false.
      endif

      return
      end

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      logical function MA_init (datatype, nominal_stack,
     $    nominal_heap)

      implicit none

      integer datatype
      integer nominal_stack
      integer nominal_heap

#include "f2c.h"

      MA_init = .false.

      if (f2c_init(datatype, nominal_stack, nominal_heap) .eq.
     $    MA_TRUE) then
          MA_init = .true.
      else
          MA_init = .false.
      endif

      return
      end

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      logical function MA_init_memhandle_iterator (ithandle)

      implicit none

      integer ithandle

#include "f2c.h"

      if (f2c_init_memhandle_iterator(ithandle) .eq. MA_TRUE)
     $    then
          MA_init_memhandle_iterator = .true.
      else
          MA_init_memhandle_iterator = .false.
      endif

      return
      end

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      integer function MA_inquire_avail (datatype)

      implicit none

      integer datatype

#include "f2c.h"

      MA_inquire_avail = f2c_inquire_avail(datatype)

      return
      end

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      integer function MA_inquire_heap (datatype)

      implicit none

      integer datatype

#include "f2c.h"

      MA_inquire_heap = f2c_inquire_heap(datatype)

      return
      end

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      integer function MA_inquire_stack (datatype)

      implicit none

      integer datatype

#include "f2c.h"

      MA_inquire_stack = f2c_inquire_stack(datatype)

      return
      end

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      logical function MA_pop_stack (memhandle)

      implicit none

      integer memhandle

#include "f2c.h"

      if (f2c_pop_stack(memhandle) .eq. MA_TRUE) then
          MA_pop_stack = .true.
      else
          MA_pop_stack = .false.
      endif

      return
      end

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      subroutine MA_print_stats

      implicit none

#include "f2c.h"

      call f2c_print_stats()

      return
      end

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      logical function MA_push_get (datatype, nelem, name, memhandle,
     $    index)

      implicit none

      integer datatype
      integer nelem
      character*(*) name
      integer memhandle
      integer index

#include "f2c.h"

      if (f2c_push_get(datatype, nelem, name, memhandle, index) .eq.
     $    MA_TRUE) then
          MA_push_get = .true.
      else
          MA_push_get = .false.
      endif

      return
      end

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      logical function MA_push_stack (datatype, nelem, name,
     $    memhandle)

      implicit none

      integer datatype
      integer nelem
      character*(*) name
      integer memhandle

#include "f2c.h"

      if (f2c_push_stack(datatype, nelem, name, memhandle) .eq.
     $    MA_TRUE) then
          MA_push_stack = .true.
      else
          MA_push_stack = .false.
      endif

      return
      end

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      logical function MA_set_auto_verify (value)

      implicit none

      logical value
      integer ivalue

#include "f2c.h"

      if (value) then
          ivalue = MA_TRUE
      else
          ivalue = MA_FALSE
      endif

      if (f2c_set_auto_verify(ivalue) .eq. MA_TRUE) then
          MA_set_auto_verify = .true.
      else
          MA_set_auto_verify = .false.
      endif

      return
      end

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      logical function MA_set_error_print (value)

      implicit none

      logical value
      integer ivalue

#include "f2c.h"

      if (value) then
          ivalue = MA_TRUE
      else
          ivalue = MA_FALSE
      endif

      if (f2c_set_error_print(ivalue) .eq. MA_TRUE) then
          MA_set_error_print = .true.
      else
          MA_set_error_print = .false.
      endif

      return
      end

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      logical function MA_set_hard_fail (value)

      implicit none

      logical value
      integer ivalue

#include "f2c.h"

      if (value) then
          ivalue = MA_TRUE
      else
          ivalue = MA_FALSE
      endif

      if (f2c_set_hard_fail(ivalue) .eq. MA_TRUE) then
          MA_set_hard_fail = .true.
      else
          MA_set_hard_fail = .false.
      endif

      return
      end

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      integer function MA_sizeof (datatype1, nelem1, datatype2)

      implicit none

      integer datatype1
      integer nelem1
      integer datatype2

#include "f2c.h"

      MA_sizeof = f2c_sizeof(datatype1, nelem1, datatype2)

      return
      end

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      integer function MA_sizeof_overhead (datatype)

      implicit none

      integer datatype

#include "f2c.h"

      MA_sizeof_overhead = f2c_sizeof_overhead(datatype)

      return
      end

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      subroutine MA_summarize_allocated_blocks

      implicit none

#include "f2c.h"

      call f2c_summarize_allocated_blocks()

      return
      end

c     --------------------------------------------------------------- c
c     --------------------------------------------------------------- c

      logical function MA_verify_allocator_stuff ()

      implicit none

#include "f2c.h"

      if (f2c_verify_allocator_stuff() .eq. MA_TRUE) then
          MA_verify_allocator_stuff = .true.
      else
          MA_verify_allocator_stuff = .false.
      endif

      return
      end

#undef MAF_INTERNAL
