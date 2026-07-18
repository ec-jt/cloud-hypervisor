// Copyright © 2025 Demi Marie Obenour
//
// SPDX-License-Identifier: Apache-2.0 OR BSD-3-Clause

//! Helpers for `mmap()`

use core::ffi::c_int;
use core::ptr::null_mut;
use std::io::{Error, ErrorKind};
use std::os::fd::{AsRawFd as _, BorrowedFd};

use libc::size_t;
use log::{info, warn};

const SZ_2M: u64 = 2 << 20;
const SZ_1G: u64 = 1 << 30;

/// A region of `mmap()`-allocated memory that calls `munmap()` when dropped.
/// This guarantees that the buffer is valid and that its address space
/// will be reserved.  The address space is not guaranteed to be accessible.
/// Atomic access to the data will not cause undefined behavior but might
/// cause SIGSEGV or SIGBUS.  Non-atomic access will generally cause data
/// races and thus Undefined Behavior.
#[derive(Debug)]
pub struct MmapRegion {
    addr: *mut u8,
    len: size_t,
}

impl Drop for MmapRegion {
    fn drop(&mut self) {
        // SAFETY: guaranteed by type validity invariant
        unsafe { assert_eq!(libc::munmap(self.addr.cast(), self.len), 0) }
    }
}
// SAFETY: the caller is responsible for avoiding data races
unsafe impl Send for MmapRegion {}
// SAFETY: the caller is responsible for avoiding data races
unsafe impl Sync for MmapRegion {}

impl MmapRegion {
    #[inline]
    pub fn addr(&self) -> *mut u8 {
        self.addr
    }

    /// Return the length of the region.
    /// This function promises that the return value fits in [`libc::size_t`]
    /// and in [`isize`] and `unsafe` code can rely on this.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Alignment used for huge mappings so kernel huge_fault handlers
    /// (e.g. vfio-pci's PUD/PMD pfnmap inserts) can use large page
    /// table entries.  A 32GB GPU BAR1 mapped with 1GB PUD entries is
    /// pinned by iommufd in 32 page-table walks instead of 8.4M —
    /// measured 4.03s -> ms-range per device on RTX 5090 (with the
    /// matching iommufd addr_mask batching in the host kernel).
    fn alignment_for_len(len: u64) -> u64 {
        if len >= SZ_1G {
            SZ_1G
        } else if len >= SZ_2M {
            SZ_2M
        } else {
            0
        }
    }

    /// Create an [`MmapRegion`] using `mmap` of a file descriptor.
    ///
    /// Mappings of 2MB/1GB or larger are aligned to the corresponding
    /// boundary via an anonymous PROT_NONE reservation + MAP_FIXED
    /// replacement, so device pfnmap faults can insert PMD/PUD-level
    /// entries (the file offset of VFIO BAR regions is naturally
    /// aligned: physical BAR start alignment equals BAR size).
    pub fn mmap(
        len: u64,
        prot: c_int,
        fd: BorrowedFd,
        offset1: u64,
        offset2: u64,
    ) -> std::io::Result<Self> {
        const BAD_LENGTH: &str = "Offsets must fit in libc::off_t";
        const BAD_OFFSET: &str = "Mapping length must fit \
in both isize and libc::size_t";
        let Some(offset) = offset1.checked_add(offset2) else {
            return Err(Error::new(ErrorKind::InvalidInput, BAD_OFFSET));
        };
        let Ok(offset) = libc::off_t::try_from(offset) else {
            return Err(Error::new(ErrorKind::InvalidInput, BAD_OFFSET));
        };
        if isize::try_from(len).is_err() {
            return Err(Error::new(ErrorKind::InvalidInput, BAD_LENGTH));
        }
        let align = Self::alignment_for_len(len);
        let Ok(len) = libc::size_t::try_from(len) else {
            return Err(Error::new(ErrorKind::InvalidInput, BAD_LENGTH));
        };

        assert!(
            (prot & !(libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC)) == 0,
            "bad protection"
        );
        let flags = libc::MAP_SHARED;

        if align > 0 && (offset as u64) % align != 0 {
            info!(
                "[MMAP_ALIGN] skipping aligned path: offset 0x{offset:x} not \
                 aligned to 0x{align:x} (len=0x{len:x}) — huge pfnmap inserts \
                 unavailable for this region"
            );
        }
        // Aligned path: reserve [len + align) of address space, then
        // MAP_FIXED the real mapping at the aligned address inside the
        // reservation (atomically replaces), then trim head/tail.
        if align > 0 && (offset as u64) % align == 0 {
            let align_sz = align as size_t;
            let Some(reserve_len) = len.checked_add(align_sz) else {
                return Err(Error::new(ErrorKind::InvalidInput, BAD_LENGTH));
            };
            // SAFETY: FFI call with correct parameters.
            let reservation = unsafe {
                libc::mmap(
                    null_mut(),
                    reserve_len,
                    libc::PROT_NONE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                    -1,
                    0,
                )
            };
            if reservation != libc::MAP_FAILED {
                let base = reservation as usize;
                let aligned = base.div_ceil(align_sz) * align_sz;
                let head = aligned - base;
                let tail = reserve_len - head - len;
                // SAFETY: MAP_FIXED inside our own fresh reservation —
                // atomically replaces the PROT_NONE pages, no race with
                // other threads' mmaps.
                let addr = unsafe {
                    libc::mmap(
                        aligned as *mut libc::c_void,
                        len,
                        prot,
                        flags | libc::MAP_FIXED,
                        fd.as_raw_fd(),
                        offset,
                    )
                };
                if addr != libc::MAP_FAILED {
                    debug_assert_eq!(addr as usize, aligned);
                    // Trim the unused reservation head/tail.
                    if head > 0 {
                        // SAFETY: unmapping our own reservation prefix.
                        unsafe { libc::munmap(reservation, head) };
                    }
                    if tail > 0 {
                        // SAFETY: unmapping our own reservation suffix.
                        unsafe {
                            libc::munmap((aligned + len) as *mut libc::c_void, tail)
                        };
                    }
                    // MADV_HUGEPAGE sets VM_HUGEPAGE on the VMA, which
                    // the THP order gate accepts under the default
                    // "madvise" global policy — without it, device
                    // huge_fault handlers (vfio-pci PUD/PMD pfnmap
                    // inserts) only fire when THP is globally "always".
                    // Best-effort: EINVAL/EBADF just means 4KB faults.
                    // SAFETY: madvise on our own fresh mapping.
                    let madv = unsafe {
                        libc::madvise(addr, len, libc::MADV_HUGEPAGE)
                    };
                    info!(
                        "[MMAP_ALIGN] aligned mmap OK: va=0x{aligned:x} \
                         len=0x{len:x} align=0x{align:x} madv_hugepage={}",
                        if madv == 0 {
                            "ok".to_string()
                        } else {
                            format!("err={}", Error::last_os_error())
                        }
                    );
                    return Ok(Self {
                        addr: addr.cast(),
                        len,
                    });
                }
                // MAP_FIXED failed — drop the reservation and fall
                // through to the unaligned path.
                warn!(
                    "[MMAP_ALIGN] MAP_FIXED failed ({}) — falling back to \
                     UNALIGNED mmap (len=0x{len:x}, align=0x{align:x}); huge \
                     pfnmap inserts will not apply, expect slow IOAS map",
                    Error::last_os_error()
                );
                // SAFETY: unmapping our own reservation.
                unsafe { libc::munmap(reservation, reserve_len) };
            } else {
                // Reservation failed (address space pressure) — fall back.
                warn!(
                    "[MMAP_ALIGN] reservation of 0x{reserve_len:x} failed ({}) \
                     — falling back to UNALIGNED mmap; huge pfnmap inserts \
                     will not apply, expect slow IOAS map",
                    Error::last_os_error()
                );
            }
        }

        // SAFETY: FFI call with correct parameters.
        let addr = unsafe { libc::mmap(null_mut(), len, prot, flags, fd.as_raw_fd(), offset) };
        if addr == libc::MAP_FAILED {
            Err(Error::last_os_error())
        } else {
            let addr = addr.cast();
            Ok(Self { addr, len })
        }
    }
}
