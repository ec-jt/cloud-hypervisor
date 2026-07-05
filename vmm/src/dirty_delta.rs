// Copyright © 2026 dc-danus
//
// SPDX-License-Identifier: Apache-2.0
//
//! Incremental memory checkpoint support (dirty-delta tracking).
//!
//! Port of the Firecracker fork's dirty tracking endpoints (branch
//! `feature/dirty-tracking`, commit `2fa9683e1`) to Cloud Hypervisor:
//!
//! - `PUT /memory/delta-hashes/init` — mmap the golden memory file,
//!   compute xxh3_64 per 4KB block, keep golden mmap alive for XOR.
//! - `GET /memory/dirty-delta` — xxHash-dedup dirty bitmap (JSON).
//! - `GET /memory/dirty-delta-packed[/keyframe]` — XOR'd packed_v2 lz4
//!   blob export of truly-changed 4KB blocks (binary).
//!
//! Unlike the FC fork (which uses UFFD write-protect pagemap bits as the
//! dirty page source), this port uses the hypervisor dirty log exposed by
//! `MemoryManager::dirty_log()` (Migratable trait). The xxHash dedup layer
//! on top makes the two sources equivalent for export purposes: any page
//! reported dirty whose content hash is unchanged is skipped.
//!
//! Block index semantics: `block_idx = gpa / 4096`, and the golden memory
//! file is assumed to be laid out so that file offset == gpa. This holds
//! for single-RAM-region VMs (guest RAM < 3GiB on x86_64), which is the
//! only configuration used by the sandbox pool (512MiB-1GiB VMs). Blocks
//! whose index is beyond the golden file are skipped, mirroring the FC
//! fork behavior.

use std::collections::HashMap;
use std::os::unix::io::AsRawFd;

use log::info;
use vm_memory::{GuestAddress, GuestMemory};
use vm_migration::protocol::MemoryRangeTable;

/// Export block size — always 4KB, matching the FC fork and the Python
/// `block_cache.py` consumer.
pub const BLOCK_SIZE: usize = 4096;

/// Cap on blocks carried over for P-frame XOR (128MB). If a checkpoint
/// exceeds this, the carry-over is cleared and the next checkpoint XORs
/// everything against golden (I-frame blocks).
const MAX_PREV_BLOCKS: usize = 32768;

/// Read-only mmap of the golden memory file, kept alive for XOR delta
/// compression in `dirty_delta_packed()`.
struct GoldenMmap {
    ptr: *const u8,
    len: usize,
}

// SAFETY: `ptr` is a read-only (PROT_READ, MAP_SHARED) mmap pointer that
// is only dereferenced while the VM is paused, from the VMM API thread.
// The mapping stays valid until Drop.
unsafe impl Send for GoldenMmap {}

impl Drop for GoldenMmap {
    fn drop(&mut self) {
        // SAFETY: ptr/len were set by a successful mmap in
        // init_delta_hashes(). munmap of a valid mapping.
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

impl GoldenMmap {
    fn as_slice(&self) -> &[u8] {
        // SAFETY: valid mmap for the whole lifetime of self.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// JSON response for `GET /drives/{id}/dirty[/reset]`.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct DriveDirtyResponse {
    /// Bitmap for dirty blocks. Each bit represents one `block_size` block.
    pub bitmap: Vec<u64>,
    /// Block size in bytes (4096).
    pub block_size: u64,
    /// Total number of blocks on the disk.
    pub total_blocks: u64,
    /// Number of dirty blocks.
    pub dirty_count: u64,
}

/// XOR'd packed_v2 lz4 blob for dirty delta memory export.
pub struct DirtyDeltaPacked {
    /// lz4 frame-compressed packed_v2 blob (XOR'd blocks).
    pub blob: Vec<u8>,
    /// Number of dirty blocks in the blob.
    pub block_count: u32,
    /// Uncompressed size of the packed_v2 blob in bytes.
    pub raw_size: u64,
    /// True if this is a P-frame (some blocks XOR'd against the previous
    /// checkpoint). False for a pure I-frame (XOR'd against golden only).
    pub is_p_frame: bool,
}

/// Per-VM incremental checkpoint state.
#[derive(Default)]
pub struct DeltaTracker {
    /// xxh3_64 per 4KB block, initialized from the golden memory file and
    /// updated on every export so only newly-changed blocks are exported.
    delta_hashes: Vec<u64>,
    /// Golden memory mmap kept alive for I-frame XOR.
    golden_base: Option<GoldenMmap>,
    /// Raw content of blocks exported by the previous checkpoint, used as
    /// the XOR base for P-frames (20-100x smaller lz4 output).
    prev_checkpoint_blocks: HashMap<u32, Vec<u8>>,
}

impl DeltaTracker {
    /// Whether `init_delta_hashes()` has been called.
    pub fn is_initialized(&self) -> bool {
        !self.delta_hashes.is_empty()
    }

    /// Clear the previous-checkpoint carry-over so the next export is a
    /// pure I-frame. Returns the number of blocks cleared.
    pub fn clear_prev_blocks(&mut self) -> usize {
        let n = self.prev_checkpoint_blocks.len();
        self.prev_checkpoint_blocks.clear();
        n
    }

    /// Re-compute stored hashes from CURRENT guest memory for the
    /// blocks marked in a bitmap file (1 bit per 4KB block, LSB-first
    /// within each byte — same layout as the sparse-overlay `.bitmap`
    /// companion the Python restore path writes).
    ///
    /// Used after a version restore where guest memory is golden +
    /// injected session blocks: hashes computed from the golden FILE
    /// would be stale for the injected blocks, so a later guest write
    /// that reverts an injected block to exact golden content would
    /// hash equal to the stored value and be silently skipped from
    /// exports. The restore chain then layers the stale injected
    /// content under a vmstate that expects golden content (observed as
    /// guest page-allocator list corruption after resume).
    ///
    /// Reads demand-fault through UFFD — call while the external
    /// handler is serving (post vm.restore, pre vm.resume). Only the
    /// marked blocks (KB-to-MB scale) are touched, so guest RSS stays
    /// demand-paged.
    ///
    /// Requires `init_delta_hashes()` first (golden mmap + hash count).
    pub fn rehash_from_guest_bitmap<M: GuestMemory>(
        &mut self,
        mem: &M,
        bitmap_path: &str,
    ) -> Result<usize, String> {
        use xxhash_rust::xxh3::xxh3_64;

        if self.delta_hashes.is_empty() {
            return Err(
                "delta_hashes not initialized - call init_delta_hashes first".to_string()
            );
        }

        let bitmap = std::fs::read(bitmap_path)
            .map_err(|e| format!("read refresh bitmap {bitmap_path}: {e}"))?;

        let num_blocks = self.delta_hashes.len();
        let mut rehashed = 0usize;
        for (byte_idx, byte) in bitmap.iter().enumerate() {
            if *byte == 0 {
                continue;
            }
            for bit in 0..8 {
                if byte & (1 << bit) == 0 {
                    continue;
                }
                let block_idx = byte_idx * 8 + bit;
                if block_idx >= num_blocks {
                    continue;
                }
                let gpa = (block_idx * BLOCK_SIZE) as u64;
                let Ok(host) = mem.get_host_address(GuestAddress(gpa)) else {
                    continue;
                };
                // SAFETY: host points to a valid mapped guest page
                // (fault serviced by the UFFD handler on first access);
                // the VM is paused so the content is stable.
                let block =
                    unsafe { std::slice::from_raw_parts(host as *const u8, BLOCK_SIZE) };
                self.delta_hashes[block_idx] = xxh3_64(block);
                rehashed += 1;
            }
        }

        info!(
            "rehash_from_guest_bitmap: refreshed {rehashed} hashes from live guest memory \
             ({bitmap_path})"
        );
        Ok(rehashed)
    }

    /// Initialize xxh3 delta hashes from the golden memory file.
    ///
    /// Mmaps the golden file read-only, computes xxh3_64 for each 4KB
    /// block, stores the hashes, and keeps the mmap alive for XOR.
    /// Returns the number of hashed blocks.
    pub fn init_delta_hashes(&mut self, golden_path: &str) -> Result<usize, String> {
        use xxhash_rust::xxh3::xxh3_64;

        let file =
            std::fs::File::open(golden_path).map_err(|e| format!("open golden: {e}"))?;
        let len = file
            .metadata()
            .map_err(|e| format!("stat golden: {e}"))?
            .len() as usize;
        if len < BLOCK_SIZE {
            return Err(format!("golden file too small: {len} bytes"));
        }

        // SAFETY: standard read-only file mmap; checked for MAP_FAILED.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED | libc::MAP_POPULATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err("mmap golden failed".to_string());
        }

        let golden = GoldenMmap {
            ptr: ptr as *const u8,
            len,
        };
        let data = golden.as_slice();
        let num_blocks = len / BLOCK_SIZE;
        let mut hashes = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            hashes.push(xxh3_64(&data[i * BLOCK_SIZE..(i + 1) * BLOCK_SIZE]));
        }

        info!(
            "init_delta_hashes: computed {} xxh3 hashes from {} ({}MB), golden mmap kept alive",
            num_blocks,
            golden_path,
            len / (1024 * 1024)
        );

        // Replaces any previous mmap (e.g. re-init after golden upgrade);
        // the old GoldenMmap is munmap'd on drop.
        self.golden_base = Some(golden);
        self.delta_hashes = hashes;
        self.prev_checkpoint_blocks.clear();
        Ok(num_blocks)
    }

    /// Get dirty delta bitmap: xxHash dedup at 4KB granularity.
    ///
    /// For each dirty page from the hypervisor dirty log, xxh3 the block
    /// and compare against stored hashes. Returns a bitmap (u64 words) of
    /// only truly-changed blocks, spanning the golden block range.
    /// Updates stored hashes for the next call.
    ///
    /// Must be called while the VM is paused.
    pub fn dirty_delta_bitmap<M: GuestMemory>(
        &mut self,
        table: &MemoryRangeTable,
        mem: &M,
    ) -> Result<Vec<u64>, String> {
        use xxhash_rust::xxh3::xxh3_64;

        if self.delta_hashes.is_empty() {
            return Err(
                "delta_hashes not initialized - call init_delta_hashes first".to_string()
            );
        }

        let mut bitmap = vec![0u64; self.delta_hashes.len().div_ceil(64)];
        let delta_hashes = &mut self.delta_hashes;

        scan_blocks(table, mem, |block_idx, block| {
            if block_idx >= delta_hashes.len() {
                return;
            }
            let hash = xxh3_64(block);
            if hash != delta_hashes[block_idx] {
                bitmap[block_idx / 64] |= 1u64 << (block_idx % 64);
                delta_hashes[block_idx] = hash;
            }
        })?;

        Ok(bitmap)
    }

    /// Get dirty delta as an XOR'd packed_v2 lz4 blob.
    ///
    /// Combines dirty log scan + xxHash dedup + XOR + pack + lz4 compress
    /// in a single call. Returns the compressed blob ready for S3 upload.
    ///
    /// **P-frame support**: blocks present in `prev_checkpoint_blocks`
    /// are XOR'd against the previous checkpoint's raw data (P-frame);
    /// all other blocks are XOR'd against golden (I-frame). A per-block
    /// XOR base bitmap in the packed_v2 format records which base was
    /// used so the restore side can decode correctly.
    ///
    /// Must be called while the VM is paused.
    pub fn dirty_delta_packed<M: GuestMemory>(
        &mut self,
        table: &MemoryRangeTable,
        mem: &M,
    ) -> Result<DirtyDeltaPacked, String> {
        use std::io::Write;

        use lz4_flex::frame::FrameEncoder;
        use xxhash_rust::xxh3::xxh3_64;

        if self.delta_hashes.is_empty() {
            return Err(
                "delta_hashes not initialized - call init_delta_hashes first".to_string()
            );
        }
        let golden_data = self
            .golden_base
            .as_ref()
            .ok_or_else(|| "golden_base not available for XOR".to_string())?
            .as_slice();

        // P-frame: if prev_checkpoint_blocks is non-empty, we have a
        // previous checkpoint to XOR against.
        let has_prev = !self.prev_checkpoint_blocks.is_empty();
        let prev_block_count = self.prev_checkpoint_blocks.len();

        let mut indices: Vec<u32> = Vec::new();
        let mut block_data: Vec<u8> = Vec::new();
        // Raw block data (before XOR) collected for the next checkpoint's
        // P-frame XOR base.
        let mut new_prev_blocks: HashMap<u32, Vec<u8>> = HashMap::new();
        let mut p_frame_blocks = 0u32;
        let mut i_frame_blocks = 0u32;
        // Per-block XOR base bitmap: bit=1 -> XOR'd against prev,
        // bit=0 -> golden. Used by the restore side to decode.
        let mut xor_base_bits: Vec<bool> = Vec::new();

        let delta_hashes = &mut self.delta_hashes;
        let prev_blocks = &self.prev_checkpoint_blocks;

        scan_blocks(table, mem, |block_idx, block| {
            if block_idx >= delta_hashes.len() {
                return;
            }
            let hash = xxh3_64(block);
            if hash == delta_hashes[block_idx] {
                return;
            }

            let offset = block_idx * BLOCK_SIZE;
            // Determine XOR base: previous checkpoint (P) or golden (I).
            let xor_base: &[u8] = if let Some(prev_data) = prev_blocks.get(&(block_idx as u32))
            {
                prev_data.as_slice()
            } else if offset + BLOCK_SIZE <= golden_data.len() {
                &golden_data[offset..offset + BLOCK_SIZE]
            } else {
                // Beyond golden range - store raw (XOR with itself's base
                // being the block means output equals input XOR base; here
                // we XOR against the block itself producing zeros would
                // lose data, so XOR against a zero base = raw).
                &[]
            };

            // XOR in 8-byte chunks.
            let mut xored = vec![0u8; BLOCK_SIZE];
            if xor_base.len() == BLOCK_SIZE {
                for i in (0..BLOCK_SIZE).step_by(8) {
                    let d = u64::from_ne_bytes(block[i..i + 8].try_into().unwrap());
                    let b = u64::from_ne_bytes(xor_base[i..i + 8].try_into().unwrap());
                    xored[i..i + 8].copy_from_slice(&(d ^ b).to_ne_bytes());
                }
            } else {
                xored.copy_from_slice(block);
            }

            // NOTE: blocks whose XOR result is all-zeros MUST still be
            // exported (FC fork commit 2fa9683e1). A block exported with
            // a non-golden value in an earlier version can later revert
            // to exact golden/parent content; if that revert is skipped,
            // chain restore layers the STALE earlier value while the
            // vmstate expects golden content - observed as guest kernel
            // list_debug corruption after restore. All-zero blocks are
            // nearly free anyway: lz4 collapses them.

            block_data.extend_from_slice(&xored);
            indices.push(block_idx as u32);
            delta_hashes[block_idx] = hash;

            let used_prev = prev_blocks.contains_key(&(block_idx as u32));
            xor_base_bits.push(used_prev);
            if used_prev {
                p_frame_blocks += 1;
            } else {
                i_frame_blocks += 1;
            }

            // Save RAW block (before XOR) for the next checkpoint.
            new_prev_blocks.insert(block_idx as u32, block.to_vec());
        })?;

        let is_p_frame = has_prev;

        // Pack into packed_v2 format (with XOR base bitmap).
        let block_count = indices.len() as u32;
        let block_size = BLOCK_SIZE as u32;

        let bitmap_bytes = (block_count as usize).div_ceil(8);
        let mut xor_bitmap = vec![0u8; bitmap_bytes];
        for (i, &used_prev) in xor_base_bits.iter().enumerate() {
            if used_prev {
                xor_bitmap[i / 8] |= 1 << (i % 8);
            }
        }

        let raw_size = 20 + bitmap_bytes + (block_count as usize) * 4 + block_data.len();
        let mut raw_blob = Vec::with_capacity(raw_size);

        // Header (20 bytes): magic "FCBK", version u16=2, block_size u32,
        // block_count u32, reserved 6 bytes.
        raw_blob.extend_from_slice(b"FCBK");
        raw_blob.extend_from_slice(&2u16.to_le_bytes());
        raw_blob.extend_from_slice(&block_size.to_le_bytes());
        raw_blob.extend_from_slice(&block_count.to_le_bytes());
        raw_blob.extend_from_slice(&[0u8; 6]);
        // XOR base bitmap (version 2): ceil(block_count/8) bytes.
        raw_blob.extend_from_slice(&xor_bitmap);
        // Index table.
        for idx in &indices {
            raw_blob.extend_from_slice(&idx.to_le_bytes());
        }
        // Block data.
        raw_blob.extend_from_slice(&block_data);

        // lz4 frame compress (compatible with Python lz4.frame.decompress).
        let mut encoder = FrameEncoder::new(Vec::new());
        encoder
            .write_all(&raw_blob)
            .map_err(|e| format!("lz4 compress write: {e}"))?;
        let compressed = encoder
            .finish()
            .map_err(|e| format!("lz4 compress finish: {e}"))?;

        // Update prev_checkpoint_blocks for the next P-frame. If the cap
        // is exceeded, clear so the next checkpoint becomes an I-frame.
        if new_prev_blocks.len() <= MAX_PREV_BLOCKS {
            self.prev_checkpoint_blocks = new_prev_blocks;
        } else {
            info!(
                "dirty_delta_packed: prev_blocks cap exceeded ({} > {}), clearing - next \
                 checkpoint will be I-frame",
                new_prev_blocks.len(),
                MAX_PREV_BLOCKS
            );
            self.prev_checkpoint_blocks.clear();
        }

        info!(
            "dirty_delta_packed: frame={}, {} blocks ({} P-frame + {} I-frame), {}B raw, {}B \
             compressed, prev_blocks: {} -> {}",
            if is_p_frame { "P" } else { "I" },
            block_count,
            p_frame_blocks,
            i_frame_blocks,
            raw_size,
            compressed.len(),
            prev_block_count,
            self.prev_checkpoint_blocks.len()
        );

        Ok(DirtyDeltaPacked {
            blob: compressed,
            block_count,
            raw_size: raw_size as u64,
            is_p_frame,
        })
    }
}

/// Iterate all 4KB blocks in the dirty range table, invoking `f` with the
/// global block index (`gpa / 4096`) and a slice of the block's content.
///
/// Ranges produced by `MemoryManager::dirty_log()` are built with 4KB page
/// size and never cross memory region boundaries, so per-block host
/// address resolution always succeeds within mapped guest RAM.
fn scan_blocks<M: GuestMemory, F: FnMut(usize, &[u8])>(
    table: &MemoryRangeTable,
    mem: &M,
    mut f: F,
) -> Result<(), String> {
    for range in table.regions() {
        let mut offset = 0u64;
        while offset + BLOCK_SIZE as u64 <= range.length {
            let gpa = range.gpa + offset;
            let host = mem
                .get_host_address(GuestAddress(gpa))
                .map_err(|e| format!("get_host_address({gpa:#x}): {e}"))?;
            // SAFETY: host points to a valid mapped guest page; the VM is
            // paused so the content is stable for the duration of the read.
            let block = unsafe { std::slice::from_raw_parts(host as *const u8, BLOCK_SIZE) };
            f((gpa / BLOCK_SIZE as u64) as usize, block);
            offset += BLOCK_SIZE as u64;
        }
    }
    Ok(())
}
