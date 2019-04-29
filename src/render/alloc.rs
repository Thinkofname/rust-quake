
use std::marker::PhantomData;
use std::ops::Range;
use hal::{
    Device,
    Limits,
    Backend,
    MemoryType,
    adapter,
    memory,
};
use crate::bitset::BitSet;

const REGION_SIZE: u64 = 64 * 1024 * 1024; // 64mb

pub struct GPUAlloc<B: Backend, A: RangeAlloc> {
    pub limits: Limits,
    memory_types: Vec<MemoryType>,
    memory: Vec<GPUMemory<B, A>>,
}

impl <B, A> GPUAlloc<B, A>
    where A: RangeAlloc,
          B: Backend,
{
    pub unsafe fn new(limits: Limits, memory_types: Vec<MemoryType>) -> GPUAlloc<B, A> {
        GPUAlloc {
            limits,
            memory_types,
            memory: Vec::with_capacity(2),
        }
    }

    pub unsafe fn allocate(&mut self, device: &B::Device, ty: Type, requirements: &memory::Requirements, properties: memory::Properties) -> Allocation<B> {
        // Try existing memory
        for m in &mut self.memory {
            if requirements.type_mask & (1 << m.type_id.0) != 0 && m.ty.properties.contains(properties) {
                // Found something, use it
                return m.allocate(&self.limits, device, ty, requirements);
            }
        }

        // New memory
        let memory_type_id = self.memory_types.iter()
                .enumerate()
                .find(|&(id, memory_type)|
                    requirements.type_mask & (1 << id) != 0
                        && memory_type.properties.contains(properties)
                )
                .map(|(id, _)| adapter::MemoryTypeId(id))
                .unwrap();
        let mut memory = GPUMemory {
            id: self.memory.len(),
            type_id: memory_type_id,
            ty: self.memory_types[memory_type_id.0],
            regions: vec![],
        };
        let ret = memory.allocate(&self.limits, device, ty, requirements);
        self.memory.push(memory);
        ret
    }

    pub unsafe fn free(&mut self, alloc: Allocation<B>) {
        let mem = &mut self.memory[alloc.owner];
        mem.free(alloc);
    }

    pub unsafe fn destroy(self, device: &B::Device) {
        for m in self.memory {
            m.destroy(device);
        }
    }
}

pub struct Allocation<B: Backend> {
    owner: usize,
    // TODO: Unsafe lifetime
    memory: &'static B::Memory,
    pub range: Range<u64>,
}

impl <B> Allocation<B>
    where B: Backend
{
    pub fn memory(&self) -> &B::Memory {
        self.memory
    }
}

struct GPUMemory<B: Backend, A: RangeAlloc> {
    id: usize,
    type_id: adapter::MemoryTypeId,
    ty: MemoryType,
    regions: Vec<(A, Box<B::Memory>)>,
}

impl <B, A> GPUMemory<B, A>
    where A: RangeAlloc,
          B: Backend,
{
    unsafe fn allocate(&mut self, limits: &Limits, device: &B::Device, ty: Type, requirements: &memory::Requirements) -> Allocation<B> {
        // Try existing regions
        for (a, mem) in &mut self.regions {
            if let Some(range) = a.allocate(ty, requirements.size, requirements.alignment) {
                // Found
                return Allocation {
                    owner: self.id,
                    memory: &*(&**mem as *const B::Memory),
                    range: range,
                };
            }
        }
        // New region
        let mut region = (
            A::new(REGION_SIZE, limits.buffer_image_granularity),
            Box::new(device.allocate_memory(self.type_id, REGION_SIZE).unwrap())
        );
        let ret = if let Some(range) =  region.0.allocate(ty, requirements.size, requirements.alignment) {
            Some(Allocation {
                owner: self.id,
                memory:  &*(&*region.1 as *const B::Memory),
                range: range,
            })
        } else {
            None
        };
        self.regions.push(region);
        ret.expect("Failed to allocate memory")
    }

    unsafe fn free(&mut self, alloc: Allocation<B>) {
        use std::ptr;
        for (a, mem) in &mut self.regions {
            if ptr::eq(&**mem, alloc.memory) {
                a.free(alloc.range);
                return;
            }
        }
    }

    unsafe fn destroy(self, device: &B::Device) {
        for r in self.regions {
            device.free_memory(*r.1);
        }
    }
}

pub trait RangeAlloc: Sized {
    fn new(size: u64, buffer_image_granularity: u64) -> Self;
    fn allocate(&mut self, ty: Type, size: u64, align: u64) -> Option<Range<u64>>;
    fn free(&mut self, range: Range<u64>);
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Type {
    Buffer,
    Image,
}

pub struct ChunkAlloc {
    chunks: u64,
    used: BitSet,
    used_types: Vec<Option<Type>>,
    buffer_image_granularity: u64,
}

const CHUNK_SIZE: u64 = 8 * 1024; // 8kb

impl RangeAlloc for ChunkAlloc {
    fn new(size: u64, buffer_image_granularity: u64) -> Self {
        let chunks = size / CHUNK_SIZE;
        ChunkAlloc {
            chunks,
            used: BitSet::new(chunks as usize),
            used_types: vec![None; (size / buffer_image_granularity) as usize],
            buffer_image_granularity,
        }
    }
    fn allocate(&mut self, ty: Type, size: u64, align: u64) -> Option<Range<u64>> {
        assert!(CHUNK_SIZE%align == 0);
        let chunks = (size + (CHUNK_SIZE-1))/CHUNK_SIZE;
        let mut idx = 0;
        let skip = (self.buffer_image_granularity + (CHUNK_SIZE-1))/CHUNK_SIZE;
    'search:
        loop {
            // Make sure the value fits within the type region
            let ty_idx = ((idx * CHUNK_SIZE)/self.buffer_image_granularity) as usize;
            let typ = *self.used_types.get(ty_idx)?;
            if typ != None && typ != Some(ty)  {
                idx += skip;
            }
            let ty_idx_end = (((idx + chunks) * CHUNK_SIZE)/self.buffer_image_granularity) as usize;
            let typ = *self.used_types.get(ty_idx_end)?;
            if typ != None && typ != Some(ty)  {
                idx += skip;
            }
            // Make sure the region itself is free
            for off in 0 .. chunks {
                if self.used.get((idx + off) as usize) {
                    idx += 1;
                    continue 'search;
                }
            }
            // Region free, claim it
            for off in 0 .. chunks {
                self.used.set((idx + off) as usize, true);
                let ty_idx = (((idx + off) * CHUNK_SIZE)/self.buffer_image_granularity) as usize;
                self.used_types[ty_idx] = Some(ty);
            }
            break Some(idx * CHUNK_SIZE .. idx * CHUNK_SIZE + size);
        }
    }


    fn free(&mut self, range: Range<u64>) {
        let start = (range.start + (CHUNK_SIZE-1))/CHUNK_SIZE;
        let end = (range.end + (CHUNK_SIZE-1))/CHUNK_SIZE;
        for i in start .. end {
            self.used.set(i as usize, false);
        }
    }
}