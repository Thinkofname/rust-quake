
use hal::{
    Backend,
    Device,
    buffer,
    memory,
};

use std::mem::ManuallyDrop;

use super::alloc;

pub struct BufferBundle<B: Backend> {
    pub buffer: ManuallyDrop<B::Buffer>,
    pub memory: ManuallyDrop<alloc::Allocation<B>>,
}

impl <B> BufferBundle<B>
    where B: Backend
{
    pub unsafe fn new(
        device: &B::Device,
        allocator: &mut alloc::GPUAlloc<B, impl alloc::RangeAlloc>,
        size: u64, usage: buffer::Usage, properties: memory::Properties,
    ) -> BufferBundle<B>
    {
        let mut buffer = device.create_buffer(size, usage)
            .unwrap();
        let req = device.get_buffer_requirements(&buffer);
        let memory = allocator.allocate(device, alloc::Type::Buffer, &req, properties);
        device.bind_buffer_memory(memory.memory(), memory.range.start, &mut buffer).unwrap();

        BufferBundle {
            buffer: ManuallyDrop::new(buffer),
            memory: ManuallyDrop::new(memory),
        }
    }

    pub unsafe fn destroy(
        self,
        device: &B::Device,
        allocator: &mut alloc::GPUAlloc<B, impl alloc::RangeAlloc>,
    ) {
        use std::ptr;
        device.destroy_buffer(ManuallyDrop::into_inner(ptr::read(&self.buffer)));
        allocator.free(ManuallyDrop::into_inner(ptr::read(&self.memory)));
    }
}

pub struct ImageBundle<B: Backend> {
    pub image: ManuallyDrop<B::Image>,
    pub image_view: ManuallyDrop<B::ImageView>,
    pub sampler: ManuallyDrop<B::Sampler>,
    pub memory: ManuallyDrop<alloc::Allocation<B>>,

    pub row_pitch: u32,
}

impl <B> ImageBundle<B>
    where B: Backend
{
    pub unsafe fn new(
        device: &B::Device,
        allocator: &mut alloc::GPUAlloc<B, impl alloc::RangeAlloc>,
        width: u32, height: u32,
        pixel_size: u32, format: hal::format::Format,
        filter: hal::image::Filter,
    ) -> ImageBundle<B>
    {
        let row_size = pixel_size * width;
        let row_alignment_mask = allocator.limits.optimal_buffer_copy_pitch_alignment as u32 - 1;
        let row_pitch = (row_size + row_alignment_mask) & !row_alignment_mask;

        let mut image = device.create_image(
            hal::image::Kind::D2(width, height, 1, 1),
            1,
            format,
            hal::image::Tiling::Optimal,
            hal::image::Usage::TRANSFER_DST | hal::image::Usage::SAMPLED,
            hal::image::ViewCapabilities::empty(),
        ).unwrap();

        let req = device.get_image_requirements(&image);
        let memory = allocator.allocate(device, alloc::Type::Image, &req, memory::Properties::DEVICE_LOCAL);
        device.bind_image_memory(&memory.memory(), memory.range.start, &mut image).unwrap();

        let image_view = device.create_image_view(
            &image,
            hal::image::ViewKind::D2,
            format,
            hal::format::Swizzle::NO,
            hal::image::SubresourceRange {
                aspects: hal::format::Aspects::COLOR,
                levels: 0..1,
                layers: 0..1,
            },
        ).unwrap();

        let sampler = device.create_sampler(hal::image::SamplerInfo::new(
            filter,
            hal::image::WrapMode::Clamp,
        )).unwrap();

        ImageBundle {
            image: ManuallyDrop::new(image),
            image_view: ManuallyDrop::new(image_view),
            sampler: ManuallyDrop::new(sampler),
            memory: ManuallyDrop::new(memory),

            row_pitch,
        }
    }

    pub unsafe fn destroy(
        self,
        device: &B::Device,
        allocator: &mut alloc::GPUAlloc<B, impl alloc::RangeAlloc>,
    ) {
        use std::ptr;
        device.destroy_sampler(ManuallyDrop::into_inner(ptr::read(&self.sampler)));
        device.destroy_image_view(ManuallyDrop::into_inner(ptr::read(&self.image_view)));
        device.destroy_image(ManuallyDrop::into_inner(ptr::read(&self.image)));
        allocator.free(ManuallyDrop::into_inner(ptr::read(&self.memory)));
    }
}


pub struct DepthImage<B: Backend> {
    pub image: ManuallyDrop<B::Image>,
    pub image_view: ManuallyDrop<B::ImageView>,
    pub memory: ManuallyDrop<alloc::Allocation<B>>,
}

impl <B> DepthImage<B>
    where B: Backend
{
    pub unsafe fn new(
        device: &B::Device,
        allocator: &mut alloc::GPUAlloc<B, impl alloc::RangeAlloc>,
        width: u32, height: u32,
    ) -> DepthImage<B>
    {
        let mut image = device.create_image(
            hal::image::Kind::D2(width, height, 1, 1),
            1,
            hal::format::Format::D32Sfloat,
            hal::image::Tiling::Optimal,
            hal::image::Usage::DEPTH_STENCIL_ATTACHMENT,
            hal::image::ViewCapabilities::empty(),
        ).unwrap();

        let req = device.get_image_requirements(&image);
        let memory = allocator.allocate(device, alloc::Type::Image, &req, memory::Properties::DEVICE_LOCAL);
        device.bind_image_memory(&memory.memory(), memory.range.start, &mut image).unwrap();

        let image_view = device.create_image_view(
            &image,
            hal::image::ViewKind::D2,
            hal::format::Format::D32Sfloat,
            hal::format::Swizzle::NO,
            hal::image::SubresourceRange {
                aspects: hal::format::Aspects::DEPTH,
                levels: 0..1,
                layers: 0..1,
            },
        ).unwrap();

        DepthImage {
            image: ManuallyDrop::new(image),
            image_view: ManuallyDrop::new(image_view),
            memory: ManuallyDrop::new(memory),
        }
    }

    pub unsafe fn destroy(
        self,
        device: &B::Device,
        allocator: &mut alloc::GPUAlloc<B, impl alloc::RangeAlloc>,
    ) {
        use std::ptr;
        device.destroy_image_view(ManuallyDrop::into_inner(ptr::read(&self.image_view)));
        device.destroy_image(ManuallyDrop::into_inner(ptr::read(&self.image)));
        allocator.free(ManuallyDrop::into_inner(ptr::read(&self.memory)));
    }
}