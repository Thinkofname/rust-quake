
mod atlas;
mod qmap;
mod alloc;
mod util;

use util::*;

use std::rc::Rc;
use std::mem::{ManuallyDrop, size_of};
use log::*;

use crate::pak::PackFile;
use crate::error;
use crate::bsp;

use hal::{
    Backend,
    Device,
    PhysicalDevice,
    Surface,
    SwapchainConfig,
    Swapchain,
    Adapter,
    CommandPool,
    DescriptorPool,
    pass::{
        self,
    },
    image::{
        self,
    },
    format::{
        self,
        ChannelType,
        Swizzle,
    },
    pso::{
        self,
        PipelineStage,
        EntryPoint,
        GraphicsShaderSet,
        Rasterizer,
    },
    window::{
        Extent2D,
    },
    pool::{
        self,
    },
    command::{
        self,
        CommandBuffer,
    },
    queue::{
        Submission,
        family::QueueGroup,
    },
};

use cgmath;

const ATLAS_SIZE: u32 = 1024;

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
    tex: [u16; 2],
    tex_info: [i16; 4],
    light_info: [i16; 2],
    light: u8,
    light_type: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Transform {
    p_matrix: cgmath::Matrix4<f32>,
    u_matrix: cgmath::Matrix4<f32>,
}

pub struct Camera {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub rot_y: cgmath::Rad<f32>,
    pub rot_x: cgmath::Rad<f32>,
}

pub struct Renderer<B: Backend> {
    pak: Rc<PackFile>,
    level: ManuallyDrop<qmap::QMap<B>>,

    pub camera: Camera,
    display_size: (u32, u32),
    frame: usize,

    adapter: Adapter<B>,
    pub(crate) surface: B::Surface,
    device: B::Device,
    queue_group: QueueGroup<B, hal::Graphics>,

    gfx: ManuallyDrop<GfxState<B>>,
    recreate_swapchain: bool,
}

struct GfxState<B: Backend> {
    render_pass: B::RenderPass,
    framebuffers: Vec<B::Framebuffer>,
    frame_images: Vec<(B::Image, B::ImageView)>,
    depth_images: Vec<DepthImage<B>>,

    swap_chain: Option<B::Swapchain>,

    allocator: alloc::GPUAlloc<B, alloc::ChunkAlloc>,

    free_acquire_semaphore: B::Semaphore,
    image_acquire_semaphores: Vec<B::Semaphore>,
    submission_complete_fences: Vec<B::Fence>,
    submission_complete_semaphores: Vec<B::Semaphore>,

    cmd_pools: Vec<CommandPool<B, hal::Graphics>>,
    cmd_buffers: Vec<CommandBuffer<B, hal::Graphics, command::MultiShot>>,

    pipeline: B::GraphicsPipeline,
    depth_pipeline: B::GraphicsPipeline,
    sky_pipeline: B::GraphicsPipeline,
    pipeline_layout: B::PipelineLayout,

    descriptor_set_layouts: Vec<B::DescriptorSetLayout>,
    descriptor_pool: B::DescriptorPool,
    descriptor_set: B::DescriptorSet,

    texture_colour_map: ImageBundle<B>,
    texture_palette_map: ImageBundle<B>,
}

impl <B: Backend> Renderer<B> {
    pub fn new(
        pak: Rc<PackFile>, level: bsp::BspFile,
        mut adapter: Adapter<B>,
        mut surface: B::Surface,
        size: (f64, f64),
    ) -> error::Result<Renderer<B>>
    {
        let size = (size.0 as u32, size.1 as u32);

        let (device, mut queue_group) = adapter
            .open_with::<_, hal::Graphics>(1, |family| surface.supports_queue_family(family))
            .unwrap();

        let limits = adapter.physical_device.limits();
        let memory_types = adapter.physical_device.memory_properties().memory_types;
        let mut allocator = unsafe { alloc::GPUAlloc::new(limits, memory_types) };

        let (_caps, formats, _present_modes) = surface.compatibility(&mut adapter.physical_device);
        let format = formats.map_or(format::Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .map(|format| *format)
                .unwrap_or(formats[0])
        });

        let render_pass = {
            let attachment = pass::Attachment {
                format: Some(format),
                samples: 1,
                ops: pass::AttachmentOps::new(
                    pass::AttachmentLoadOp::Clear,
                    pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: pass::AttachmentOps::DONT_CARE,
                layouts: image::Layout::Undefined..image::Layout::Present,
            };
            let attachment_depth = pass::Attachment {
                format: Some(format::Format::D32Sfloat),
                samples: 1,
                ops: pass::AttachmentOps::new(
                    pass::AttachmentLoadOp::Clear,
                    pass::AttachmentStoreOp::DontCare,
                ),
                stencil_ops: pass::AttachmentOps::DONT_CARE,
                layouts: image::Layout::Undefined..image::Layout::DepthStencilAttachmentOptimal,
            };

            let subpass = pass::SubpassDesc {
                colors: &[(0, image::Layout::ColorAttachmentOptimal)],
                depth_stencil: Some(&(1, image::Layout::DepthStencilAttachmentOptimal)),
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            let in_dependency = pass::SubpassDependency {
                passes: pass::SubpassRef::External..pass::SubpassRef::Pass(0),
                stages: PipelineStage::COLOR_ATTACHMENT_OUTPUT
                    .. PipelineStage::COLOR_ATTACHMENT_OUTPUT | PipelineStage::EARLY_FRAGMENT_TESTS,
                accesses: image::Access::empty()
                    ..(
                        image::Access::COLOR_ATTACHMENT_READ | image::Access::COLOR_ATTACHMENT_WRITE
                        | image::Access::DEPTH_STENCIL_ATTACHMENT_READ | image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE
                    ),
            };

            let out_dependency = pass::SubpassDependency {
                passes: pass::SubpassRef::Pass(0) .. pass::SubpassRef::External,
                stages: PipelineStage::COLOR_ATTACHMENT_OUTPUT | PipelineStage::EARLY_FRAGMENT_TESTS
                    .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                accesses: (
                        image::Access::COLOR_ATTACHMENT_READ | image::Access::COLOR_ATTACHMENT_WRITE
                        | image::Access::DEPTH_STENCIL_ATTACHMENT_READ | image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE
                    ) .. image::Access::empty(),
            };

            unsafe { device.create_render_pass(
                &[attachment, attachment_depth],
                &[subpass],
                &[in_dependency, out_dependency]
            ) }
                .expect("Can't create render pass")
        };

        let (swap_chain, framebuffers, frame_images, depth_images) = Self::make_swapchain(
            &mut adapter, &device, &mut allocator, &mut surface, &render_pass, None,
            size.0, size.1,
        );

        let num_framebuffers = framebuffers.len();
        let frames_in_flight = num_framebuffers + 1;
        println!("Frames in flight: {}", frames_in_flight);

        let mut image_acquire_semaphores = Vec::with_capacity(num_framebuffers);
        let free_acquire_semaphore = device
            .create_semaphore()
            .expect("Could not create semaphore");

        let mut submission_complete_semaphores = Vec::with_capacity(frames_in_flight);
        let mut submission_complete_fences = Vec::with_capacity(frames_in_flight);
        let mut cmd_pools = Vec::with_capacity(frames_in_flight);
        let mut cmd_buffers = Vec::with_capacity(frames_in_flight);

        for _ in 0..frames_in_flight {
            unsafe {
                cmd_pools.push(
                    device
                        .create_command_pool_typed(&queue_group, pool::CommandPoolCreateFlags::empty())
                        .expect("Can't create command pool"),
                );
            }
        }

        for _ in 0..num_framebuffers {
            image_acquire_semaphores.push(
                device
                    .create_semaphore()
                    .expect("Could not create semaphore"),
            );
        }

        for i in 0..frames_in_flight {
            submission_complete_semaphores.push(
                device
                    .create_semaphore()
                    .expect("Could not create semaphore"),
            );
            submission_complete_fences.push(
                device
                    .create_fence(true)
                    .expect("Could not create semaphore"),
            );
            cmd_buffers.push(cmd_pools[i].acquire_command_buffer::<command::MultiShot>());
        }

        let (texture_colour_map, texture_palette_map) = unsafe {
            let colour_map = pak.file("gfx/colormap.lmp")?;
            let palette_map = pak.file("gfx/palette.lmp")?;

            let texture_colour_map = ImageBundle::new(
                &device, &mut allocator, 256, 64, 1,
                format::Format::R8Unorm,
                hal::image::Filter::Nearest
            );
            let texture_palette_map = ImageBundle::new(
                &device, &mut allocator, 16, 16, 4,
                format::Format::Rgba8Srgb,
                hal::image::Filter::Nearest
            );
            let mut pm = Vec::with_capacity((palette_map.len() / 3) * 4);
            for data in palette_map.chunks_exact(3) {
                pm.push(data[0]);
                pm.push(data[1]);
                pm.push(data[2]);
                pm.push(255);
            }

            let staging_buffer = BufferBundle::new(
                &device,
                &mut allocator,
                (texture_colour_map.row_pitch * 64) as u64,
                hal::buffer::Usage::TRANSFER_SRC,
                hal::memory::Properties::CPU_VISIBLE
            );
            let staging_buffer_p = BufferBundle::new(
                &device,
                &mut allocator,
                (texture_palette_map.row_pitch * 16) as u64,
                hal::buffer::Usage::TRANSFER_SRC,
                hal::memory::Properties::CPU_VISIBLE
            );

            {
                let mut data_target = device.acquire_mapping_writer(staging_buffer.memory.memory(), staging_buffer.memory.range.clone()).unwrap();
                for y in 0 .. 64 {
                    let idx = y * 256;
                    let data = &colour_map[idx as usize .. (idx + 256) as usize];
                    let d_idx = y * texture_colour_map.row_pitch;
                    data_target[d_idx as usize..(d_idx + 256) as usize].copy_from_slice(&data);
                }
                device.release_mapping_writer(data_target).unwrap();
            }
            {
                let mut data_target = device.acquire_mapping_writer(staging_buffer_p.memory.memory(), staging_buffer_p.memory.range.clone()).unwrap();
                for y in 0 .. 16 {
                    let idx = y * 16 * 4;
                    let data = &pm[idx as usize .. (idx + 16*4) as usize];
                    let d_idx = y * texture_palette_map.row_pitch;
                    data_target[d_idx as usize..(d_idx + 16*4) as usize].copy_from_slice(&data);
                }
                device.release_mapping_writer(data_target).unwrap();
            }

            // Copy from staging to image
            let mut cmd = cmd_pools[0].acquire_command_buffer::<command::OneShot>();
            cmd.begin();
            cmd.pipeline_barrier(
                pso::PipelineStage::TOP_OF_PIPE .. pso::PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                &[
                    hal::memory::Barrier::Image {
                        states: (image::Access::empty(), image::Layout::Undefined)
                            .. (image::Access::TRANSFER_WRITE, image::Layout::TransferDstOptimal),
                        target: &*texture_colour_map.image,
                        families: None,
                        range: image::SubresourceRange {
                            aspects: format::Aspects::COLOR,
                            levels: 0..1,
                            layers: 0..1,
                        },
                    },
                    hal::memory::Barrier::Image {
                        states: (image::Access::empty(), image::Layout::Undefined)
                            .. (image::Access::TRANSFER_WRITE, image::Layout::TransferDstOptimal),
                        target: &*texture_palette_map.image,
                        families: None,
                        range: image::SubresourceRange {
                            aspects: format::Aspects::COLOR,
                            levels: 0..1,
                            layers: 0..1,
                        },
                    }
                ]
            );
            cmd.copy_buffer_to_image(
                &staging_buffer.buffer,
                &texture_colour_map.image,
                image::Layout::TransferDstOptimal,
                &[command::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_width: texture_colour_map.row_pitch / 1,
                    buffer_height: 64,
                    image_layers: image::SubresourceLayers {
                        aspects: format::Aspects::COLOR,
                        level: 0,
                        layers: 0..1,
                    },
                    image_offset: image::Offset { x: 0, y: 0, z: 0},
                    image_extent: image::Extent {
                        width: 256,
                        height: 64,
                        depth: 1,
                    },
                }],
            );
            cmd.copy_buffer_to_image(
                &staging_buffer_p.buffer,
                &texture_palette_map.image,
                image::Layout::TransferDstOptimal,
                &[command::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_width: texture_palette_map.row_pitch / 4,
                    buffer_height: 16,
                    image_layers: image::SubresourceLayers {
                        aspects: format::Aspects::COLOR,
                        level: 0,
                        layers: 0..1,
                    },
                    image_offset: image::Offset { x: 0, y: 0, z: 0},
                    image_extent: image::Extent {
                        width: 16,
                        height: 16,
                        depth: 1,
                    },
                }],
            );
            cmd.pipeline_barrier(
                pso::PipelineStage::TRANSFER .. pso::PipelineStage::FRAGMENT_SHADER,
                hal::memory::Dependencies::empty(),
                &[
                    hal::memory::Barrier::Image {
                        states: (image::Access::TRANSFER_WRITE, image::Layout::TransferDstOptimal)
                            .. (image::Access::SHADER_READ, image::Layout::ShaderReadOnlyOptimal),
                        target: &*texture_colour_map.image,
                        families: None,
                        range: image::SubresourceRange {
                            aspects: format::Aspects::COLOR,
                            levels: 0..1,
                            layers: 0..1,
                        },
                    },
                    hal::memory::Barrier::Image {
                        states: (image::Access::TRANSFER_WRITE, image::Layout::TransferDstOptimal)
                            .. (image::Access::SHADER_READ, image::Layout::ShaderReadOnlyOptimal),
                        target: &*texture_palette_map.image,
                        families: None,
                        range: image::SubresourceRange {
                            aspects: format::Aspects::COLOR,
                            levels: 0..1,
                            layers: 0..1,
                        },
                    },
                ]
            );
            cmd.finish();

            queue_group.queues[0].submit_nosemaphores(Some(&cmd), None);
            queue_group.queues[0].wait_idle().unwrap();

            cmd_pools[0].free(Some(cmd));
            staging_buffer.destroy(&device, &mut allocator);
            staging_buffer_p.destroy(&device, &mut allocator);

            (texture_colour_map, texture_palette_map)
        };


        let level = qmap::QMap::new(level, &mut adapter, &device, &mut queue_group.queues[0], &mut cmd_pools[0], &mut allocator)?;

        let mut compiler = shaderc::Compiler::new().unwrap();
        let vca = compiler
            .compile_into_spirv(include_str!("shader/main.glslv"), shaderc::ShaderKind::Vertex, "main.glslv", "main", None)
            .map_err(|e| {error!("{}", e); e})
            .unwrap();
        let fca = compiler
            .compile_into_spirv(include_str!("shader/main.glslf"), shaderc::ShaderKind::Fragment, "main.glslf", "main", None)
            .map_err(|e| {error!("{}", e); e})
            .unwrap();

        let s_vca = compiler
            .compile_into_spirv(include_str!("shader/sky.glslv"), shaderc::ShaderKind::Vertex, "sky.glslv", "main", None)
            .map_err(|e| {error!("{}", e); e})
            .unwrap();
        let s_fca = compiler
            .compile_into_spirv(include_str!("shader/sky.glslf"), shaderc::ShaderKind::Fragment, "sky.glslf", "main", None)
            .map_err(|e| {error!("{}", e); e})
            .unwrap();

        let vsm = unsafe {
            device.create_shader_module(vca.as_binary_u8())
                .unwrap()
        };
        let fsm = unsafe {
            device.create_shader_module(fca.as_binary_u8())
                .unwrap()
        };
        let s_vsm = unsafe {
            device.create_shader_module(s_vca.as_binary_u8())
                .unwrap()
        };
        let s_fsm = unsafe {
            device.create_shader_module(s_fca.as_binary_u8())
                .unwrap()
        };

        let vs_entry = EntryPoint {
            entry: "main",
            module: &vsm,
            specialization: hal::pso::Specialization::default(),
        };
        let fs_entry = EntryPoint {
            entry: "main",
            module: &fsm,
            specialization: hal::pso::Specialization::default(),
        };
        let shaders = GraphicsShaderSet {
            vertex: vs_entry.clone(),
            hull: None,
            domain: None,
            geometry: None,
            fragment: Some(fs_entry),
        };
        let depth_shaders = GraphicsShaderSet {
            vertex: vs_entry,
            hull: None,
            domain: None,
            geometry: None,
            fragment: None,
        };

        let s_vs_entry = EntryPoint {
            entry: "main",
            module: &s_vsm,
            specialization: hal::pso::Specialization::default(),
        };
        let s_fs_entry = EntryPoint {
            entry: "main",
            module: &s_fsm,
            specialization: hal::pso::Specialization::default(),
        };
        let s_shaders = GraphicsShaderSet {
            vertex: s_vs_entry,
            hull: None,
            domain: None,
            geometry: None,
            fragment: Some(s_fs_entry),
        };

        let vertex_buffers = vec![pso::VertexBufferDesc {
            binding: 0,
            stride: size_of::<Vertex>() as u32,
            rate: pso::VertexInputRate::Vertex,
        }];
        let attributes = vec![
            pso::AttributeDesc {
                location: 0,
                binding: 0,
                element: pso::Element {
                    format: format::Format::Rgb32Sfloat,
                    offset: 0,
                }
            },
            pso::AttributeDesc {
                location: 1,
                binding: 0,
                element: pso::Element {
                    format: format::Format::Rg16Uint,
                    offset: size_of::<[f32; 3]>() as u32,
                }
            },
            pso::AttributeDesc {
                location: 2,
                binding: 0,
                element: pso::Element {
                    format: format::Format::Rgba16Sint,
                    offset: (
                        size_of::<[f32; 3]>()
                        + size_of::<[u16; 2]>()
                    ) as u32,
                }
            },
            pso::AttributeDesc {
                location: 3,
                binding: 0,
                element: pso::Element {
                    format: format::Format::Rg16Sint,
                    offset: (
                        size_of::<[f32; 3]>()
                        + size_of::<[u16; 2]>()
                        + size_of::<[i16; 4]>()
                    ) as u32,
                }
            },
            pso::AttributeDesc {
                location: 4,
                binding: 0,
                element: pso::Element {
                    format: format::Format::R8Uint,
                    offset: (
                        size_of::<[f32; 3]>()
                        + size_of::<[u16; 2]>()
                        + size_of::<[i16; 4]>()
                        + size_of::<[i16; 2]>()
                    ) as u32,
                }
            },
            pso::AttributeDesc {
                location: 5,
                binding: 0,
                element: pso::Element {
                    format: format::Format::R8Uint,
                    offset: (
                        size_of::<[f32; 3]>()
                        + size_of::<[u16; 2]>()
                        + size_of::<[i16; 4]>()
                        + size_of::<[i16; 2]>()
                        + size_of::<u8>()
                    ) as u32,
                }
            },
        ];

        let rasterizer = Rasterizer {
            depth_clamping: false,
            polygon_mode: pso::PolygonMode::Fill,
            cull_face: pso::Face::BACK,
            front_face: pso::FrontFace::CounterClockwise,
            depth_bias: None,
            conservative: false,
        };

        let depth_stencil = pso::DepthStencilDesc {
            depth: pso::DepthTest::On {
                fun: pso::Comparison::LessEqual,
                write: true
            },
            depth_bounds: false,
            stencil: pso::StencilTest::Off,
        };

        let blender = pso::BlendDesc {
            logic_op: Some(pso::LogicOp::Copy),
            targets: vec![pso::ColorBlendDesc(pso::ColorMask::ALL, pso::BlendState::Off)],
        };
        let baked_states = pso::BakedStates::default();

        let descriptor_set_layouts = unsafe { vec![
            device.create_descriptor_set_layout(
                &[
                    pso::DescriptorSetLayoutBinding {
                        binding: 0,
                        ty: pso::DescriptorType::SampledImage,
                        count: 1,
                        stage_flags: pso::ShaderStageFlags::FRAGMENT,
                        immutable_samplers: false,
                    },
                    pso::DescriptorSetLayoutBinding {
                        binding: 1,
                        ty: pso::DescriptorType::Sampler,
                        count: 1,
                        stage_flags: pso::ShaderStageFlags::FRAGMENT,
                        immutable_samplers: false,
                    },
                    pso::DescriptorSetLayoutBinding {
                        binding: 2,
                        ty: pso::DescriptorType::SampledImage,
                        count: 1,
                        stage_flags: pso::ShaderStageFlags::FRAGMENT,
                        immutable_samplers: false,
                    },
                    pso::DescriptorSetLayoutBinding {
                        binding: 3,
                        ty: pso::DescriptorType::Sampler,
                        count: 1,
                        stage_flags: pso::ShaderStageFlags::FRAGMENT,
                        immutable_samplers: false,
                    },
                    pso::DescriptorSetLayoutBinding {
                        binding: 4,
                        ty: pso::DescriptorType::SampledImage,
                        count: 1,
                        stage_flags: pso::ShaderStageFlags::FRAGMENT,
                        immutable_samplers: false,
                    },
                    pso::DescriptorSetLayoutBinding {
                        binding: 5,
                        ty: pso::DescriptorType::Sampler,
                        count: 1,
                        stage_flags: pso::ShaderStageFlags::FRAGMENT,
                        immutable_samplers: false,
                    },
                    pso::DescriptorSetLayoutBinding {
                        binding: 6,
                        ty: pso::DescriptorType::SampledImage,
                        count: 1,
                        stage_flags: pso::ShaderStageFlags::FRAGMENT,
                        immutable_samplers: false,
                    },
                    pso::DescriptorSetLayoutBinding {
                        binding: 7,
                        ty: pso::DescriptorType::Sampler,
                        count: 1,
                        stage_flags: pso::ShaderStageFlags::FRAGMENT,
                        immutable_samplers: false,
                    },
                ],
                Vec::<B::Sampler>::new(),
            ).unwrap(),
        ] };
        let mut descriptor_pool = unsafe {
            device.create_descriptor_pool(
                1,
                &[
                    pso::DescriptorRangeDesc {
                        ty: pso::DescriptorType::SampledImage,
                        count: 4,
                    },
                    pso::DescriptorRangeDesc {
                        ty: pso::DescriptorType::Sampler,
                        count: 4,
                    },
                ],
                pso::DescriptorPoolCreateFlags::empty(),
            ).unwrap()
        };

        let descriptor_set = unsafe {
            descriptor_pool.allocate_set(&descriptor_set_layouts[0]).unwrap()
        };

        unsafe {
            device.write_descriptor_sets(vec![
                pso::DescriptorSetWrite {
                    set: &descriptor_set,
                    binding: 0,
                    array_offset: 0,
                    descriptors: Some(pso::Descriptor::Image(
                        &*texture_colour_map.image_view,
                        image::Layout::ShaderReadOnlyOptimal,
                    )),
                },
                pso::DescriptorSetWrite {
                    set: &descriptor_set,
                    binding: 1,
                    array_offset: 0,
                    descriptors: Some(pso::Descriptor::Sampler(
                        &*texture_colour_map.sampler,
                    )),
                },
                pso::DescriptorSetWrite {
                    set: &descriptor_set,
                    binding: 2,
                    array_offset: 0,
                    descriptors: Some(pso::Descriptor::Image(
                        &*texture_palette_map.image_view,
                        image::Layout::ShaderReadOnlyOptimal,
                    )),
                },
                pso::DescriptorSetWrite {
                    set: &descriptor_set,
                    binding: 3,
                    array_offset: 0,
                    descriptors: Some(pso::Descriptor::Sampler(
                        &*texture_palette_map.sampler,
                    )),
                },
                pso::DescriptorSetWrite {
                    set: &descriptor_set,
                    binding: 4,
                    array_offset: 0,
                    descriptors: Some(pso::Descriptor::Image(
                        &*level.texture_light.image_view,
                        image::Layout::ShaderReadOnlyOptimal,
                    )),
                },
                pso::DescriptorSetWrite {
                    set: &descriptor_set,
                    binding: 5,
                    array_offset: 0,
                    descriptors: Some(pso::Descriptor::Sampler(
                        &*level.texture_light.sampler,
                    )),
                },
                pso::DescriptorSetWrite {
                    set: &descriptor_set,
                    binding: 6,
                    array_offset: 0,
                    descriptors: Some(pso::Descriptor::Image(
                        &*level.texture.image_view,
                        image::Layout::ShaderReadOnlyOptimal,
                    )),
                },
                pso::DescriptorSetWrite {
                    set: &descriptor_set,
                    binding: 7,
                    array_offset: 0,
                    descriptors: Some(pso::Descriptor::Sampler(
                        &*level.texture.sampler,
                    )),
                },
            ])
        }

        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &descriptor_set_layouts,
                &[
                    (pso::ShaderStageFlags::VERTEX, 0..4*4),
                    (pso::ShaderStageFlags::FRAGMENT, 4*4..4*4+4),
                ],
            )
                .unwrap()
        };

        let pipeline = {
            let desc = pso::GraphicsPipelineDesc {
                shaders,
                rasterizer: rasterizer.clone(),
                vertex_buffers: vertex_buffers.clone(),
                attributes: attributes.clone(),
                input_assembler: pso::InputAssemblerDesc::new(hal::Primitive::TriangleList),
                blender: blender.clone(),
                depth_stencil,
                multisampling: None,
                baked_states: baked_states.clone(),
                layout: &pipeline_layout,
                subpass: pass::Subpass {
                    index: 0,
                    main_pass: &render_pass,
                },
                flags: pso::PipelineCreationFlags::empty(),
                parent: pso::BasePipeline::None,
            };

            unsafe {
                device.create_graphics_pipeline(&desc, None)
                    .unwrap()
            }
        };

        let depth_pipeline = {
            let desc = pso::GraphicsPipelineDesc {
                shaders: depth_shaders,
                rasterizer: rasterizer.clone(),
                vertex_buffers: vertex_buffers.clone(),
                attributes: attributes.clone(),
                input_assembler: pso::InputAssemblerDesc::new(hal::Primitive::TriangleList),
                blender: blender.clone(),
                depth_stencil,
                multisampling: None,
                baked_states: baked_states.clone(),
                layout: &pipeline_layout,
                subpass: pass::Subpass {
                    index: 0,
                    main_pass: &render_pass,
                },
                flags: pso::PipelineCreationFlags::empty(),
                parent: pso::BasePipeline::None,
            };

            unsafe {
                device.create_graphics_pipeline(&desc, None)
                    .unwrap()
            }
        };

        let sky_pipeline = {
            let desc = pso::GraphicsPipelineDesc {
                shaders: s_shaders,
                rasterizer,
                vertex_buffers,
                attributes,
                input_assembler: pso::InputAssemblerDesc::new(hal::Primitive::TriangleList),
                blender,
                depth_stencil,
                multisampling: None,
                baked_states,
                layout: &pipeline_layout,
                subpass: pass::Subpass {
                    index: 0,
                    main_pass: &render_pass,
                },
                flags: pso::PipelineCreationFlags::empty(),
                parent: pso::BasePipeline::None,
            };

            unsafe {
                device.create_graphics_pipeline(&desc, None)
                    .unwrap()
            }
        };

        unsafe {
            device.destroy_shader_module(vsm);
            device.destroy_shader_module(fsm);
            device.destroy_shader_module(s_vsm);
            device.destroy_shader_module(s_fsm);
        }

        Ok(Renderer {
            pak: pak,
            level: ManuallyDrop::new(level),
            display_size: size,
            frame: 0,

            camera: Camera {
                x: 504.0,
                y: 401.0,
                z: 75.0,
                rot_y: cgmath::Rad(0.0),
                rot_x: cgmath::Rad(::std::f32::consts::PI),
            },

            adapter,
            surface,
            device,
            queue_group,
            recreate_swapchain: false,

            gfx: ManuallyDrop::new(GfxState {
                allocator,

                render_pass,
                framebuffers,
                frame_images,
                depth_images,
                swap_chain: Some(swap_chain),

                free_acquire_semaphore,
                image_acquire_semaphores,
                submission_complete_fences,
                submission_complete_semaphores,

                cmd_pools,
                cmd_buffers,

                pipeline,
                depth_pipeline,
                sky_pipeline,
                pipeline_layout,

                descriptor_set_layouts,
                descriptor_pool,
                descriptor_set,

                texture_colour_map,
                texture_palette_map,
            }),
        })
    }

    fn make_swapchain(
        adapter: &mut Adapter<B>,
        device: &B::Device,
        allocator: &mut alloc::GPUAlloc<B, impl alloc::RangeAlloc>,
        surface: &mut B::Surface,
        render_pass: &B::RenderPass,
        previous: Option<B::Swapchain>,
        width: u32, height: u32,
    ) -> (
        B::Swapchain,
        Vec<B::Framebuffer>,
        Vec<(B::Image, B::ImageView)>,
        Vec<DepthImage<B>>,
    ){
        let (caps, formats, present_modes) = surface.compatibility(&mut adapter.physical_device);
        println!("New swap chain: {:?}", present_modes);
        let format = formats.map_or(format::Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .map(|format| *format)
                .unwrap_or(formats[0])
        });

        let mut swap_config = SwapchainConfig::from_caps(&caps, format, Extent2D {
            width,
            height,
        });
        // swap_config.present_mode = hal::PresentMode::Immediate;
        let extent = swap_config.extent.to_extent();

        let (swap_chain, images) = unsafe { device.create_swapchain(surface, swap_config, previous) }
            .expect("Can't create swapchain");

        let (frame_images, depth_images, framebuffers) = {
            let pairs = images
                .into_iter()
                .map(|image| unsafe {
                    let rtv = device
                        .create_image_view(
                            &image,
                            image::ViewKind::D2,
                            format,
                            Swizzle::NO,
                            image::SubresourceRange {
                                aspects: format::Aspects::COLOR,
                                levels: 0..1,
                                layers: 0..1,
                            },
                        )
                        .unwrap();
                    (image, rtv)
                })
                .collect::<Vec<_>>();
            let depth_images = pairs
                .iter()
                .map(|_| unsafe { DepthImage::new(device, allocator, width, height) })
                .collect::<Vec<_>>();;
            let fbos = pairs
                .iter()
                .zip(&depth_images)
                .map(|(&(_, ref rtv), ref depth)| unsafe {
                    device
                        .create_framebuffer(render_pass, vec![rtv, &depth.image_view], extent)
                        .unwrap()
                })
                .collect();
            (pairs, depth_images, fbos)
        };
        (swap_chain, framebuffers, frame_images, depth_images)
    }

    pub fn draw(&mut self,
        delta: f32,
        display_size: (u32, u32),
    ) {
        let gfx = &mut *self.gfx;
        if self.display_size != display_size || self.recreate_swapchain {
            self.recreate_swapchain = false;
            self.display_size = display_size;

            self.device.wait_idle().unwrap();
            unsafe {
                for framebuffer in gfx.framebuffers.drain(..) {
                    self.device.destroy_framebuffer(framebuffer);
                }
                for (_, rtv) in gfx.frame_images.drain(..) {
                    self.device.destroy_image_view(rtv);
                }
                for depth in gfx.depth_images.drain(..) {
                    depth.destroy(&self.device, &mut gfx.allocator);
                }
            }
            let (swap_chain, framebuffers, frame_images, depth_images) = Self::make_swapchain(
                &mut self.adapter, &self.device, &mut gfx.allocator, &mut self.surface, &gfx.render_pass,
                gfx.swap_chain.take(),
                display_size.0, display_size.1
            );

            gfx.swap_chain = Some(swap_chain);
            gfx.framebuffers = framebuffers;
            gfx.frame_images = frame_images;
            gfx.depth_images = depth_images;
        }
        let viewport = pso::Viewport {
            rect: pso::Rect {
                x: 0,
                y: 0,
                w: self.display_size.0 as _,
                h: self.display_size.1 as _,
            },
            depth: 0.0..1.0,
        };

        let swap_chain = gfx.swap_chain.as_mut().unwrap();
        let swap_image = unsafe {
            match swap_chain.acquire_image(!0, Some(&gfx.free_acquire_semaphore), None) {
                Ok(i) => i.0 as usize,
                Err(_) => {
                    self.recreate_swapchain = true;
                    return;
                }
            }
        };
        core::mem::swap(
            &mut gfx.free_acquire_semaphore,
            &mut gfx.image_acquire_semaphores[swap_image],
        );

        let frame_idx = self.frame as usize % gfx.submission_complete_fences.len();
        unsafe {
            self.device
                .wait_for_fence(&gfx.submission_complete_fences[frame_idx], !0)
                .expect("Failed to wait for fence");
            self.device
                .reset_fence(&gfx.submission_complete_fences[frame_idx])
                .expect("Failed to reset fence");
            gfx.cmd_pools[frame_idx].reset();
        }

        let cmd_buffer = &mut gfx.cmd_buffers[frame_idx];
        unsafe {
            cmd_buffer.begin(false);
            cmd_buffer.set_viewports(0, &[viewport.clone()]);
            cmd_buffer.set_scissors(0, &[viewport.rect]);

            let p_matrix: cgmath::Matrix4<f32> = cgmath::PerspectiveFov {
                fovy: cgmath::Deg(75.0).into(),
                aspect: self.display_size.0 as f32 / self.display_size.1 as f32,
                near: 0.1,
                far: 10_000.0,
            }.into();
            let u_matrix =
                cgmath::Matrix4::from_nonuniform_scale(1.0, -1.0, 1.0)
                * cgmath::Matrix4::from_angle_x(self.camera.rot_x + cgmath::Rad(::std::f32::consts::PI / 2.0))
                * cgmath::Matrix4::from_angle_z(self.camera.rot_y)
                * cgmath::Matrix4::from_translation(
                    cgmath::Vector3::new(-self.camera.x, -self.camera.y, -self.camera.z)
                );
            let matrix: [[f32; 4]; 4] = (p_matrix * u_matrix).into();

            cmd_buffer.push_graphics_constants(&gfx.pipeline_layout, pso::ShaderStageFlags::VERTEX, 0, hal::memory::cast_slice(&[matrix]));

            {
                let mut encoder = cmd_buffer.begin_render_pass_inline(
                    &gfx.render_pass,
                    &gfx.framebuffers[swap_image],
                    viewport.rect,
                    &[
                        command::ClearValue::Color(command::ClearColor::Float(
                            [0.0, 0.0, 0.0, 1.0]
                        )),
                        command::ClearValue::DepthStencil(command::ClearDepthStencil(1.0, 0)),
                    ],
                );

                encoder.bind_graphics_descriptor_sets(
                    &gfx.pipeline_layout,
                    0,
                    Some(&gfx.descriptor_set),
                    &[],
                );

                self.level.draw(
                    delta,
                    &self.device,
                    &gfx.pipeline_layout,
                    &gfx.pipeline,
                    &gfx.depth_pipeline,
                    &gfx.sky_pipeline,
                    &mut encoder,
                ).unwrap();
            }

            cmd_buffer.finish();

            let submission = Submission {
                command_buffers: Some(&*cmd_buffer),
                wait_semaphores: Some((
                    &gfx.image_acquire_semaphores[swap_image],
                    PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                )),
                signal_semaphores: Some(&gfx.submission_complete_semaphores[frame_idx]),
            };
            self.queue_group.queues[0].submit(submission, Some(&gfx.submission_complete_fences[frame_idx]));

            if let Err(_) = swap_chain.present(
                &mut self.queue_group.queues[0],
                swap_image as hal::SwapImageIndex,
                Some(&gfx.submission_complete_semaphores[frame_idx]),
            ) {
                self.recreate_swapchain = true;
            }
        }

        self.frame = self.frame.wrapping_add(1);
    }

    pub fn change_level(
        &mut self,
        level: bsp::BspFile,
    ) -> error::Result<()>
    {
        use std::ptr;
        unsafe {
            let gfx = &mut *self.gfx;
            self.device.wait_idle().unwrap();
            let old_level = ManuallyDrop::into_inner(ptr::read(&self.level));
            old_level.destroy(&self.device, &mut gfx.allocator);
            let frame_idx = self.frame as usize % gfx.submission_complete_fences.len();
            let level = qmap::QMap::new(
                level,
                &mut self.adapter, &self.device,
                &mut self.queue_group.queues[0],
                &mut gfx.cmd_pools[frame_idx],
                &mut gfx.allocator
            )?;

            self.device.write_descriptor_sets(vec![
                pso::DescriptorSetWrite {
                    set: &gfx.descriptor_set,
                    binding: 4,
                    array_offset: 0,
                    descriptors: Some(pso::Descriptor::Image(
                        &*level.texture_light.image_view,
                        image::Layout::ShaderReadOnlyOptimal,
                    )),
                },
                pso::DescriptorSetWrite {
                    set: &gfx.descriptor_set,
                    binding: 5,
                    array_offset: 0,
                    descriptors: Some(pso::Descriptor::Sampler(
                        &*level.texture_light.sampler,
                    )),
                },
                pso::DescriptorSetWrite {
                    set: &gfx.descriptor_set,
                    binding: 6,
                    array_offset: 0,
                    descriptors: Some(pso::Descriptor::Image(
                        &*level.texture.image_view,
                        image::Layout::ShaderReadOnlyOptimal,
                    )),
                },
                pso::DescriptorSetWrite {
                    set: &gfx.descriptor_set,
                    binding: 7,
                    array_offset: 0,
                    descriptors: Some(pso::Descriptor::Sampler(
                        &*level.texture.sampler,
                    )),
                },
            ]);

            self.level = ManuallyDrop::new(level);
            self.device.wait_idle().unwrap();
            Ok(())
        }
    }
}

impl <B: Backend> Drop for Renderer<B> {
    fn drop(&mut self) {
        use std::ptr;
        self.device.wait_idle().unwrap();
        unsafe {
            let mut gfx = ManuallyDrop::into_inner(ptr::read(&mut self.gfx));
            let level = ManuallyDrop::into_inner(ptr::read(&mut self.level));
            level.destroy(&self.device, &mut gfx.allocator);

            gfx.texture_colour_map.destroy(&self.device, &mut gfx.allocator);
            gfx.texture_palette_map.destroy(&self.device, &mut gfx.allocator);

            self.device.destroy_pipeline_layout(gfx.pipeline_layout);
            self.device.destroy_graphics_pipeline(gfx.pipeline);
            self.device.destroy_graphics_pipeline(gfx.sky_pipeline);
            self.device.destroy_graphics_pipeline(gfx.depth_pipeline);

            self.device.destroy_descriptor_pool(gfx.descriptor_pool);
            for d in gfx.descriptor_set_layouts {
                self.device.destroy_descriptor_set_layout(d);
            }

            self.device.destroy_semaphore(gfx.free_acquire_semaphore);
            for p in gfx.cmd_pools {
                self.device.destroy_command_pool(p.into_raw());
            }
            for s in gfx.image_acquire_semaphores {
                self.device.destroy_semaphore(s);
            }
            for s in gfx.submission_complete_semaphores {
                self.device.destroy_semaphore(s);
            }
            for f in gfx.submission_complete_fences {
                self.device.destroy_fence(f);
            }
            self.device.destroy_render_pass(gfx.render_pass);
            for framebuffer in gfx.framebuffers {
                self.device.destroy_framebuffer(framebuffer);
            }
            for (_, rtv) in gfx.frame_images {
                self.device.destroy_image_view(rtv);
            }
            for depth in gfx.depth_images {
                depth.destroy(&self.device, &mut gfx.allocator);
            }

            gfx.allocator.destroy(&self.device);
            if let Some(swap_chain) = gfx.swap_chain {
                self.device.destroy_swapchain(swap_chain);
            }
        }
    }
}