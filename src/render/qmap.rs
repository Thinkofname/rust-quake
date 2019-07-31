
use std::cmp::Ordering;
use std::collections::HashMap;
use std::mem::size_of;
use cgmath::prelude::*;
use cgmath::Vector3;

use hal::{
    Backend,
    Device,
    PhysicalDevice,
    Surface,
    SwapchainConfig,
    Swapchain,
    Adapter,
    CommandPool,
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
        self,
        Submission,
        family::QueueGroup,
    },
    buffer::{
        self,
    },
    memory::{
        self,
    },
    adapter::{
        self,
    },
};

use crate::error;
use super::atlas;
use crate::bsp;
use super::alloc;
use super::{BufferBundle, ImageBundle};

pub struct QMap<B: Backend> {
    buffer: BufferBundle<B>,
    buffer_count: usize,
    buffer_sky: BufferBundle<B>,
    buffer_sky_count: usize,
    buffer_sky_box: BufferBundle<B>,
    buffer_sky_box_count: usize,

    pub texture: ImageBundle<B>,
    pub texture_light: ImageBundle<B>,

    time_offset: f32,
}

impl <B> QMap<B>
    where B: Backend,
{
    pub fn new(
        b: bsp::BspFile,
        adapter: &mut Adapter<B>,
        device: &B::Device,
        queue: &mut queue::CommandQueue<B, hal::Graphics>,
        command_pool: &mut CommandPool<B, hal::Graphics>,
        allocator: &mut alloc::GPUAlloc<B, impl alloc::RangeAlloc>,
    ) -> error::Result<QMap<B>>
    {
        use std::f32;
        use std::cmp::{min, max};
        let mut atlas = atlas::TextureAtlas::new(
            super::ATLAS_SIZE as i32,
            super::ATLAS_SIZE as i32
        );
        let mut light_atlas = atlas::TextureAtlas::new_padded(
            super::ATLAS_SIZE as i32,
            super::ATLAS_SIZE as i32,
            1
        );

        let mut textures = vec![atlas::Rect::default(); b.textures.len()];
        let mut texture_data = (0..3).map(|v| {
            let size = super::ATLAS_SIZE as usize >> v;
            vec![0u8; size * size]
        }).collect::<Vec<_>>();

        let mut t_list = b.textures.iter()
            .filter(|v| v.id != -1)
            .map(|v| TSortable {
                idx: v.id,
                width: v.width,
                height: v.height,
            })
            .collect::<Vec<_>>();
        t_list.sort();

        for t in t_list {
            let tex = &b.textures[t.idx as usize];
            let rect = atlas.find(tex.width as i32, tex.height as i32).unwrap();
            textures[tex.id as usize] = rect;

            for (mip, pic) in tex.pictures.iter().enumerate().take(3) {
                let target = &mut texture_data[mip];
                for y in 0 .. pic.height {
                    for x in 0 .. pic.width {
                        let idx = (rect.x as usize >> mip) + x as usize
                            + ((rect.y as usize >> mip) + y as usize)
                            * (super::ATLAS_SIZE as usize >> mip);
                        let sidx = x as usize + y as usize * pic.width as usize;
                        target[idx] = pic.data[sidx];
                    }
                }
            }
        }

        let mut lights = Vec::new();

        for model in &b.models {
            for face in &b.faces[model.faces.clone()] {
                let tex_info = &b.texture_info[face.texture_info];
                let tex = &b.textures[tex_info.texture];
                if tex.id == -1 || tex.name == "trigger" {
                    continue;
                }
                if face.light_map == -1 || face.type_light == 0xFF {
                    continue;
                }

                let mut min_s = f32::INFINITY;
                let mut min_t = f32::INFINITY;
                let mut max_s = f32::NEG_INFINITY;
                let mut max_t = f32::NEG_INFINITY;

                for ledge in &b.ledges[face.ledges.clone()] {
                    let vert = if *ledge < 0 {
                        b.edges[(-*ledge) as usize].1
                    } else {
                        b.edges[*ledge as usize].0
                    };

                    let val_s = vert.dot(tex_info.vector_s) + tex_info.dist_s;
                    let val_t = vert.dot(tex_info.vector_t) + tex_info.dist_t;

                    min_s = min_s.min(val_s);
                    max_s = max_s.max(val_s);
                    min_t = min_t.min(val_t);
                    max_t = max_t.max(val_t);
                }

                let light_s = (min_s / 16.0).floor();
                let light_t = (min_t / 16.0).floor();
                let light_sm = (max_s / 16.0).ceil();
                let light_tm = (max_t / 16.0).ceil();

                let width = (light_sm - light_s) + 1.0;
                let height = (light_tm - light_t) + 1.0;

                lights.push(TSortable {
                    idx: face.light_map,
                    width: width as u32,
                    height: height as u32,
                });
            }
        }

        let mut light_map_data = vec![0; (super::ATLAS_SIZE * super::ATLAS_SIZE) as usize];

        lights.sort();
        let lights = lights.into_iter()
            .map(|v| {
                let rect = light_atlas.find(v.width as i32, v.height as i32).unwrap();
                for y in -1.. v.height as i32 + 1 {
                    for x in -1 .. v.width as i32 + 1 {
                        let idx = (rect.x + x) as usize
                            + (rect.y  + y) as usize
                            * (super::ATLAS_SIZE as usize);
                        let y = max(min(y, v.height as i32 - 1), 0);
                        let x = max(min(x, v.width as i32 - 1), 0);
                        let sidx = x as usize + y as usize * v.width as usize;
                        light_map_data[idx] = b.light_maps[v.idx as usize + sidx];
                    }
                }
                (v.idx, rect)
            })
            .collect::<HashMap<_, _>>();

        let mut verts = vec![];
        let mut verts_sky = vec![];
        let mut sky_texture = None;
        let mut sky_min: Vector3<f32> = Vector3::zero();
        let mut sky_max: Vector3<f32> = Vector3::zero();

        for model in &b.models {
            for face in &b.faces[model.faces.clone()] {
                let tex_info = &b.texture_info[face.texture_info];
                let tex = &b.textures[tex_info.texture];
                if tex.id == -1 || tex.name == "trigger" {
                    continue;
                }

                let (buffer, is_sky) = if tex.name.starts_with("sky") {
                    sky_texture = Some(tex.id);
                    (&mut verts_sky, true)
                } else {
                    (&mut verts, false)
                };

                let (base_light, type_light) = match tex.name.chars().next() {
                    Some('+') | Some('*') => (127, 0xFF),
                    _ => (face.base_light, face.type_light),
                };

                let mut center_x = 0.0;
                let mut center_y = 0.0;
                let mut center_z = 0.0;
                let mut ec = 0.0;
                for ledge in &b.ledges[face.ledges.clone()] {
                    let e = &b.edges[ledge.abs() as usize];
                    ec += 2.0;
                    center_x += e.0.x;
                    center_x += e.1.x;
                    center_y += e.0.y;
                    center_y += e.1.y;
                    center_z += e.0.z;
                    center_z += e.1.z;
                }
                center_x /= ec;
                center_y /= ec;
                center_z /= ec;

                let light = if base_light == 255 {
                    0
                } else { face.base_light };

                let mut t_offset_x = 0.0;
                let mut t_offset_y = 0.0;
                let mut light_s = 0.0;
                let mut light_t = 0.0;

                if type_light != 0xFF && face.light_map != -1 {
                    let mut min_s = f32::INFINITY;
                    let mut min_t = f32::INFINITY;

                    for ledge in &b.ledges[face.ledges.clone()] {
                        let vert = if *ledge < 0 {
                            b.edges[(-*ledge) as usize].1
                        } else {
                            b.edges[*ledge as usize].0
                        };

                        let val_s = vert.dot(tex_info.vector_s) + tex_info.dist_s;
                        let val_t = vert.dot(tex_info.vector_t) + tex_info.dist_t;

                        min_s = min_s.min(val_s);
                        min_t = min_t.min(val_t);
                    }

                    light_s = (min_s / 16.0).floor();
                    light_t = (min_t / 16.0).floor();

                    let tex = &lights[&face.light_map];
                    t_offset_x = tex.x as f32;
                    t_offset_y = tex.y as f32;
                }

                let s = tex_info.vector_s;
                let t = tex_info.vector_t;

                let trect = textures[tex.id as usize];

                for ledge in &b.ledges[face.ledges.clone()] {
                    let e = &b.edges[(*ledge).abs() as usize];
                    let (av, bv) = if *ledge >= 0 {
                        (e.1, e.0)
                    } else {
                        (e.0, e.1)
                    };

                    if is_sky {
                        for v in &[av, bv] {
                            sky_min.x = sky_min.x.min(model.origin.x + v.x);
                            sky_min.y = sky_min.y.min(model.origin.y + v.y);
                            sky_min.z = sky_min.z.min(model.origin.z + v.z);
                            sky_max.x = sky_max.x.max(model.origin.x + v.x);
                            sky_max.y = sky_max.y.max(model.origin.y + v.y);
                            sky_max.z = sky_max.z.max(model.origin.z + v.z);
                        }
                    }

                    let a_s = av.dot(s) + tex_info.dist_s;
                    let a_t = av.dot(t) + tex_info.dist_t;

                    let (a_tx, a_ty) = if face.light_map != -1 {
                        (
                            (a_s/16.0).floor() - light_s,
                            (a_t/16.0).floor() - light_t
                        )
                    } else {
                        (-1.0, -1.0)
                    };

                    buffer.push(super::Vertex {
                        position: [
                            model.origin.x + av.x,
                            model.origin.y + av.y,
                            model.origin.z + av.z,
                        ],
                        tex: [trect.x as u16, trect.y as u16],
                        tex_info: [
                            a_s as i16,
                            a_t as i16,
                            tex.width as i16,
                            tex.height as i16,
                        ],
                        light_info: [
                            (t_offset_x + a_tx) as i16,
                            (t_offset_y + a_ty) as i16,
                        ],
                        light: light,
                        light_type: type_light,
                    });

                    let b_s = bv.dot(s) + tex_info.dist_s;
                    let b_t = bv.dot(t) + tex_info.dist_t;

                    let (b_tx, b_ty) = if face.light_map != -1 {
                        (
                            (b_s/16.0).floor() - light_s,
                            (b_t/16.0).floor() - light_t
                        )
                    } else {
                        (-1.0, -1.0)
                    };

                    buffer.push(super::Vertex {
                        position: [
                            model.origin.x + bv.x,
                            model.origin.y + bv.y,
                            model.origin.z + bv.z,
                        ],
                        tex: [trect.x as u16, trect.y as u16],
                        tex_info: [
                            b_s as i16,
                            b_t as i16,
                            tex.width as i16,
                            tex.height as i16,
                        ],
                        light_info: [
                            (t_offset_x + b_tx) as i16,
                            (t_offset_y + b_ty) as i16,
                        ],
                        light: light,
                        light_type: type_light,
                    });

                    let center = Vector3::new(
                        center_x,
                        center_y,
                        center_z
                    );

                    let c_s = center.dot(s) + tex_info.dist_s;
                    let c_t = center.dot(t) + tex_info.dist_t;

                    let (c_tx, c_ty) = if face.light_map != -1 {
                        (
                            (c_s/16.0).floor() - light_s,
                            (c_t/16.0).floor() - light_t
                        )
                    } else {
                        (-1.0, -1.0)
                    };

                    buffer.push(super::Vertex {
                        position: [
                            model.origin.x + center.x,
                            model.origin.y + center.y,
                            model.origin.z + center.z,
                        ],
                        tex: [trect.x as u16, trect.y as u16],
                        tex_info: [
                            c_s as i16,
                            c_t as i16,
                            tex.width as i16,
                            tex.height as i16,
                        ],
                        light_info: [
                            (t_offset_x + c_tx) as i16,
                            (t_offset_y + c_ty) as i16,
                        ],
                        light: light,
                        light_type: type_light,
                    });
                }
            }
        }

        let buffer = unsafe {
            let staging_buffer = BufferBundle::new(
                device,
                allocator,
                (size_of::<super::Vertex>() * verts.len()) as u64,
                buffer::Usage::TRANSFER_SRC,
                memory::Properties::CPU_VISIBLE
            );

            {
                let mut data_target = device.acquire_mapping_writer(staging_buffer.memory.memory(), staging_buffer.memory.range.clone()).unwrap();
                data_target[..verts.len()].copy_from_slice(&verts);
                device.release_mapping_writer(data_target).unwrap();
            }

            let buffer = BufferBundle::new(
                device,
                allocator,
                (size_of::<super::Vertex>() * verts.len()) as u64,
                buffer::Usage::VERTEX | buffer::Usage::TRANSFER_DST,
                memory::Properties::DEVICE_LOCAL
            );

            // Copy from staging to real buffer
            let mut cmd = command_pool.acquire_command_buffer::<command::OneShot>();
            cmd.begin();
            cmd.copy_buffer(&staging_buffer.buffer, &buffer.buffer, Some(command::BufferCopy {
                src: 0,
                dst: 0,
                size: (size_of::<super::Vertex>() * verts.len()) as u64,
            }));
            cmd.finish();

            queue.submit_nosemaphores(Some(&cmd), None);
            queue.wait_idle().unwrap();

            command_pool.free(Some(cmd));
            staging_buffer.destroy(device, allocator);

            buffer
        };

        let buffer_sky = unsafe {
            let staging_buffer = BufferBundle::new(
                device,
                allocator,
                (size_of::<super::Vertex>() * verts_sky.len()) as u64,
                buffer::Usage::TRANSFER_SRC,
                memory::Properties::CPU_VISIBLE
            );

            {
                let mut data_target = device.acquire_mapping_writer(staging_buffer.memory.memory(), staging_buffer.memory.range.clone()).unwrap();
                data_target[..verts_sky.len()].copy_from_slice(&verts_sky);
                device.release_mapping_writer(data_target).unwrap();
            }

            let buffer = BufferBundle::new(
                device,
                allocator,
                (size_of::<super::Vertex>() * verts_sky.len()) as u64,
                buffer::Usage::VERTEX | buffer::Usage::TRANSFER_DST,
                memory::Properties::DEVICE_LOCAL
            );

            // Copy from staging to real buffer
            let mut cmd = command_pool.acquire_command_buffer::<command::OneShot>();
            cmd.begin();
            cmd.copy_buffer(&staging_buffer.buffer, &buffer.buffer, Some(command::BufferCopy {
                src: 0,
                dst: 0,
                size: (size_of::<super::Vertex>() * verts_sky.len()) as u64,
            }));
            cmd.finish();

            queue.submit_nosemaphores(Some(&cmd), None);
            queue.wait_idle().unwrap();

            command_pool.free(Some(cmd));
            staging_buffer.destroy(device, allocator);

            buffer
        };


        let sky_box_verts = sky_texture.map_or_else(Vec::new, |v| Self::gen_sky_box(
            &textures, v, sky_min + Vector3::new(-2000.0, -2000.0, 0.0), sky_max + Vector3::new(2000.0, 2000.0, 0.0),
        ));
        let buffer_sky_box = unsafe {
            let staging_buffer = BufferBundle::new(
                device,
                allocator,
                (size_of::<super::Vertex>() * sky_box_verts.len()) as u64,
                buffer::Usage::TRANSFER_SRC,
                memory::Properties::CPU_VISIBLE
            );

            {
                let mut data_target = device.acquire_mapping_writer(staging_buffer.memory.memory(), staging_buffer.memory.range.clone()).unwrap();
                data_target[..sky_box_verts.len()].copy_from_slice(&sky_box_verts);
                device.release_mapping_writer(data_target).unwrap();
            }

            let buffer = BufferBundle::new(
                device,
                allocator,
                (size_of::<super::Vertex>() * sky_box_verts.len()) as u64,
                buffer::Usage::VERTEX | buffer::Usage::TRANSFER_DST,
                memory::Properties::DEVICE_LOCAL
            );

            // Copy from staging to real buffer
            let mut cmd = command_pool.acquire_command_buffer::<command::OneShot>();
            cmd.begin();
            cmd.copy_buffer(&staging_buffer.buffer, &buffer.buffer, Some(command::BufferCopy {
                src: 0,
                dst: 0,
                size: (size_of::<super::Vertex>() * sky_box_verts.len()) as u64,
            }));
            cmd.finish();

            queue.submit_nosemaphores(Some(&cmd), None);
            queue.wait_idle().unwrap();

            command_pool.free(Some(cmd));
            staging_buffer.destroy(device, allocator);

            buffer
        };

        let (texture, texture_light) = unsafe {
            let texture_light = ImageBundle::new(
                device, allocator, super::ATLAS_SIZE, super::ATLAS_SIZE, 1,
                format::Format::R8Unorm,
                hal::image::Filter::Linear
            );
            let texture = ImageBundle::new(
                device, allocator, super::ATLAS_SIZE, super::ATLAS_SIZE, 1,
                format::Format::R8Unorm,
                hal::image::Filter::Nearest
            );

            let staging_buffer_l = BufferBundle::new(
                device,
                allocator,
                (texture_light.row_pitch * super::ATLAS_SIZE) as u64,
                buffer::Usage::TRANSFER_SRC,
                memory::Properties::CPU_VISIBLE
            );
            let staging_buffer = BufferBundle::new(
                device,
                allocator,
                (texture.row_pitch * super::ATLAS_SIZE) as u64,
                buffer::Usage::TRANSFER_SRC,
                memory::Properties::CPU_VISIBLE
            );

            {
                let mut data_target = device.acquire_mapping_writer(staging_buffer_l.memory.memory(), staging_buffer_l.memory.range.clone()).unwrap();
                for y in 0 .. super::ATLAS_SIZE {
                    let idx = y * super::ATLAS_SIZE;
                    let data = &light_map_data[idx as usize .. (idx + super::ATLAS_SIZE) as usize];
                    let d_idx = y * texture_light.row_pitch;
                    data_target[d_idx as usize..(d_idx + super::ATLAS_SIZE) as usize].copy_from_slice(&data);
                }
                device.release_mapping_writer(data_target).unwrap();
            }
            {
                let mut data_target = device.acquire_mapping_writer(staging_buffer.memory.memory(), staging_buffer.memory.range.clone()).unwrap();
                for y in 0 .. super::ATLAS_SIZE {
                    let idx = y * super::ATLAS_SIZE;
                    let data = &texture_data[0][idx as usize .. (idx + super::ATLAS_SIZE) as usize];
                    let d_idx = y * texture_light.row_pitch;
                    data_target[d_idx as usize..(d_idx + super::ATLAS_SIZE) as usize].copy_from_slice(&data);
                }
                device.release_mapping_writer(data_target).unwrap();
            }

            // Copy from staging to image
            let mut cmd = command_pool.acquire_command_buffer::<command::OneShot>();
            cmd.begin();
            cmd.pipeline_barrier(
                pso::PipelineStage::TOP_OF_PIPE .. pso::PipelineStage::TRANSFER,
                memory::Dependencies::empty(),
                &[
                    memory::Barrier::Image {
                        states: (image::Access::empty(), image::Layout::Undefined)
                            .. (image::Access::TRANSFER_WRITE, image::Layout::TransferDstOptimal),
                        target: &*texture_light.image,
                        families: None,
                        range: image::SubresourceRange {
                            aspects: format::Aspects::COLOR,
                            levels: 0..1,
                            layers: 0..1,
                        },
                    },
                    memory::Barrier::Image {
                        states: (image::Access::empty(), image::Layout::Undefined)
                            .. (image::Access::TRANSFER_WRITE, image::Layout::TransferDstOptimal),
                        target: &*texture.image,
                        families: None,
                        range: image::SubresourceRange {
                            aspects: format::Aspects::COLOR,
                            levels: 0..1,
                            layers: 0..1,
                        },
                    },
                ]
            );
            cmd.copy_buffer_to_image(
                &staging_buffer_l.buffer,
                &texture_light.image,
                image::Layout::TransferDstOptimal,
                &[command::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_width: texture_light.row_pitch / 1,
                    buffer_height: super::ATLAS_SIZE,
                    image_layers: image::SubresourceLayers {
                        aspects: format::Aspects::COLOR,
                        level: 0,
                        layers: 0..1,
                    },
                    image_offset: image::Offset { x: 0, y: 0, z: 0},
                    image_extent: image::Extent {
                        width: super::ATLAS_SIZE,
                        height: super::ATLAS_SIZE,
                        depth: 1,
                    },
                }],
            );
            cmd.copy_buffer_to_image(
                &staging_buffer.buffer,
                &texture.image,
                image::Layout::TransferDstOptimal,
                &[command::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_width: texture.row_pitch / 1,
                    buffer_height: super::ATLAS_SIZE,
                    image_layers: image::SubresourceLayers {
                        aspects: format::Aspects::COLOR,
                        level: 0,
                        layers: 0..1,
                    },
                    image_offset: image::Offset { x: 0, y: 0, z: 0},
                    image_extent: image::Extent {
                        width: super::ATLAS_SIZE,
                        height: super::ATLAS_SIZE,
                        depth: 1,
                    },
                }],
            );
            cmd.pipeline_barrier(
                pso::PipelineStage::TRANSFER .. pso::PipelineStage::FRAGMENT_SHADER,
                memory::Dependencies::empty(),
                &[
                    memory::Barrier::Image {
                        states: (image::Access::TRANSFER_WRITE, image::Layout::TransferDstOptimal)
                            .. (image::Access::SHADER_READ, image::Layout::ShaderReadOnlyOptimal),
                        target: &*texture_light.image,
                        families: None,
                        range: image::SubresourceRange {
                            aspects: format::Aspects::COLOR,
                            levels: 0..1,
                            layers: 0..1,
                        },
                    },
                    memory::Barrier::Image {
                        states: (image::Access::TRANSFER_WRITE, image::Layout::TransferDstOptimal)
                            .. (image::Access::SHADER_READ, image::Layout::ShaderReadOnlyOptimal),
                        target: &*texture.image,
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

            queue.submit_nosemaphores(Some(&cmd), None);
            queue.wait_idle().unwrap();

            command_pool.free(Some(cmd));
            staging_buffer_l.destroy(device, allocator);
            staging_buffer.destroy(device, allocator);

            (texture, texture_light)
        };

        Ok(QMap {
            buffer,
            buffer_count: verts.len(),
            buffer_sky,
            buffer_sky_count: verts_sky.len(),
            buffer_sky_box,
            buffer_sky_box_count: sky_box_verts.len(),
            texture,
            texture_light,

            time_offset: 0.0,
        })
    }

    pub fn draw(
        &mut self,
        delta: f32,
        device: &B::Device,
        layout: &B::PipelineLayout,
        pipeline: &B::GraphicsPipeline,
        depth_pipeline: &B::GraphicsPipeline,
        sky_pipeline: &B::GraphicsPipeline,
        encoder: &mut command::RenderPassInlineEncoder<B>,
    ) -> error::Result<()>
    {
        self.time_offset += delta * 0.0007;

        unsafe {
            encoder.push_graphics_constants(layout, pso::ShaderStageFlags::FRAGMENT, 4*4*4, &[self.time_offset.to_bits()]);
            // Skybox
            encoder.bind_graphics_pipeline(sky_pipeline);
            encoder.bind_vertex_buffers(0, Some((&*self.buffer_sky_box.buffer, 0)));
            encoder.draw(0..self.buffer_sky_box_count as u32, 0..1);

            // Fill the depth buffer with the sky areas.
            // This cuts holes into the level to allow the sky
            // to show as the sky box sometimes covers parts of the
            // level in quake.
            encoder.bind_graphics_pipeline(depth_pipeline);
            encoder.bind_vertex_buffers(0, Some((&*self.buffer_sky.buffer, 0)));
            encoder.draw(0..self.buffer_sky_count as u32, 0..1);

            // Render the level
            encoder.bind_graphics_pipeline(pipeline);
            encoder.bind_vertex_buffers(0, Some((&*self.buffer.buffer, 0)));
            encoder.draw(0..self.buffer_count as u32, 0..1);
        }
        Ok(())
    }

    pub unsafe fn destroy(self, device: &B::Device, allocator: &mut alloc::GPUAlloc<B, impl alloc::RangeAlloc>) {
        self.buffer.destroy(device, allocator);
        self.buffer_sky.destroy(device, allocator);
        self.buffer_sky_box.destroy(device, allocator);

        self.texture.destroy(device, allocator);
        self.texture_light.destroy(device, allocator);
    }

    fn gen_sky_box(textures: &Vec<atlas::Rect>, tex: i32, min: Vector3<f32>, max: Vector3<f32>) -> Vec<super::Vertex> {
        let tex = textures[tex as usize];

        let mut verts = vec![];

        let width = (tex.width / 2) as u16;

        for z in 0 .. 2 {
            let offset = z as f32 * 100.0;
            verts.push(super::Vertex {
                position: [
                    min.x,
                    min.y,
                    max.z + offset
                ],
                tex: [tex.x as u16 + width * z, tex.y as u16],
                tex_info: [
                    0,
                    0,
                    width as i16,
                    tex.height as i16,
                ],
                light_info: [0, 0],
                light: 0,
                light_type: z as u8,
            });
            verts.push(super::Vertex {
                position: [
                    min.x,
                    max.y,
                    max.z + offset
                ],
                tex: [tex.x as u16 + width * z, tex.y as u16],
                tex_info: [
                    0,
                    0,
                    width as i16,
                    tex.height as i16,
                ],
                light_info: [0, 0],
                light: 0,
                light_type: z as u8,
            });
            verts.push(super::Vertex {
                position: [
                    max.x,
                    min.y,
                    max.z + offset
                ],
                tex: [tex.x as u16 + width * z, tex.y as u16],
                tex_info: [
                    0,
                    0,
                    width as i16,
                    tex.height as i16,
                ],
                light_info: [0, 0],
                light: 0,
                light_type: z as u8,
            });

            verts.push(super::Vertex {
                position: [
                    min.x,
                    max.y,
                    max.z + offset
                ],
                tex: [tex.x as u16 + width * z, tex.y as u16],
                tex_info: [
                    0,
                    0,
                    width as i16,
                    tex.height as i16,
                ],
                light_info: [0, 0],
                light: 0,
                light_type: z as u8,
            });
            verts.push(super::Vertex {
                position: [
                    max.x,
                    max.y,
                    max.z + offset
                ],
                tex: [tex.x as u16 + width * z, tex.y as u16],
                tex_info: [
                    0,
                    0,
                    width as i16,
                    tex.height as i16,
                ],
                light_info: [0, 0],
                light: 0,
                light_type: z as u8,
            });
            verts.push(super::Vertex {
                position: [
                    max.x,
                    min.y,
                    max.z + offset
                ],
                tex: [tex.x as u16 + width * z, tex.y as u16],
                tex_info: [
                    0,
                    0,
                    width as i16,
                    tex.height as i16,
                ],
                light_info: [0, 0],
                light: 0,
                light_type: z as u8,
            });
        }

        verts
    }
}

#[derive(PartialEq, Eq)]
struct TSortable {
    idx: i32,
    width: u32,
    height: u32,
}

impl PartialOrd for TSortable {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TSortable {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.width * self.height).cmp(&(other.width * other.height))
    }
}