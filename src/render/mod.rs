
mod atlas;
mod qmap;

use std::rc::Rc;

use pak::PackFile;
use error;
use bsp;
use gfx;
use gfx::texture::{
    SamplerInfo,
    FilterMethod,
    WrapMode,
    Kind as TKind,
    AaMode,
};
use gfx::format;
use gfx::memory;
use gfx::handle::{
    Texture,
    Sampler,
    RenderTargetView,
    DepthStencilView,
    ShaderResourceView,
    Buffer,
};
use gfx::traits::FactoryExt;
use cgmath;

pub type ColorFormat = gfx::format::Rgba8;
pub type DepthFormat = gfx::format::DepthStencil;

const ATLAS_SIZE: u32 = 1024;

gfx_defines!{
    vertex Vertex {
        position: [f32; 3] = "a_position",
        tex: [u16; 2] = "a_tex",
        tex_info: [i16; 4] = "a_texInfo",
        light_info: [i16; 2] = "a_lightInfo",
        light: u8 = "a_light",
        light_type: u8 = "a_lightType",
    }
    constant Transform {
        p_matrix: [[f32; 4]; 4] = "pMat",
        u_matrix: [[f32; 4]; 4] = "uMat",
    }

    pipeline pipe {
        vbuf: gfx::VertexBuffer<Vertex> = (),
        transform: gfx::ConstantBuffer<Transform> = "Transform",

        palette: gfx::TextureSampler<[f32; 4]> = "palette",
        colour_map: gfx::TextureSampler<u32> = "colourMap",
        texture: gfx::TextureSampler<u32> = "textures",
        texture_light: gfx::TextureSampler<f32> = "textureLight",

        out: gfx::RenderTarget<ColorFormat> = "fragColor",
        out_depth: gfx::DepthTarget<DepthFormat> = gfx::preset::depth::LESS_EQUAL_WRITE,
    }
    pipeline pipe_depth {
        vbuf: gfx::VertexBuffer<Vertex> = (),
        transform: gfx::ConstantBuffer<Transform> = "Transform",

        out_depth: gfx::DepthTarget<DepthFormat> = gfx::preset::depth::LESS_EQUAL_WRITE,
    }
    pipeline pipe_sky {
        vbuf: gfx::VertexBuffer<Vertex> = (),
        transform: gfx::ConstantBuffer<Transform> = "Transform",
        time_offset: gfx::Global<f32> = "timeOffset",

        palette: gfx::TextureSampler<[f32; 4]> = "palette",
        colour_map: gfx::TextureSampler<u32> = "colourMap",
        texture: gfx::TextureSampler<u32> = "textures",
        texture_light: gfx::TextureSampler<f32> = "textureLight",

        out: gfx::RenderTarget<ColorFormat> = "fragColor",
        out_depth: gfx::DepthTarget<DepthFormat> = gfx::preset::depth::LESS_EQUAL_WRITE,
    }
}
pub(crate) use self::pipe::{
    Data as PData,
    Meta as PMeta,
};
pub(crate) use self::pipe_depth::{
    Data as PDData,
    Meta as PDMeta,
};
pub(crate) use self::pipe_sky::{
    Data as PSData,
    Meta as PSMeta,
};

pub struct Camera {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub rot_y: cgmath::Rad<f32>,
    pub rot_x: cgmath::Rad<f32>,
}

pub struct Renderer<R: gfx::Resources> {
    pak: Rc<PackFile>,
    level: qmap::QMap<R>,

    pub camera: Camera,

    colour_map: (ShaderResourceView<R, u32>, Sampler<R>),
    palette_map: (ShaderResourceView<R, [f32; 4]>, Sampler<R>),

    pub main_color: RenderTargetView<R, ColorFormat>,
    pub main_depth: DepthStencilView<R, DepthFormat>,
    main_pipeline: gfx::PipelineState<R, pipe::Meta>,
    depth_pipeline: gfx::PipelineState<R, pipe_depth::Meta>,
    sky_pipeline: gfx::PipelineState<R, pipe_sky::Meta>,

    matrix_buffer: Buffer<R, Transform>,
}

impl <R: gfx::Resources> Renderer<R> {
    pub fn new<D, F>(
        pak: Rc<PackFile>, level: bsp::BspFile,
        device: &mut D, factory: &mut F,
        main_color: RenderTargetView<R, ColorFormat>,
        main_depth: DepthStencilView<R, DepthFormat>,
    ) -> error::Result<Renderer<R>>
        where F: gfx::Factory<R>,
              R: gfx::Resources,
              D: gfx::Device,
    {

        let colour_map = pak.file("gfx/colormap.lmp")?;
        let colour_map_sampler = factory.create_sampler(SamplerInfo::new(FilterMethod::Scale, WrapMode::Clamp));
        let (_, colour_map) = factory.create_texture_immutable_u8::<(format::R8, format::Uint)>(
            TKind::D2(256 as u16, 64 as u16, AaMode::Single),
            gfx::texture::Mipmap::Provided,
            &[&colour_map],
        )?;

        let palette_map = pak.file("gfx/palette.lmp")?;
        // Gfx doesn't seem to support RGB? Has everything else though
        let mut pm = Vec::with_capacity((palette_map.len() / 3) * 4);
        for data in palette_map.chunks(3) {
            pm.push(data[0]);
            pm.push(data[1]);
            pm.push(data[2]);
            pm.push(255);
        }
        let palette_map_sampler = factory.create_sampler(SamplerInfo::new(FilterMethod::Scale, WrapMode::Clamp));
        let (_, palette_map) = factory.create_texture_immutable_u8::<format::Srgba8>(
            TKind::D2(16 as u16, 16 as u16, AaMode::Single),
            gfx::texture::Mipmap::Provided,
            &[&pm],
        )?;

        let pg = factory.link_program(
            include_bytes!("shader/main_150.glslv"),
            include_bytes!("shader/main_150.glslf"),
        )?;
        let main_pipeline = factory.create_pipeline_from_program(
            &pg,
            gfx::Primitive::TriangleList,
            gfx::state::Rasterizer::new_fill()
                .with_cull_back(),
            pipe::new()
        ).map_err(|v| -> gfx::PipelineStateError<String> { v.into() })?;

        let frag = include_str!("shader/main_150.glslf").replace("COLOR_RENDER", "DEPTH_ONLY");
        let pg_depth = factory.link_program(
            include_bytes!("shader/main_150.glslv"),
            frag.as_bytes(),
        )?;
        let depth_pipeline = factory.create_pipeline_from_program(
            &pg_depth,
            gfx::Primitive::TriangleList,
            gfx::state::Rasterizer::new_fill()
                .with_cull_back(),
            pipe_depth::new()
        ).map_err(|v| -> gfx::PipelineStateError<String> { v.into() })?;

        let pgs = factory.link_program(
            include_bytes!("shader/sky_150.glslv"),
            include_bytes!("shader/sky_150.glslf"),
        )?;
        let sky_pipeline = factory.create_pipeline_from_program(
            &pgs,
            gfx::Primitive::TriangleList,
            gfx::state::Rasterizer::new_fill()
                .with_cull_back(),
            pipe_sky::new()
        ).map_err(|v| -> gfx::PipelineStateError<String> { v.into() })?;

        let level = qmap::QMap::new(level, device, factory)?;

        Ok(Renderer {
            pak: pak,
            level: level,

            camera: Camera {
                x: 504.0,
                y: 401.0,
                z: 75.0,
                rot_y: cgmath::Rad(0.0),
                rot_x: cgmath::Rad(::std::f32::consts::PI),
            },

            colour_map: (colour_map, colour_map_sampler),
            palette_map: (palette_map, palette_map_sampler),

            main_pipeline: main_pipeline,
            depth_pipeline: depth_pipeline,
            sky_pipeline: sky_pipeline,
            main_color: main_color,
            main_depth: main_depth,

            matrix_buffer: factory.create_constant_buffer(1),
        })
    }

    pub fn draw<D, F, C>(&mut self,
        delta: f32,
        width: f64, height: f64,
        device: &mut D, factory: &mut F, encoder: &mut gfx::Encoder<R, C>
    )
        where F: gfx::Factory<R>,
              R: gfx::Resources,
              D: gfx::Device,
              C: gfx::CommandBuffer<R>,
    {
        let p_matrix: cgmath::Matrix4<f32> = cgmath::PerspectiveFov {
            fovy: cgmath::Deg(75.0).into(),
            aspect: width as f32 / height as f32,
            near: 0.1,
            far: 10_000.0,
        }.into();
        let u_matrix = cgmath::Matrix4::from_angle_x(self.camera.rot_x + cgmath::Rad(::std::f32::consts::PI / 2.0))
            * cgmath::Matrix4::from_angle_z(self.camera.rot_y)
            * cgmath::Matrix4::from_translation(
                cgmath::Vector3::new(-self.camera.x, -self.camera.y, -self.camera.z)
            );
        encoder.update_buffer(&self.matrix_buffer, &[Transform {
            p_matrix: p_matrix.into(),
            u_matrix: u_matrix.into(),
        }], 0).unwrap();

        encoder.clear(&self.main_color, [0.0, 0.0, 0.0, 1.0]);
        encoder.clear_depth(&self.main_depth, 1.0);

        self.level.draw(
            delta,
            device, factory, encoder,
            &self.main_pipeline,
            &self.depth_pipeline,
            &self.sky_pipeline,
            &self.matrix_buffer,
            self.colour_map.clone(),
            self.palette_map.clone(),
            self.main_color.clone(),
            self.main_depth.clone(),
        ).unwrap();
    }

    pub fn change_level<D, F>(
        &mut self,
        level: bsp::BspFile,
        device: &mut D, factory: &mut F,
    ) -> error::Result<()>
        where F: gfx::Factory<R>,
              R: gfx::Resources,
              D: gfx::Device,
    {
        let level = qmap::QMap::new(level, device, factory)?;
        self.level = level;
        Ok(())
    }
}