
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use cgmath::Vector3;

use crate::error;
use crate::parse::*;

const SIZE_TEXTURE_INFO: usize = 4*6 + 4*2 + 4*2;
const SIZE_VERTEX: usize = 4 * 3;
const SIZE_EDGE: usize = 2 + 2;
const SIZE_PLANE: usize = 4*3 + 4 + 4;
const SIZE_FACE: usize = 2 + 2 + 4 + 2 + 2 + 4 + 4;
const SIZE_MODEL: usize = (4*3)*3 + 4*4 + 4 + 4 + 4;

pub struct BspFile {
    pub light_maps: Vec<u8>,
    pub textures: Vec<Texture>,
    pub texture_info: Vec<TextureInfo>,
    pub edges: Vec<Edge>,
    pub ledges: Vec<i32>,
    pub planes: Vec<Plane>,
    pub faces: Vec<Face>,
    pub models: Vec<Model>,
}

impl BspFile {
    pub fn parse<R>(r: &mut R) -> error::Result<BspFile>
        where R: Read + Seek,
    {

        let version = r.read_long()?;

        if version != 29 {
            bail!("Unsupported BSP version");
        }

        let _e_entities = Entry::read(r)?;
        let e_planes = Entry::read(r)?;
        let e_wall_textures = Entry::read(r)?;
        let e_vertices = Entry::read(r)?;
        let _e_visibility_list = Entry::read(r)?;
        let _e_nodes = Entry::read(r)?;
        let e_texture_info = Entry::read(r)?;
        let e_faces = Entry::read(r)?;
        let e_light_maps = Entry::read(r)?;
        let _e_clip_nodes = Entry::read(r)?;
        let _e_leaves = Entry::read(r)?;
        let _e_face_list = Entry::read(r)?;
        let e_edges = Entry::read(r)?;
        let e_ledges = Entry::read(r)?;
        let e_models = Entry::read(r)?;

        let mut light_maps = vec![0; e_light_maps.size as usize];
        r.seek(SeekFrom::Start(e_light_maps.offset as u64))?;
        r.read_exact(&mut light_maps)?;

        r.seek(SeekFrom::Start(e_wall_textures.offset as u64))?;
        let textures = Texture::parse_textures(r)?;

        r.seek(SeekFrom::Start(e_texture_info.offset as u64))?;
        let texture_info = TextureInfo::parse(e_texture_info.size as usize / SIZE_TEXTURE_INFO, r)?;

        r.seek(SeekFrom::Start(e_vertices.offset as u64))?;
        let vlen = e_vertices.size as usize / SIZE_VERTEX;
        let mut vertices = Vec::with_capacity(vlen);
        for _ in 0 .. vlen {
            vertices.push(Vector3::new(
                r.read_float()?,
                r.read_float()?,
                r.read_float()?,
            ));
        }

        r.seek(SeekFrom::Start(e_edges.offset as u64))?;
        let edges = Edge::parse(e_edges.size as usize / SIZE_EDGE, vertices, r)?;

        let lelen = e_ledges.size as usize / 4;
        let mut ledges = Vec::with_capacity(lelen);
        r.seek(SeekFrom::Start(e_ledges.offset as u64))?;
        for _ in 0 .. lelen {
            ledges.push(r.read_long()?);
        }

        r.seek(SeekFrom::Start(e_planes.offset as u64))?;
        let planes = Plane::parse(e_planes.size as usize / SIZE_PLANE, r)?;

        r.seek(SeekFrom::Start(e_faces.offset as u64))?;
        let faces = Face::parse(e_faces.size as usize / SIZE_FACE, r)?;

        r.seek(SeekFrom::Start(e_models.offset as u64))?;
        let models = Model::parse(e_models.size as usize / SIZE_MODEL, r)?;

        Ok(BspFile {
            light_maps: light_maps,
            textures: textures,
            texture_info: texture_info,
            edges: edges,
            ledges: ledges,
            planes: planes,
            faces: faces,
            models: models,
        })
    }
}
pub struct Model {
    pub bound: (Vector3<f32>, Vector3<f32>),
    pub origin: Vector3<f32>,
    pub faces: Range<usize>,
}

impl Model {
    pub fn parse<R>(count: usize, r: &mut R) -> error::Result<Vec<Model>>
        where R: Read + Seek,
    {
        let mut models = Vec::with_capacity(count);

        for _ in 0 .. count {
            let bound_min = Vector3::new(
                r.read_float()?,
                r.read_float()?,
                r.read_float()?,
            );
            let bound_max = Vector3::new(
                r.read_float()?,
                r.read_float()?,
                r.read_float()?,
            );
            let origin = Vector3::new(
                r.read_float()?,
                r.read_float()?,
                r.read_float()?,
            );
            let _node_id = [
                r.read_long()?,
                r.read_long()?,
                r.read_long()?,
                r.read_long()?,
            ];
            let _leafs = r.read_long()?;
            let face_start = r.read_long()?;
            let face_number = r.read_long()?;

            models.push(Model {
                bound: (bound_min, bound_max),
                origin: origin,
                faces: face_start as usize .. (face_start as usize + face_number as usize),
            });
        }

        Ok(models)
    }
}

pub struct Face {
    pub plane: usize,
    pub front: bool,
    pub ledges: Range<usize>,
    pub texture_info: usize,
    pub type_light: u8,
    pub base_light: u8,
    pub light: [u8; 2],
    pub light_map: i32,
}

impl Face {
    pub fn parse<R>(count: usize, r: &mut R) -> error::Result<Vec<Face>>
        where R: Read + Seek,
    {
        let mut faces = Vec::with_capacity(count);

        for _ in 0 .. count {
            faces.push(Face {
                plane: r.read_ushort()? as usize,
                front: r.read_ushort()? == 0,
                ledges: {
                    let start = r.read_long()? as usize;
                    start .. (start + r.read_ushort()? as usize)
                },
                texture_info: r.read_ushort()? as usize,
                type_light: r.read_uchar()?,
                base_light: r.read_uchar()?,
                light: [
                    r.read_uchar()?,
                    r.read_uchar()?,
                ],
                light_map: r.read_long()?,
            });
        }

        Ok(faces)
    }
}

pub struct Plane {
    pub normal: Vector3<f32>,
    pub distance: f32,
    pub kind: i32,
}

impl Plane {
    pub fn parse<R>(count: usize, r: &mut R) -> error::Result<Vec<Plane>>
        where R: Read + Seek,
    {
        let mut planes = Vec::with_capacity(count);

        for _ in 0 .. count {
            planes.push(Plane {
                normal: Vector3::new(
                    r.read_float()?,
                    r.read_float()?,
                    r.read_float()?,
                ),
                distance: r.read_float()?,
                kind: r.read_long()?,
            });
        }

        Ok(planes)
    }
}

pub struct Edge(pub Vector3<f32>, pub Vector3<f32>);

impl Edge {
    pub fn parse<R>(count: usize, vertices: Vec<Vector3<f32>>, r: &mut R) -> error::Result<Vec<Edge>>
        where R: Read + Seek,
    {
        let mut edges = Vec::with_capacity(count);
        for _ in 0 .. count {
            edges.push(Edge(
                vertices[r.read_ushort()? as usize],
                vertices[r.read_ushort()? as usize],
            ));
        }
        Ok(edges)
    }
}

pub struct TextureInfo {
    pub vector_s: Vector3<f32>,
    pub dist_s: f32,
    pub vector_t: Vector3<f32>,
    pub dist_t: f32,
    pub texture: usize,
    pub animated: bool,
}

impl TextureInfo {
    pub fn parse<R>(count: usize, r: &mut R) -> error::Result<Vec<TextureInfo>>
        where R: Read + Seek,
    {
        let mut info = Vec::with_capacity(count);

        for _ in 0 .. count {
            let vector_s = Vector3::new(
                r.read_float()?,
                r.read_float()?,
                r.read_float()?,
            );
            let dist_s = r.read_float()?;
            let vector_t = Vector3::new(
                r.read_float()?,
                r.read_float()?,
                r.read_float()?,
            );
            let dist_t = r.read_float()?;
            let texture = r.read_ulong()?;
            let animated = r.read_ulong()? != 0;

            info.push(TextureInfo {
                vector_s: vector_s,
                dist_s: dist_s,
                vector_t: vector_t,
                dist_t: dist_t,
                texture: texture as usize,
                animated: animated,
            })
        }

        Ok(info)
    }
}

#[derive(Debug, Default)]
pub struct Texture {
    pub id: i32,
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub pictures: [Picture; 4],
}

impl Texture {
    pub fn parse_textures<R>(r: &mut R) -> error::Result<Vec<Texture>>
        where R: Read + Seek,
    {
        let base_offset = r.seek(SeekFrom::Current(0))?;
        let count = r.read_long()?;

        let mut textures = Vec::with_capacity(count as usize);
        for id in 0 .. count {
            let offset = r.read_long()?;
            if offset == -1 {
                textures.push(Texture {
                    id: -1,
                    .. Texture::default()
                });
                continue;
            }

            let coffset = r.seek(SeekFrom::Current(0))?;

            r.seek(SeekFrom::Start(base_offset + offset as u64))?;
            let name = read_string!(r, 16);
            let name = from_cstring(&name)?;
            let width = r.read_ulong()?;
            let height = r.read_ulong()?;
            let offsets = [
                r.read_ulong()?,
                r.read_ulong()?,
                r.read_ulong()?,
                r.read_ulong()?,
            ];

            let mut tex = Texture {
                id: id,
                name: name,
                width: width,
                height: height,
                pictures: [
                    Picture::default(),
                    Picture::default(),
                    Picture::default(),
                    Picture::default(),
                ],
            };

            for (i, o) in offsets.into_iter().enumerate() {
                r.seek(SeekFrom::Start(base_offset + offset as u64 + *o as u64))?;
                let w = width >> i;
                let h = height >> i;
                let mut data = vec![0; (w * h) as usize];
                r.read_exact(&mut data)?;
                tex.pictures[i] = Picture {
                    width: w,
                    height: h,
                    data: data,
                };
            }

            textures.push(tex);
            r.seek(SeekFrom::Start(coffset))?;
        }

        Ok(textures)
    }
}

#[derive(Debug, Default)]
pub struct Picture {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

struct Entry {
    offset: i32,
    size: i32,
}

impl Entry {
    fn read<R>(r: &mut R) -> error::Result<Entry>
        where R: Read,
    {
        Ok(Entry {
            offset: r.read_long()?,
            size: r.read_long()?,
        })
    }
}