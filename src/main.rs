extern crate byteorder;
extern crate cgmath;
#[macro_use]
extern crate error_chain;

#[cfg(feature = "dx11")]
extern crate gfx_backend_dx11 as back;
#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(feature = "gl")]
extern crate gfx_backend_gl as back;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as back;
extern crate gfx_hal as hal;

extern crate winit;
extern crate env_logger;

#[macro_use]
mod parse;
pub mod pak;
pub mod error;
pub mod render;
pub mod bsp;
pub mod bitset;

use std::time::{Instant, Duration};
use std::rc::Rc;
use std::io::Cursor;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;

use hal::{
    Instance,
};
#[cfg(feature = "gl")]
use hal::format::AsFormat as _;

fn main() {
    env_logger::init_from_env("QUAKE_LOG");
    let pak = Rc::new(pak::PackFile::new("id1/PAK0.PAK").unwrap());

    let wb = winit::WindowBuilder::new()
        .with_dimensions(winit::dpi::LogicalSize::new(
            WIDTH as _,
            HEIGHT as _,
        ))
        .with_title("RQuake");

    let mut events_loop = winit::EventsLoop::new();

    #[cfg(not(feature = "gl"))]
    let (window, _instance, mut adapters, surface, size) = {
        let window = wb.build(&events_loop).unwrap();
        let instance = back::Instance::create("RQuake", 1);
        let surface = instance.create_surface(&window);
        let adapters = instance.enumerate_adapters();
        let size = window.get_inner_size().map(Into::into).unwrap_or((WIDTH as f64, HEIGHT as f64));
        (window, instance, adapters, surface, size)
    };
    #[cfg(feature = "gl")]
    let (mut adapters, surface, size) = {
        let window = {
            let builder =
                back::config_context(back::glutin::ContextBuilder::new(), hal::format::Rgba8Srgb::SELF, None)
                    .with_vsync(true);
            builder.build_windowed(wb, &events_loop).unwrap()
        };
        let size = window.get_inner_size().map(Into::into).unwrap_or((WIDTH as f64, HEIGHT as f64));

        let surface = back::Surface::from_window(window);
        let adapters = surface.enumerate_adapters();
        (adapters, surface, size)
    };

    let adapter = adapters.remove(0);

    let start = bsp::BspFile::parse(
        &mut Cursor::new(pak.file("maps/start.bsp").unwrap())
    ).unwrap();

    let mut renderer = render::Renderer::new(
        pak.clone(), start,
        adapter, surface,
        size,
    ).unwrap();

    let mut running = true;
    let mut moving_forward = false;
    let mut lock_mouse = false;
    let mut level_idx = 0;
    let mut last_frame = Instant::now();
    let mut display_size: (u32, u32) = (WIDTH, HEIGHT);

    let mut frames = 0;
    let mut last_fps = Instant::now();
    while running {
        let start = Instant::now();
        let diff = last_frame.elapsed();
        last_frame = start;
        let delta =
            (diff.as_secs() * 1_000_000_000 + diff.subsec_nanos() as u64) as f32 / (1_000_000_000.0 / 60.0);

        events_loop.poll_events(|event| {
            use winit::{Event, WindowEvent, VirtualKeyCode, ElementState, MouseButton};

            #[cfg(feature = "gl")]
            let window = renderer.surface.window().window();
            #[cfg(not(feature = "gl"))]
            let window = &window;

            let (width, height): (f64, f64) = window.get_inner_size().unwrap().into();

            match event {
                Event::WindowEvent{event: WindowEvent::KeyboardInput{input:key, ..}, ..} => {
                    if key.virtual_keycode == Some(VirtualKeyCode::Escape) && key.state == ElementState::Released {
                        lock_mouse = !lock_mouse;
                        if lock_mouse {
                            window.hide_cursor(true);
                        } else {
                            window.hide_cursor(false);
                        }
                    }
                    if key.virtual_keycode == Some(VirtualKeyCode::W) {
                        moving_forward = key.state == ElementState::Pressed;

                    } else if key.virtual_keycode == Some(VirtualKeyCode::P) && key.state == ElementState::Released {
                        level_idx = (level_idx + 1) % LEVELS.len();
                        let level = bsp::BspFile::parse(
                            &mut Cursor::new(pak.file(&format!("maps/{}.bsp", LEVELS[level_idx])).unwrap())
                        ).unwrap();
                        renderer.change_level(level).unwrap();
                    }
                },
                Event::WindowEvent{event: WindowEvent::MouseInput{state: ElementState::Pressed, button: MouseButton::Left, ..}, ..} => {
                    window.hide_cursor(true);
                    lock_mouse = true;
                },
                Event::WindowEvent{event: WindowEvent::CloseRequested, ..} => {
                    running = false;
                },
                Event::WindowEvent{event: WindowEvent::Resized(dims), ..} => {
                    display_size = (dims.width as u32, dims.height as u32);
                },
                Event::WindowEvent{event: WindowEvent::CursorMoved{position, ..}, ..} => {
                    if !lock_mouse {
                        return;
                    }
                    let position: (f64, f64) = position.into();
                    let dx = (width * 0.5) - position.0;
                    let dy = (height * 0.5) - position.1;

                    window.set_cursor_position((width / 2.0, height / 2.0).into()).unwrap();

                    renderer.camera.rot_x -= cgmath::Rad(dy as f32 / 2000.0);
                    renderer.camera.rot_y -= cgmath::Rad(dx as f32 / 2000.0);
                },
                _ => {},
            }
        });

        if moving_forward {
            renderer.camera.x += 5.0 * renderer.camera.rot_y.0.sin() * delta;
            renderer.camera.y += 5.0 * renderer.camera.rot_y.0.cos() * delta;
            renderer.camera.z -= 5.0 * (-renderer.camera.rot_x.0).sin() * delta;
        }

        renderer.draw(delta, display_size);

        frames += 1;
        if last_fps.elapsed() > Duration::from_secs(1) {
            println!("FPS: {}", frames);
            frames = 0;
            last_fps = Instant::now();
        }
    }
}

const LEVELS: &'static [&'static str] = &[
    "start",
    "e1m1",
    "e1m2",
    "e1m3",
    "e1m4",
    "e1m5",
    "e1m6",
    "e1m7",
    "e1m8",
];