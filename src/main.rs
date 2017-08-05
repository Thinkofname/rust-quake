extern crate byteorder;
extern crate cgmath;
#[macro_use]
extern crate error_chain;
#[macro_use]
extern crate gfx;
extern crate gfx_window_glutin;
extern crate glutin;
extern crate env_logger;

#[macro_use]
mod parse;
pub mod pak;
pub mod error;
pub mod render;
pub mod bsp;

use std::time::Instant;
use std::rc::Rc;
use std::io::Cursor;
use gfx::Device;
use glutin::{
    Event,
    WindowEvent,
    DeviceEvent,
    VirtualKeyCode,
    GlContext,
    ElementState,
    MouseButton,
};

use render::{
    ColorFormat,
    DepthFormat,
};

fn main() {
    env_logger::init().unwrap();
    let pak = Rc::new(pak::PackFile::new("id1/PAK0.PAK").unwrap());

    let mut events_loop = glutin::EventsLoop::new();
    let builder = glutin::WindowBuilder::new()
        .with_title("RQuake".to_string())
        .with_dimensions(640, 480);
    let context = glutin::ContextBuilder::new();

    let (window, mut device, mut factory, main_color, main_depth) =
        gfx_window_glutin::init::<ColorFormat, DepthFormat>(builder, context, &events_loop);

    let start = bsp::BspFile::parse(
        &mut Cursor::new(pak.file("maps/start.bsp").unwrap())
    ).unwrap();
    let mut renderer = render::Renderer::new(
        pak.clone(), start,
        &mut device, &mut factory,
        main_color, main_depth,
    ).unwrap();

    let mut running = true;
    let mut moving_forward = false;
    let mut lock_mouse = false;
    let mut level_idx = 0;
    let mut last_frame = Instant::now();

    let mut encoder = factory.create_command_buffer().into();

    while running {
        let start = Instant::now();
        let diff = last_frame.elapsed();
        last_frame = start;
        let delta =
            (diff.as_secs() * 1_000_000_000 + diff.subsec_nanos() as u64) as f32 / (1_000_000_000.0 / 60.0);

        let (width, height) = window.get_inner_size().unwrap();
        events_loop.poll_events(|event| {
            match event {
                Event::WindowEvent{event: WindowEvent::KeyboardInput{input:key, ..}, ..} => {
                    if key.virtual_keycode == Some(VirtualKeyCode::Escape) && key.state == ElementState::Released {
                        lock_mouse = !lock_mouse;
                        if lock_mouse {
                            window.set_cursor_state(glutin::CursorState::Hide).unwrap();
                        } else {
                            window.set_cursor_state(glutin::CursorState::Normal).unwrap();
                        }
                    }
                    if key.virtual_keycode == Some(VirtualKeyCode::W) {
                        moving_forward = key.state == ElementState::Pressed;

                    } else if key.virtual_keycode == Some(VirtualKeyCode::P) && key.state == ElementState::Released {
                        level_idx = (level_idx + 1) % LEVELS.len();
                        let level = bsp::BspFile::parse(
                            &mut Cursor::new(pak.file(&format!("maps/{}.bsp", LEVELS[level_idx])).unwrap())
                        ).unwrap();
                        renderer.change_level(level, &mut device, &mut factory).unwrap();
                    }
                },
                Event::WindowEvent{event: WindowEvent::MouseInput{state: ElementState::Pressed, button: MouseButton::Left, ..}, ..} => {
                    window.set_cursor_state(glutin::CursorState::Hide).unwrap();
                    lock_mouse = true;
                },
                Event::WindowEvent{event: WindowEvent::Closed, ..} => {
                    running = false;
                },
                Event::WindowEvent{event: WindowEvent::MouseMoved{position, ..}, ..} => {
                    if !lock_mouse {
                        return;
                    }
                    let dx = (width as f64 * 0.5) - position.0;
                    let dy = (height as f64 * 0.5) - position.1;

                    window.set_cursor_position(width as i32 / 2, height as i32 / 2).unwrap();

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

        gfx_window_glutin::update_views(&window, &mut renderer.main_color, &mut renderer.main_depth);

        renderer.draw(delta, width, height, &mut device, &mut factory, &mut encoder);
        encoder.flush(&mut device);

        window.swap_buffers().unwrap();
        device.cleanup();
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