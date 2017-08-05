

pub struct TextureAtlas {
    free_rects: Vec<Rect>,
    padding: i32,
}

impl TextureAtlas {
    pub fn new(width: i32, height: i32) -> TextureAtlas {
        TextureAtlas {
            free_rects: vec![Rect{x: 0, y: 0, width: width, height: height}],
            padding: 0,
        }
    }

    pub fn new_padded(width: i32, height: i32, padding: i32) -> TextureAtlas {
        TextureAtlas {
            free_rects: vec![Rect{x: 0, y: 0, width: width, height: height}],
            padding: padding,
        }
    }

    pub fn find(&mut self, mut width: i32, mut height: i32) -> Option<Rect> {
        width += self.padding * 2;
        height += self.padding * 2;
        let mut best: Option<(i32, usize)> = None;
        for (idx, free) in self.free_rects.iter().enumerate() {
            let score = (free.width - width) * (free.height - height);
            // Will it fit the requested size and is it
            // a tighter fit than the previous match we found?
            if score >= 0
                && free.width >= width && free.height >= height
                && best.map_or(true, |v| v.0 > score) {
                best = Some((score, idx));
                if score == 0 {
                    // Found a perfect match
                    // no need to continue searching
                    break;
                }
            }
        }

        if let Some(best) = best {
            let mut rect = self.free_rects.remove(best.1);
            // Use the location of the match but our position.
            let ret = Rect {
                x: rect.x,
                y: rect.y,
                width: width,
                height: height,
            };

            // Split up the remaining space to reuse
            if rect.width - width > 0 {
                self.free_rects.push(Rect {
                    x: rect.x + width,
                    y: rect.y,
                    width: rect.width - width,
                    height: rect.height,
                });
                rect.width = width;
            }
            if rect.height - height > 0 {
                self.free_rects.push(Rect {
                    x: rect.x,
                    y: rect.y + height,
                    width: rect.width,
                    height: rect.height - height,
                });
            }

            Some(Rect {
                x: ret.x + self.padding,
                y: ret.y + self.padding,
                width: ret.width - self.padding*2,
                height: ret.height - self.padding*2,
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}
