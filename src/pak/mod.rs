
use std::path::Path;
use std::io::{self, Read, Seek, SeekFrom};
use std::fs::File;
use std::collections::HashMap;
use std::cell::RefCell;

use crate::parse::*;
use crate::error;

pub struct PackFile {
    file: RefCell<File>,
    entries: HashMap<String, Entry>,
}

struct Entry {
    offset: u64,
    size: u64,
}

impl PackFile {
    // TODO: Fix error type
    pub fn new<P>(name: P) -> error::Result<PackFile>
        where P: AsRef<Path>
    {
        let mut f = File::open(name)?;

        let magic = read_string!(f, 4);

        if &magic != b"PACK" {
            bail!("Invalid pak magic");
        }

        let offset = f.read_long()?;
        let size = f.read_long()? / 0x40;
        f.seek(SeekFrom::Start(offset as u64))?;

        let mut entries = HashMap::default();

        for _ in 0 .. size {
            let name = read_string!(f, 0x38);
            let entry_offset = f.read_long()?;
            let entry_size = f.read_long()?;

            let name = from_cstring(&name)?;
            entries.insert(name, Entry {
                offset: entry_offset as u64,
                size: entry_size as u64,
            });
        }

        Ok(PackFile {
            file: RefCell::new(f),
            entries,
        })
    }

    pub fn file(&self, name: &str) -> io::Result<Vec<u8>> {
        let mut file = self.file.borrow_mut();
        if let Some(e) = self.entries.get(name) {
            file.seek(SeekFrom::Start(e.offset))?;
            let mut data = vec![0; e.size as usize];
            file.read_exact(&mut data)?;
            Ok(data)
        } else {
            Err(io::Error::new(io::ErrorKind::NotFound, "No such file in the pak"))
        }
    }
}