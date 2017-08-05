
error_chain! {
    types {
        Error, ErrorKind, ResultExt, Result;
    }

    links {
    }

    foreign_links {
        Io(::std::io::Error);
        Str(::std::str::Utf8Error);
        Gfx(::gfx::CombinedError);
        GfxTexture(::gfx::texture::CreationError);
        GfxRView(::gfx::ResourceViewError);
        GfxPipeline(::gfx::PipelineStateError<String>);
        GfxProgram(::gfx::shade::ProgramError);
    }

    errors {

    }
}