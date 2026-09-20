use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("io error: {0}")]
    PlainIo(#[from] std::io::Error),

    #[error("config error: {0}")]
    Config(String),

    #[error("missing config key `{0}`")]
    MissingKey(String),

    #[error("config key `{key}` has wrong type: {detail}")]
    KeyType { key: String, detail: String },

    #[error("checkpoint error: {0}")]
    Checkpoint(String),

    #[error("pickle error: {0}")]
    Pickle(String),

    #[error("weight `{0}` not found in checkpoint")]
    MissingWeight(String),

    #[error("weight `{name}` shape mismatch: checkpoint {found:?}, model expects {expected:?}")]
    WeightShape {
        name: String,
        found: Vec<usize>,
        expected: Vec<usize>,
    },

    #[error("unsupported model type `{0}`")]
    UnsupportedModel(String),

    #[error("audio error: {0}")]
    Audio(String),

    #[error("shape error: {0}")]
    Shape(String),

    #[error("gpu error: {0}")]
    Gpu(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub trait IoContext<T> {
    fn with_path(self, path: impl Into<PathBuf>) -> Result<T>;
}

impl<T> IoContext<T> for std::result::Result<T, std::io::Error> {
    fn with_path(self, path: impl Into<PathBuf>) -> Result<T> {
        self.map_err(|source| Error::Io {
            path: path.into(),
            source,
        })
    }
}
