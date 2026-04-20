import enum


class datasets(enum.Enum):
    LIBRITTS = "Libri-TTS"
    MAESTRO = "Maestro"
    MUSICNET = "MusicNet"


class tokenizers(enum.Enum):
    WAVTOKENIZER = "wavtokenizer"
    UNICODEC = "unicodec"
    SPEECHTOKENIZER = "speechtokenizer"


class data_paths(enum.Enum):
    LIBRITTS_TRAIN_WAVTOKENIZER = "data/LibriTTS/LibriTTS/train-clean-360-tokens/"
    LIBRITTS_TEST_WAVTOKENIZER = "data/LibriTTS/LibriTTS/test-clean-tokens/"
    MAESTRO_CSV = "data/Maestro/maestro-v3.0.0.csv"
    MAESTRO_TOKENS_WAVTOKENIZER = "data/Maestro/tokens/"
    MAESTRO_TOKENS_UNICODEC = "data/Maestro/tokens_unicodec/"
    MUSICNET_WAVTOKENIZER = "data/MusicNet/Train/"
    MUSICNET_UNICODEC = "data/MusicNet/unicodec_tokens"
