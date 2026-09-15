import sentencepiece as spm


class Tokenizer:
    def __init__(self, model_path: str = "data/tokenizer/phase1.model"):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    @classmethod
    def train(cls, input_file: str, vocab_size: int = 32000, model_prefix: str = "data/tokenizer/unrealistic"):
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=0.9995,
            byte_fallback=True,
            max_sentence_length=4096,
            split_digits=True,
            allow_whitespace_only_pieces=True,
            remove_extra_whitespaces=False,
        )

    def encode(self, text: str) -> list[int]:
        return self.sp.EncodeAsIds(text)

    def decode(self, ids: list[int]) -> str:
        return self.sp.DecodeIds(ids)

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    @property
    def eos_token(self) -> int:
        return self.sp.eos_id()

    @property
    def bos_token(self) -> int:
        return self.sp.bos_id()
