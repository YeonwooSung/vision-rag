import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available


class ColQwen2_5Loader:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self._dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        self._attn_implementation = (
            "flash_attention_2" if is_flash_attn_2_available() else None
        )

    def load(self) -> tuple[ColQwen2_5, ColQwen2_5_Processor]:
        model = self.load_model()
        processor = self.load_processor()
        return model, processor

    def load_model(self) -> ColQwen2_5:
        model = ColQwen2_5.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            device_map=self._device,
            torch_dtype=self._dtype,
            attn_implementation=self._attn_implementation,
        ).eval()
        return model

    def load_processor(self) -> ColQwen2_5_Processor:
        processor = ColQwen2_5_Processor.from_pretrained(
            pretrained_model_name_or_path=self.model_name
        )
        assert isinstance(processor, ColQwen2_5_Processor)
        return processor
