"""
Machine Translation module for English → French.
Uses NLLB-200 for translation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List


class MachineTranslator:
    """
    Machine translator for English → French using NLLB-200.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        src_lang: str = "eng_Latn",
        tgt_lang: str = "fra_Latn",
        device: str = "cuda",
        max_length: int = 256,
        num_beams: int = 5
    ):
        """
        Initialize translator.
        
        Args:
            model_name: Model name
            src_lang: Source language code
            tgt_lang: Target language code
            device: Device to use
            max_length: Max sequence length
            num_beams: Beam search size
        """
        print(f"Loading translation model: {model_name}")
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        self.num_beams = num_beams
        
        # Set source language for tokenizer
        self.tokenizer.src_lang = src_lang
        
        print(f"✓ Translation model loaded on {self.device}")
        print(f"  Translation: {src_lang} → {tgt_lang}")
    
    @torch.no_grad()
    def translate(
        self,
        texts: List[str],
        batch_size: int = 8
    ) -> List[str]:
        """
        Translate a batch of texts.
        
        Args:
            texts: List of source texts
            batch_size: Batch size for processing
            
        Returns:
            translations: List of translated texts
        """
        if not texts:
            return []
        
        translations = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            # Generate
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang),
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True
            )
            
            # Decode
            batch_translations = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )
            
            translations.extend(batch_translations)
        
        return translations
    
    def translate_single(self, text: str) -> str:
        """
        Translate a single text.
        
        Args:
            text: Source text
            
        Returns:
            translation: Translated text
        """
        return self.translate([text])[0]
    
    def translate_streaming(self, text_buffer: str) -> str:
        """
        Translate accumulated text (for streaming).
        
        Args:
            text_buffer: Accumulated source text
            
        Returns:
            translation: Translated text
        """
        # For sentence-level translation
        # Split by sentence boundaries
        sentences = []
        current = ""
        
        for char in text_buffer:
            current += char
            if char in ['.', '!', '?']:
                sentences.append(current.strip())
                current = ""
        
        if current.strip():
            sentences.append(current.strip())
        
        # Translate sentences
        if sentences:
            translations = self.translate(sentences)
            return " ".join(translations)
        
        return ""


def translate_text(
    text: str,
    src_lang: str = "eng_Latn",
    tgt_lang: str = "fra_Latn",
    model_name: str = "facebook/nllb-200-distilled-600M",
    device: str = "cuda"
) -> str:
    """
    Translate text (convenience function).
    
    Args:
        text: Source text
        src_lang: Source language code
        tgt_lang: Target language code
        model_name: Model name
        device: Device to use
        
    Returns:
        translation: Translated text
    """
    translator = MachineTranslator(
        model_name=model_name,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        device=device
    )
    
    return translator.translate_single(text)


if __name__ == "__main__":
    # Test translation
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Machine Translation")
    parser.add_argument("--text", type=str, required=True, help="Text to translate")
    parser.add_argument("--src", type=str, default="eng_Latn", help="Source language")
    parser.add_argument("--tgt", type=str, default="fra_Latn", help="Target language")
    parser.add_argument("--model", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Translate
    translation = translate_text(
        text=args.text,
        src_lang=args.src,
        tgt_lang=args.tgt,
        model_name=args.model,
        device=args.device
    )
    
    print(f"\nSource ({args.src}): {args.text}")
    print(f"Translation ({args.tgt}): {translation}")
