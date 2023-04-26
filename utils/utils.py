# from pypapago import Translator as papago_trans
from googletrans import Translator

def back_translate(translator, text_org: str, 
                   src_lang: str = 'ko', dest_lang: str = 'en'):
    """기존 텍스트를 back translate.
    
    Args:
        translator: googletrans 혹은 pypapago 등 라이브러리의 Translator 인스턴스.
        text_org: back translation할 텍스트 (기본 언어: 한국어).
    
    Returns:
        text_back: back translation 결과 텍스트 (기본 언어: 한국어).
    """
    try:
        text_tgt = translator.translate(text_org, src=src_lang, dest=dest_lang).text
        text_back = translator.translate(text_tgt, src=dest_lang, dest=src_lang).text
    except KeyError:
        return 'Translation failed.'
    return text_back
