from eeve.data.formatters.base import AdvancedFormatter


class QuoteReplacer(AdvancedFormatter):
    """
    Formatter for Standardizing Quotation Marks

    Standardizes quotation marks in text by replacing them with chevron quotes (« »).
    In case of nested quotation marks, only replaces the outermost pair with chevron quotes, leaving the inner ones unchanged.
    """
    name = "💬 Quotes"

    def __init__(self, list_path: str | list[str] = "text"):
        super().__init__(list_path=list_path)

    def format(self, text: str) -> str:
        if not text:
            return text
        
        result = []
        quote_stack = []  
        i = 0
        
        while i < len(text):
            if text[i] == '"':
                if not quote_stack:
                    result.append('«')
                    quote_stack.append('guillemet')
                elif quote_stack[-1] == 'guillemet':
                    if i + 1 < len(text) and text[i + 1].isalpha():
                        result.append('"')
                        quote_stack.append('quote')
                    else:
                        result.append('»')
                        quote_stack.pop()
                elif quote_stack[-1] == 'quote':
                    result.append('"')
                    quote_stack.pop()
            else:
                result.append(text[i])
            i += 1
        
        return ''.join(result)