

class Tokenizer:
    def __init__(self):
        pass

    def tokenize(self, *, text, text_type):
        if text_type == 'ph':
            return text.replace('%%', '').replace('_', '#').replace('   ', '#').replace(' ', '').split('#')
        else:
            return text.split()
