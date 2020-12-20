import re
from xml_parser import XMLParser


def tokenizer(text):
    word_pattern = re.compile(r'''([0-9]+|[-/,\[\]{}`~!@#$%\^&*()_\+=:;"'?])|(\.) |(\.$)|([a-z]'[a-z])| ''')
    tokens = [token for token in word_pattern.split(text) if token]
    return tokens


def printAnnotations(annotations, tokens):
    for tag, bounds in annotations.items():
        heading = f"| {tag}"
        print( " "+"="*(len(list(heading))) )
        print(heading+" |")
        print( " "+"="*(len(list(heading))) )
        print()
        for beg,end in bounds:
            print(tokens[beg:end+1])
        print()


txt='''
<document>  
    Page 1 REPORTABLE
    <sent> Hi, how are you? <law>section <section> 302 </section> of <act> ipc </act> </law></sent>
    <sent> I am good <law> section <section> 3 </section> of <act> crpc </act> </law> </sent>
    J. R. Nair Page 3<as>
</document>
'''.replace("\n", " ").replace("\t", " ")

parser = XMLParser(tokenizer)

tokens, annotations = parser.parse(txt)

print(f"\nAnnotations => {dict(annotations)}\n")

printAnnotations(annotations, tokens)